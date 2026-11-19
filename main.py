#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster v2 — earners x players decomposition with a
refresh-aware, backtest-calibrated spread.

Median: forecast earners and players separately, then divide. log(players) is
fit against ln(1 + hours since release); each trophy's log(earners) is fit the
same way with its slope shrunk 30% toward the players slope (a trophy's earners
cannot outgrow the player base forever). The median is anchored to the last
read and capped to 3 logit units of movement.

Spread (the v2 fix): the total logit sigma combines, in quadrature, the direct
log-rate fit's residual error, a random walk growing with the horizon, a
refresh-shock term (Exophase refreshes move rates in jumps; when the history
is frozen the next refresh is unobserved, so the band must span it), and a
disagreement penalty between the decomposition median and the direct-trend
median. A global multiplier k calibrates the whole band so the 90% interval
covers about 90% in a synthetic backtest whose dynamics differ from this
model's (logistic adoption, per-player earn times, Poisson refreshes).

Stdlib only, no network, no clock, deterministic: two runs are byte-identical.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# --- Tuned constants (synthetic backtest, 48 cases, 3 scenarios, FIXED
#     seeds — reproducible across runs; see the builder's scratch design.md
#     + backtest.py). Stage-1 sweep (SIGMA_RW x CAL_K x SIGMA_SHOCK_FROZEN)
#     then stage 2 scored on the 4.5-day kinds only — the final IS a 4.5-day
#     case; the frozen 90-minute kind is the practice regime and does not
#     pay. Pick: SIGMA_RW 0.15 (0.10/0.15/0.20 differ by <0.005 loss),
#     CAL_K 1.50, shock 2.2: 4.5d pinball loss 0.7206, coverage_90 0.911
#     (baselines pooled: house 1.92, house-curve 1.36, persistence 2.49).
#     k=1.25 ties on 4.5d loss (0.7210) but under-covers (0.826); k=1.50
#     matches the loss and lands coverage on the 0.90 target, and it was
#     the robust pick across two independent game draws (cov 0.911 / 0.931).
#     Loss rises past k=1.50 (k=2.00 -> 0.7596), so 1.50 is an interior
#     optimum. The frozen shock helps 4.5d loss monotonically to 2.2 (rare
#     trophies frozen for 48h genuinely jump); gains past 2.2 are within
#     noise, so the sweep stops there.
SIGMA_RW = 0.15            # random-walk sigma per sqrt(day)
SIGMA_FIT_DEFAULT = 0.35   # fit-error sigma when too few points to fit
SIGMA_SHOCK_FROZEN = 2.2   # refresh-shock sigma, frozen/thin history
SIGMA_SHOCK_MOVING = 0.3   # refresh-shock sigma, history shows movement
SIGMA_FLOOR = 0.10         # never claim more certainty than this
SIGMA_TOP = 0.35           # spread for a trophy sitting at 100%
CAL_K = 1.50               # global calibration multiplier on sigma
SLOPE_KEEP = 0.7           # keep 70% of a trophy slope's deviation (30% shrink)
MOVE_CAP = 3.0             # max median move from last read, logit units
FIT_WINDOW_HOURS = 48.0    # only the last 48h of reads are fitted
MAX_PLAYER_GROWTH_LOG = 3.0  # cap on log player growth over the horizon


def parse_time(text):
    """ISO 8601 (with Z) -> aware datetime, or None."""
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def logit_pct(rate):
    """logit of a percent rate, clamped to [0.01, 99.99] like the scorer."""
    p = min(99.99, max(0.01, float(rate))) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_sd, n) or None when
    the fit is undefined (fewer than 2 distinct x). resid_sd is Bessel
    corrected (n-2 dof) and None when n < 3."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sd = None
    if n >= 3:
        rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        sd = math.sqrt(rss / (n - 2))
    return (a, b, sd, n)


def clean_reads(history, origin, checkpoint):
    """Dedupe identical consecutive reads, thin to at most one read per hour
    (keep the latest in each hour bucket), keep only the last 48h before the
    checkpoint. Returns a list of (x, read) with x = ln(1 + hours since
    origin), oldest first."""
    rows = []
    prev_key = None
    for read in history:
        at = parse_time(read.get("at") or read.get("slot"))
        if at is None:
            continue
        players = read.get("players_tracked")
        rates = read.get("rates")
        if not isinstance(players, (int, float)) or players <= 0:
            continue
        if not isinstance(rates, list):
            continue
        key = (players, tuple(rates))
        if key == prev_key:
            continue  # Exophase had not refreshed yet: same page again
        prev_key = key
        hours = (at - origin).total_seconds() / 3600.0
        if hours < 0:
            hours = 0.0
        rows.append((hours, read))
    # Last 48h before the checkpoint.
    if checkpoint is not None:
        rows = [r for r in rows
                if (checkpoint - origin).total_seconds() / 3600.0 - r[0]
                <= FIT_WINDOW_HOURS]
    # One read per hour bucket: keep the latest in each bucket.
    by_hour = {}
    for hours, read in rows:
        by_hour[int(hours)] = (hours, read)
    out = []
    for hours in sorted(by_hour):
        h, read = by_hour[hours]
        out.append((math.log1p(h), read))
    return out


def forecast_case(case):
    """Forecast one case dict -> {"trophies": [{"id": n, "q": [...]}, ...]}."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    checkpoint = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    horizon_days = None
    hm = case.get("horizon_minutes")
    if isinstance(hm, (int, float)) and hm > 0:
        horizon_days = hm / 1440.0
    if horizon_days is None and checkpoint is not None and target is not None:
        horizon_days = (target - checkpoint).total_seconds() / 86400.0
    if horizon_days is None or horizon_days <= 0:
        horizon_days = 1.0

    game = case.get("game") or {}
    origin = None
    rel = game.get("release_date")
    if isinstance(rel, str):
        origin = parse_time(rel + "T00:00:00Z") if "T" not in rel else parse_time(rel)
    if origin is None and history:
        origin = parse_time(history[0].get("at") or history[0].get("slot"))
    if origin is None:
        origin = checkpoint or datetime(2026, 11, 19, tzinfo=timezone.utc)

    if target is not None:
        x_target = math.log1p(max(0.0, (target - origin).total_seconds() / 3600.0))
    else:
        x_target = math.log1p(max(0.0, (checkpoint - origin).total_seconds() / 3600.0)
                              + horizon_days * 24.0)

    cleaned = clean_reads(history, origin, checkpoint)

    # --- Players fit: log(N) vs x.
    n_last = None
    if history:
        for read in reversed(history):
            p = read.get("players_tracked")
            if isinstance(p, (int, float)) and p > 0:
                n_last = float(p)
                break
    if n_last is None:
        n_last = 1.0
    n_hat = n_last
    b_players = 0.0
    players_fit = None
    if len(cleaned) >= 2:
        xs = [x for x, _ in cleaned]
        ys = [math.log(float(r["players_tracked"])) for _, r in cleaned]
        players_fit = ols(xs, ys)
    if players_fit is not None:
        a_n, b_n, _, _ = players_fit
        b_players = b_n
        log_n_hat = a_n + b_n * x_target
        log_n_hat = max(math.log(n_last),
                        min(log_n_hat, math.log(n_last) + MAX_PLAYER_GROWTH_LOG))
        n_hat = math.exp(log_n_hat)

    rows = []
    for idx, trophy in enumerate(trophies):
        tid = trophy.get("id", idx + 1)
        # Last observed rate for this trophy.
        r_last = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                r_last = float(rates[idx])
                break
        if r_last is None:
            r_last = 0.0

        # Per-trophy series over the cleaned reads.
        series = []  # (x, rate, players)
        for x, read in cleaned:
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                series.append((x, float(rates[idx]),
                               float(read["players_tracked"])))
        distinct_rates = len(set(r for _, r, _ in series))

        # --- Earners fit with slope shrinkage toward the players slope.
        e_last = max(0.5, r_last / 100.0 * n_last)
        x_last = series[-1][0] if series else 0.0
        b_earn = 0.0
        if len(series) >= 2:
            xs = [x for x, _, _ in series]
            ys = [math.log(max(0.5, r / 100.0 * n)) for _, r, n in series]
            fit_e = ols(xs, ys)
            if fit_e is not None:
                b_earn = b_players + SLOPE_KEEP * (fit_e[1] - b_players)
        e_hat = math.exp(math.log(e_last) + b_earn * (x_target - x_last))
        e_hat = min(e_hat, n_hat)
        median = 100.0 * e_hat / n_hat if n_hat > 0 else r_last

        # Movement cap: at most MOVE_CAP logit units from the last read.
        lo = logit_pct(r_last) - MOVE_CAP
        hi = logit_pct(r_last) + MOVE_CAP
        median = 100.0 * sigmoid(min(hi, max(lo, logit_pct(median))))
        median = min(99.99, max(0.01, median))

        # --- Direct log-rate fit: residual error + disagreement.
        sigma_fit = SIGMA_FIT_DEFAULT
        m_direct = r_last
        if len(series) >= 2:
            xs = [x for x, _, _ in series]
            ys = [logit_pct(r) for _, r, _ in series]
            fit_d = ols(xs, ys)
            if fit_d is not None:
                a_d, b_d, sd_d, _ = fit_d
                m_direct = 100.0 * sigmoid(a_d + b_d * x_target)
                if sd_d is not None:
                    sigma_fit = sd_d
        sigma_disagree = abs(logit_pct(median) - logit_pct(m_direct)) / 2.0

        # --- Refresh shock: frozen history means the next refresh is unseen.
        if distinct_rates < 3:
            sigma_shock = SIGMA_SHOCK_FROZEN
        else:
            sigma_shock = SIGMA_SHOCK_MOVING
        sigma_rw = SIGMA_RW * math.sqrt(horizon_days)

        sigma = CAL_K * math.sqrt(sigma_fit ** 2 + sigma_rw ** 2
                                  + sigma_shock ** 2 + sigma_disagree ** 2)
        sigma = max(SIGMA_FLOOR, sigma)

        # Trophy sitting at 100%: it stayed at 100% through the big launch
        # refresh, so keep the median pinned with a small spread.
        if r_last >= 99.99:
            median = 99.99
            sigma = SIGMA_TOP

        centre = logit_pct(median)
        qs = []
        prev = -1.0
        for z in Z:
            q = round(100.0 * sigmoid(centre + z * sigma), 4)
            q = min(100.0, max(0.0, q))
            if q < prev:
                q = prev
            prev = q
            qs.append(q)
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def persistence_fallback(case):
    """Valid-but-dumb forecast: every quantile equals the last read's rate.
    Only used if the main forecast raises; a valid case beats a crash."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    rows = []
    for idx, trophy in enumerate(trophies):
        r_last = 0.0
        for read in reversed(history):
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                r_last = float(rates[idx])
                break
        r_last = min(100.0, max(0.0, r_last))
        rows.append({"id": trophy.get("id", idx + 1), "q": [r_last] * 5})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        path = os.path.join(in_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
        except (OSError, ValueError):
            continue
        try:
            result = forecast_case(case)
        except Exception:
            try:
                result = persistence_fallback(case)
            except Exception:
                continue
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, separators=(",", ":"))
                fh.write("\n")
        except OSError:
            continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
