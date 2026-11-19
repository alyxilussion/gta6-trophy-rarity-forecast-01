"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Model, in plain words
--------------------
An earn rate is a ratio: earners / players tracked. Both sides move in launch
week -- new players join Exophase tracking (diluting every rate) while the
players already there earn more trophies (lifting rates). Forecasting the
ratio directly confounds the two, so this program forecasts each side
separately and divides:

  * players tracked: log(N) is fitted by least squares against
    ln(1 + hours since release) over the recent reads, and read at the
    target;
  * earners per trophy: E = rate x N / 100 is fitted the same way, with its
    slope shrunk 30% of the way toward the players' slope (rates stabilise;
    a trophy's earners cannot keep outgrowing the player base for long);
  * the median rate is 100 x E_hat / N_hat, capped so earners never exceed
    players and the move from the last read stays within 3 logit units.

The spread is honest about what a multi-day extrapolation does not know: the
prediction error of a direct log-rate trend fit (which sees the same scatter
the decomposition does), a random-walk term that grows with the horizon, a
term for disagreement between the decomposition median and the direct-trend
median, and a floor. Trophies with too little history fall back to
persistence with a wider band; a frozen history (the practice cases) falls
back to persistence entirely.

Reads are cleaned before fitting: consecutive identical reads (Exophase
had not refreshed) count once, and at most one read per hour is kept, so a
10-minute recorder does not pretend to be 6 independent observations an
hour. Only the last 48 hours of reads are fitted.

Everything is deterministic standard-library arithmetic: two runs on the
same case write the same bytes. No network, no model calls, no randomness.

Run: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Grader: python3 -P -s /bot/main.py, one case in /input, forecast to /output.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Release day, the origin of the log-time axis (same origin the house curve
# baseline uses, so the two models are comparable).
ORIGIN = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Fit window: only reads this recent (hours before the checkpoint) are used.
WINDOW_HOURS = 48.0
# Thinning: at most one read per hour bucket is kept for fitting.
THIN_MINUTES = 60.0
# Minimum reads for a least-squares fit (fewer: fall back to persistence).
MIN_POINTS = 3
# Slope shrinkage for earners: b* = SHRINK * b_fit + (1 - SHRINK) * b_players.
SHRINK = 0.7
# Random-walk process noise, in logit units per square-root day. Set to 0.15
# after a synthetic backtest (three launch-week scenarios): 0.20 was
# measurably over-wide (90% coverage 0.96), 0.10 scored better still in the
# simulator, but the simulator is tamer than the real week (no weekend
# surge, no Exophase quirks), so 0.15 keeps a margin for genuine shocks.
SIGMA_RW = 0.15
# Minimum spread, in logit units: no forecast claims to know more than this.
SIGMA_FLOOR = 0.10
# Maximum spread: bands wider than this pay more in width than they save.
SIGMA_CAP = 1.20
# Sanity cap on how far the median may move from the last read, logit units.
MAX_LOGIT_MOVE = 3.0
# Extra spread (added in quadrature) for a trophy with 1-2 positive reads.
SIGMA_THIN_DATA = 0.30
# Extra spread for a trophy with no positive read at all (it may emerge).
SIGMA_NO_DATA = 0.50


def clamp_rate(x: float) -> float:
    """A percent clamped to the range the scorer's logit is defined on."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    """Logit back to percent, guarded against overflow either way."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up (as the house baseline rounds it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(t: datetime) -> float:
    """ln(1 + hours since release), floored so pre-release reads stay valid."""
    hours = (t - ORIGIN).total_seconds() / 3600.0
    return math.log(max(1.0 + hours, 0.01))


def ols(xs, ys):
    """Least squares y = a + b x. Returns (a, b, s2, xbar, sxx, n) or None.

    s2 is the residual variance (RSS / (n - 2)); None when the fit is
    degenerate (too few points or no spread in x).
    """
    n = len(xs)
    if n < MIN_POINTS:
        return None
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    s2 = rss / (n - 2) if n > 2 else 0.0
    return (a, b, s2, xbar, sxx, n)


def pred_se(fit, x_star: float) -> float:
    """Prediction standard error of a fitted line at x_star.

    sqrt(s2 * (1 + 1/n + (x* - xbar)^2 / Sxx)): the usual prediction
    interval, which widens as the target moves away from the fitted data.
    """
    a, b, s2, xbar, sxx, n = fit
    if s2 <= 0.0:
        s2 = 1e-6
    return math.sqrt(s2 * (1.0 + 1.0 / n + (x_star - xbar) ** 2 / sxx))


def clean_reads(history, checkpoint: datetime):
    """Dedupe, thin and window the reads; returns [(time, players, rates)]."""
    rows = []
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates")
        if players is None or players <= 0 or rates is None:
            continue
        try:
            when = parse_time(read["at"])
        except (KeyError, ValueError):
            continue
        rows.append((when, players, tuple(rates)))
    rows.sort(key=lambda r: r[0])
    # Consecutive identical reads are one observation (no refresh happened).
    deduped = []
    for row in rows:
        if deduped and deduped[-1][1] == row[1] and deduped[-1][2] == row[2]:
            continue
        deduped.append(row)
    # At most one read per hour: the last one in each hour bucket.
    thinned = {}
    for row in deduped:
        bucket = int(row[0].timestamp() // (THIN_MINUTES * 60))
        thinned[bucket] = row
    thinned = [thinned[k] for k in sorted(thinned)]
    # The recent window; fall back to everything if it leaves too little.
    cutoff = checkpoint.timestamp() - WINDOW_HOURS * 3600.0
    windowed = [r for r in thinned if r[0].timestamp() >= cutoff]
    return windowed if len(windowed) >= MIN_POINTS else thinned


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    rw = SIGMA_RW * math.sqrt(max(horizon_days, 0.0))

    if not history:
        # No data at all: a valid but maximally uncertain forecast.
        rows = []
        for trophy in trophies:
            rows.append({"id": trophy["id"], "q": [round4(inv_logit(0.0 + z * SIGMA_CAP)) for z in Z]})
        return {"trophies": rows}

    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    x_star = log_time(target)
    reads = clean_reads(history, checkpoint)
    last = history[-1]
    last_rates = last.get("rates") or []

    # --- players model: log(N) against log-time -------------------------
    players_fit = None
    n_hat = None
    se_n = None
    if len(reads) >= MIN_POINTS:
        xs = [log_time(r[0]) for r in reads]
        ys = [math.log(r[1]) for r in reads]
        players_fit = ols(xs, ys)
        if players_fit is not None:
            a, b, _, _, _, _ = players_fit
            n_hat = math.exp(a + b * x_star)
            # Players tracked never falls, and will not octuple in a week.
            n_hat = min(max(n_hat, reads[-1][1]), reads[-1][1] * 8.0)
            se_n = pred_se(players_fit, x_star)
    if n_hat is None:
        n_hat = reads[-1][1] if reads else last.get("players_tracked") or 1
    b_players = players_fit[1] if players_fit is not None else 0.0

    # --- per-trophy forecasts -------------------------------------------
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last_rates[i] if i < len(last_rates) else None
        if last_rate is None:
            last_rate = 0.0
        last_logit = logit(last_rate)

        # Positive reads for this trophy: (x, earners, rate).
        points = []
        for when, players, rates in reads:
            if i >= len(rates) or rates[i] is None or rates[i] <= 0:
                continue
            points.append((log_time(when), rates[i] * players / 100.0, rates[i]))
        n_pos = len(points)

        median_logit = last_logit
        se_rate = None
        direct_logit = None
        extra2 = 0.0

        if n_pos >= MIN_POINTS and players_fit is not None:
            xs = [p[0] for p in points]
            # Earners fit, slope shrunk toward the players' slope, line
            # re-anchored through the data's centre.
            ys_e = [math.log(p[1]) for p in points]
            fit_e = ols(xs, ys_e)
            if fit_e is not None:
                _, b_e, _, xbar_e, _, _ = fit_e
                ybar_e = sum(ys_e) / len(ys_e)
                b_star = SHRINK * b_e + (1.0 - SHRINK) * b_players
                a_star = ybar_e - b_star * xbar_e
                e_hat = math.exp(a_star + b_star * x_star)
                # Earners never decrease and never exceed the players.
                e_max = max(p[1] for p in points)
                e_hat = min(max(e_hat, e_max), 0.999 * n_hat)
                rate_hat = 100.0 * e_hat / n_hat
                median_logit = logit(rate_hat)
            # Direct log-rate fit: its prediction error calibrates the
            # spread, and its median flags disagreement with the ratio.
            ys_r = [math.log(p[2]) for p in points]
            fit_r = ols(xs, ys_r)
            if fit_r is not None:
                a_r, b_r, _, _, _, _ = fit_r
                se_rate = pred_se(fit_r, x_star)
                direct_logit = logit(math.exp(a_r + b_r * x_star))
        elif n_pos > 0:
            extra2 = SIGMA_THIN_DATA ** 2
        else:
            extra2 = SIGMA_NO_DATA ** 2

        # The median may not run away from the last read.
        median_logit = min(max(median_logit, last_logit - MAX_LOGIT_MOVE),
                           last_logit + MAX_LOGIT_MOVE)

        # --- spread ------------------------------------------------------
        var = SIGMA_FLOOR ** 2 + rw ** 2 + extra2
        if se_rate is not None:
            var += se_rate ** 2
        elif se_n is not None:
            var += se_n ** 2 + 0.09  # players error plus an assumed earner error
        else:
            var += 0.09
        if direct_logit is not None:
            var += (0.25 * abs(median_logit - direct_logit)) ** 2
        sigma = min(math.sqrt(var), SIGMA_CAP)

        rows.append({
            "id": trophy["id"],
            "q": [round4(inv_logit(median_logit + z * sigma)) for z in Z],
        })
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
