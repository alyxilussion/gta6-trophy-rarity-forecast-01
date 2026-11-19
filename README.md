[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alyxilussion/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alyxilussion/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecaster — v2

Entry for the [make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26 00:00 UTC, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy, scored by pinball loss on the logit scale.

v2 is a fresh rebuild of the v1 entry. v1's median was sound but its intervals were far too narrow — practice coverage_90 was 0.018. v2 keeps the proven median and rebuilds the uncertainty calibration from scratch.

## The model

An earn rate is a ratio — earners / players tracked — and both sides move in launch week: new players join Exophase tracking (diluting every rate) while the players already there earn more trophies (lifting rates). Forecasting the ratio directly confounds the two, so `main.py` forecasts each side separately and divides:

- **Players tracked**: `log(N)` fitted by least squares against `ln(1 + hours since release)` over the recent reads, read at the target.
- **Earners per trophy**: `E = rate x N / 100` fitted the same way, with its slope shrunk 30% toward the players' slope (rates stabilise; a trophy's earners cannot keep outgrowing the player base). The forecast is anchored to the last read: `log(E_hat) = log(E_last) + slope x (x_target - x_last)`.
- **Median rate**: `100 x E_hat / N_hat`, capped so earners never exceed players and the move from the last read stays within 3 logit units.

The spread is where v2 differs. The total logit sigma combines, in quadrature: the direct log-rate trend fit's residual error, a random walk growing with the horizon (σ = 0.15 per √day), a **refresh-shock term** — Exophase refreshes move rates in jumps, and when a trophy's history is frozen the next refresh is unobserved, so frozen trophies get a wide shock band (σ = 2.2) while moving trophies get a small one (σ = 0.3) — and a disagreement penalty between the ratio median and the direct-trend median. A global multiplier (k = 1.50) calibrates the whole band so the 90% interval covers about 90% in backtest. A trophy sitting at 100% stays pinned there with a small spread: it survived the big launch-night refresh unchanged.

Reads are cleaned before fitting: consecutive identical reads (Exophase had not refreshed) count once, at most one read per hour is kept, and only the last 48 hours are fitted.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. No network, no model calls, deterministic — two runs on the same case write the same bytes (verified). One case in, one `<id>.out` file out: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`. Runs in ~0.05 s per case, far inside the 8 s limit. A grader simulation (uid 65534, read-only repo) produces byte-identical output.

## Backtest

A synthetic launch week with dynamics deliberately different from the forecaster's (logistic adoption + daily seasonality, per-player lognormal earn times convolved with join times, Poisson Exophase refreshes, 2-decimal rounding) was used to validate the entry end to end: 48 cases across 3 player-growth scenarios, fixed seeds, reproducible. On the 4.5-day cases — the final's shape — v2 scores pinball loss **0.7206**, 90% coverage **0.911**, typical miss 9.19 points, against **1.6764** / 0.624 for the house curve baseline, **2.4945** / 0.076 for the house baseline, and **3.2532** / 0.003 for pure persistence. v2 beats the house curve baseline in 32 of 36 cases and the house baseline in 36 of 36. The spread constants (k, shock) were picked by a two-stage sweep scored on the 4.5-day cases only; the frozen 90-minute kind is the practice regime and does not pay.

As a second check, the documented launch-night refresh (players 638 → 3,868) was replayed onto the real practice case p00: v2 scores pinball **0.91–0.99** with coverage **0.982** across earners-growth assumptions, against the house baseline's 1.51 / 0.000 on the same shock (matching the rules' stated house practice numbers) and v1's real practice 1.29 / 0.018.

Honest limits: the final is one real event, and the house model forecast (superbot, made 21 Nov 09:30 UTC on the full data) is an unmeasurable wildcard — beating it cannot be proven in advance. The backtest says v2 clears both house baselines by a wide margin; whether it clears the house model is decided on 26 Nov.

## License

MIT — see [LICENSE](LICENSE).
