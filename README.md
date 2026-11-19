[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alyxilussion/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alyxilussion/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecaster

Entry for the [make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26 00:00 UTC, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy, scored by pinball loss on the logit scale.

## The model

An earn rate is a ratio — earners / players tracked — and both sides move in launch week: new players join Exophase tracking (diluting every rate) while the players already there earn more trophies (lifting rates). Forecasting the ratio directly confounds the two, so `main.py` forecasts each side separately and divides:

- **Players tracked**: `log(N)` fitted by least squares against `ln(1 + hours since release)` over the recent reads, read at the target.
- **Earners per trophy**: `E = rate x N / 100` fitted the same way, with its slope shrunk 30% toward the players' slope (rates stabilise; a trophy's earners cannot keep outgrowing the player base).
- **Median rate**: `100 x E_hat / N_hat`, capped so earners never exceed players and the move from the last read stays within 3 logit units.

The spread is honest about a multi-day extrapolation: the prediction error of a direct log-rate trend fit, a random-walk term growing with the horizon (σ = 0.15 per √day, calibrated on a synthetic backtest), a term for disagreement between the ratio median and the direct-trend median, and a floor. Trophies with too little history fall back to persistence with a wider band; a frozen history falls back to persistence entirely.

Reads are cleaned before fitting: consecutive identical reads (Exophase had not refreshed) count once, at most one read per hour is kept, and only the last 48 hours are fitted.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. No network, no model calls, deterministic — two runs on the same case write the same bytes. One case in, one `<id>.out` file out: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Backtest

A synthetic launch week (different dynamics than the forecaster assumes: exponential-saturation adoption convolved with player join times, refresh staleness, 2-decimal rounding) was used to validate the entry end to end. Across three player-growth scenarios the forecaster scored a pinball loss of **0.150 / 0.159 / 0.170**, against **0.251–0.260** for the trend baseline and **0.368–0.383** for the persistence baseline, with 90% coverage of 0.96. The practice cases in the kit have a frozen history, so they only exercise the format path (persistence + wide band); they pay nothing and say little about the final.

## License

MIT — see [LICENSE](LICENSE).
