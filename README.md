# F1 Pit Stop Prediction — Data Pipeline & Modeling

Part of the Final Year Project **"Interactive Explainable AI Framework for Formula 1 Pit Stop Optimization"**.

This notebook covers **Objective 2 (dataset construction & feature engineering)** and **Objective 3 (pit-lap prediction models)**: building a unified, multi-season lap-by-lap F1 dataset from three data sources, engineering predictive features, and training/comparing five models to predict whether a given lap is a pit-in lap.

> Exported from Google Colab (`Final Year Project.ipynb`). Some cells expect a mounted Google Drive and are written to run in a Colab environment — see [Environment notes](#environment-notes) below.

---

## What this notebook does

1. **Extracts** lap-by-lap race data for the 2021–2025 F1 seasons from three sources (FastF1, OpenF1, Jolpica-F1), reconciles their schemas, and merges them into one combined dataset.
2. **Engineers features**: tyre age/compound, circuit identity, safety-car status, pace-degradation vs. rolling median, and position-trend (undercut-risk) features for the driver and their immediate rivals.
3. **Rebuilds the prediction target** (`is_pit_lap`) from stint-change ground truth, since the two raw sources flagged pit laps inconsistently.
4. **Trains and compares five models** — a heuristic baseline, logistic regression, XGBoost, Bi-LSTM, and GRU — on a temporal train/test split (train: 2021–2024, test: 2025).
5. **Validates the winning model** (GRU) with season-based cross-validation and a prediction-confidence/calibration check.

## Data sources

| Source | Seasons used | Provides |
|---|---|---|
| [FastF1](https://docs.fastf1.dev/) | 2021–2022 | Laps, tyres, weather |
| [OpenF1](https://openf1.org/) | 2023–2025 | Laps, tyres, weather, safety-car status, position |
| [Jolpica-F1](https://api.jolpi.ca/) (Ergast successor) | 2023 pit stops only | Pit-stop records (OpenF1's `/pit` endpoint has incomplete 2023 coverage) |

**2020 is excluded** — the FastF1 historical mirror doesn't reliably serve that season. Three races are missing entirely across all sources (Imola 2023 — cancelled, Mexico City 2024, Interlagos 2025).

**Final combined dataset:** 120,730 laps × 18 unified columns before feature engineering.

## Pipeline stages

### 1. Extraction (`load_fastf1_race`, `get_openf1_race_data`, `get_jolpica_pitstops_2023`)
Each source's raw per-lap/per-session data is pulled and mapped onto a shared schema (season, round, driver, team, lap number, lap time, compound, tyre life, stint, `is_pit_lap`, track status, weather, position, data source). FastF1 downloads are checkpointed per season to Drive with rate-limit backoff; OpenF1 calls go through a retry-with-backoff wrapper (`safe_get_json`).

### 2. Merge & clean
Six per-season files are concatenated; known dtype mismatches between sources (`rainfall` as bool vs. float, `position` never populated by OpenF1) are reconciled before combining.

### 3. Feature engineering
- **Simple features**: `tyre_age`, one-hot compound (6 categories), one-hot circuit (29 circuits, FastF1/OpenF1 naming reconciled), `is_safety_car_lap`.
- **Pace-degradation**: rolling median lap time over the previous 3 and 5 laps within the same driver+stint (`.shift(1)`, so no leakage from the current lap).
- **Position-trend / undercut-risk**: own, car-ahead, and car-behind position change over 2- and 3-lap windows — resolved via a two-step lookup that finds each neighboring driver's actual stint at the relevant lap before pulling their historical position.

### 4. Target rebuild
`is_pit_lap` is rebuilt from `stint`-change ground truth (the in-lap before a stint increment) rather than trusting either source's native pit flag, after finding FastF1 flagged laps at ~2x the rate of OpenF1 for the same event type.

### 5. Train/test split
Temporal holdout by season — **train on 2021–2024** (95,273 rows), **test on 2025** (25,457 rows, held out entirely) — to mimic real deployment on a genuinely unseen future season and avoid leakage between correlated laps in the same race.

### 6. Modeling — five models compared on the 2025 test set

| Model | F1 |
|---|---|
| Rule-based heuristic (tyre-age threshold) | 0.091 |
| Logistic Regression | 0.223 |
| XGBoost (default, raw NaN) | 0.560 |
| XGBoost (Optuna-tuned) | 0.559 |
| Bi-LSTM (full-race sequences) | 0.810 |
| **GRU (full-race sequences)** | **0.829** (precision 0.757, recall 0.915) |

**GRU is the winning model.** It uses fewer parameters than the Bi-LSTM (51,777 vs. 67,137) on the same full-race sequence framing, and was validated with 3-fold season-based cross-validation (train on growing history, test on the next season): mean F1 = 0.821, std = 0.032 across folds — confirming the result is stable rather than a lucky split.

A sliding-window reformulation of the sequence task ("will this driver pit within the next *N* laps?") was tried as a fallback for the Bi-LSTM's small sequence count, but consistently underperformed (F1 ≈ 0.47–0.53 across window configurations) — documented as a negative finding: the windowed framing is a harder task, not a fix for data scarcity.

Two NaN-handling strategies are also compared for XGBoost specifically (native missing-value handling vs. median/zero-imputation with missingness flags) — native handling wins.

## Environment notes

This is a Colab export, so:
- `!pip install fastf1 pyarrow -q` and `from google.colab import drive` / `drive.mount(...)` only work inside Colab.
- Paths like `/content/drive/MyDrive/fyp_f1/...` assume a mounted Drive with that folder structure.
- To run locally, replace the Drive mount and paths with local directories, and install: `fastf1`, `pyarrow`, `pandas`, `numpy`, `requests`, `scikit-learn`, `xgboost`, `optuna`, `tensorflow`.
- FastF1 season builds can take a long time — they self-checkpoint per season (skip if the season's `.parquet` already exists) and pace themselves against an hourly request budget to avoid the livetiming mirror rate limit.

## Known limitations (documented for the FYP report)

- 2020 season excluded (FastF1 mirror unavailable for that year).
- 3 races missing entirely: Imola 2023, Mexico City 2024, Interlagos 2025.
- `driver_number` is null for 2021–2022 rows (FastF1 identifies drivers by code, not numeric number — a data-model difference between sources, not a defect).
- `lap_time_s` has ~1% missing values overall, concentrated on pit laps (8.0% vs. 0.7% on non-pit laps).
- The tuned XGBoost model's validation F1 (0.833) didn't transfer to the real 2025 test set (0.559) — diagnosed as genuine temporal distribution shift plus some circuit-specific memorization in top features, not a tuning bug.

## Status

O2 (feature engineering) and O3 (prediction models) are complete. Next stages (not in this notebook): O4 TimeSHAP explainability module, O5 what-if scenario engine, O6 Streamlit dashboard.
