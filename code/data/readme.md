# Data Pipeline Overview

This folder contains the **data ingestion and preparation pipeline** used in the F1 Lap Time Predictor project. The goal of this pipeline is to reliably transform raw data fetched from the FastF1 API into clean, ML‑ready datasets.

```
code/
└── data/
    ├── data_downloader.py
    └── data_preparation.py
```

The pipeline is intentionally split into **two independent stages**:

1. **Data Downloading (raw data)**
2. **Data Preparation / Cleaning (clean data)**

This separation ensures reproducibility, debuggability, and prevents data leakage.

---

## 1. `data_downloader.py`

### Purpose

`data_downloader.py` is responsible for **fetching raw Formula 1 data** from the FastF1 library and storing it on disk **without applying any ML‑specific transformations**.

The output of this script is considered **immutable raw data**.

### What it downloads

For each season and race:

* Event schedules
* Lap‑level data for each session (race, practice, qualifying)
* Weather data aligned with session time

Minimal enrichment is applied:

* Track length (when available)
* Total number of race laps

No cleaning, filtering, or feature engineering is performed at this stage.

### Output structure

```text
data/raw_data/
├── schedule/
│   └── schedule_2023.csv
├── laps/
│   └── 2023/
│       └── Bahrain Grand Prix/
│           └── Race.csv
├── weather/
│   └── 2023/
│       └── Bahrain Grand Prix/
│           └── Race.csv
```

### How to use

From the project root:

```bash
python code/data/data_downloader.py \
    --start-year 2020 \
    --end-year 2023 \
    --overwrite
```

### Key characteristics

* Uses FastF1 cache for efficiency
* Retries transient API failures
* Skips already downloaded sessions
* Safe to re‑run multiple times

---

## 2. `data_preparation.py`

### Purpose

`data_preparation.py` converts **raw FastF1 data** into **clean, consistent, ML‑ready datasets**.

### Main responsibilities

* Convert all time columns to milliseconds
* Join weather data to lap data
* Correct missing or inconsistent track lengths
* Compute missing lap times
* Add lap‑type flags:

  * `is_normal_lap`
  * `is_outlap`
  * `is_inlap`
  * `is_pitlap`
* Handle missing values explicitly
* Drop non‑useful or leakage‑prone columns

No modeling or normalization is done at this stage.

### Output structure

```text
data/clean_data/
└── year=2023/
    └── circuit=Bahrain/
        └── Race.parquet
```

### How to use

```bash
python code/data/data_preparation.py
```

Optional behaviors (depending on implementation):

* Skip specific 2018 (this season is having more missing data and is the only season with the previous tyre compounds. Ex: super-soft, ...)
* Run safely on partial datasets

---

## Recommended Workflow

```text
FastF1 API
   ↓
data_downloader.py
   ↓
data/raw_data/
   ↓
data_preparation.py
   ↓
data/clean_data/
   ↓
Feature engineering / Dataloaders / Models
```

---
