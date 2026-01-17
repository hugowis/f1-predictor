# F1 Lap Time Predictor

The goal of this project is to build a deep learning–based solution to **predict the next lap times of an F1 driver**, given the information available from **previous laps only**.

This is formulated as a **next-step time series prediction problem**:
> Given laps 1..t for a driver in a race, predict lap time at t+1, t+2, ...

The project focuses on race laps only (not qualifying).

---

## Data

All data is obtained using the **FastF1** Python framework and downloaded via:
```code/data/data_downloader.py```


### Raw Lap Data (FastF1)

Each row corresponds to a single lap for a driver in a race and includes:

- Lap timing information
- Sector times
- Speed traps
- Tyre information
- Driver and team metadata
- Track status flags (yellow flags, safety car, etc.)

Example columns:

```Time, Driver, LapTime, LapNumber, Stint, Sector1Time, Sector2Time, Sector3Time,SpeedI1, SpeedI2, SpeedFL, SpeedST, Compound, TyreLife, FreshTyre, Team, TrackStatus, Position, PitInTime, PitOutTime, ...```


### Additional Data

The following data sources are joined at lap level:

- **Weather data**
  - Track temperature
  - Air temperature
  - Humidity
  - Rain
  - Wind speed (if available)
- **Track / schedule metadata**
  - Circuit name
  - Track length
  - Season / year
  - Race total laps

---

## Task Definition

**Primary task**:  
Predict the **next lap times** for a given driver.

- Input: laps `[t-N, ..., t]`
- Output: `LapTime(t+1), LapTime(t+2), ...`
- Only information available **up to lap t** is allowed (no data leakage)

---

## TODOLIST

### 1. Data Preparation

#### 1.1 Data Cleaning

- Convert all time-related columns to milliseconds
- Drop laps with missing or invalid lap times
- Remove red-flag laps
- Handle deleted / inaccurate laps using FastF1 flags
- Handle missing track length

#### 1.2 Lap Type Handling

Create explicit lap-type flags:
- `is_normal_lap`
- `is_outlap`
- `is_inlap`
- `is_pit_lap`

For initial experiments:
- Drop in-laps and out-laps
- Ignore laps under Safety Car / VSC

---

### 1.3 Feature Engineering

#### Temporal & Performance Features

- Previous lap times (t-1, t-2, ..., t-N)
- Lap time deltas:
  - `LapTime(t) - LapTime(t-1)`
- Rolling statistics:
  - Rolling mean / std over last 3–5 laps
- Gap to car ahead
- Gap to race leader 

#### Tyre Features

- Tyre compound (one-hot encoded)
- Tyre life
- Fresh tyre flag
- Stint length so far
- Normalized tyre age

#### Fuel Load Proxy

Fuel load is not available and must be approximated:

- Fuel proxy:
```fuel_proxy = 1 - (LapNumber / TotalRaceLaps)```


#### Track & Circuit Features

- Circuit name (categorical / embedding)
- Track length
- Average lap speed
- Track type (low / medium / high downforce – optional)

#### Driver & Car Features

- Driver ID (categorical / embedding)
- Team ID (categorical / embedding)
- Season / car year

---

### 1.4 Categorical Encoding

- One-hot encoding for:
    - Tyre compound
    - Track status (green / yellow / SC)
- Embeddings for:
    - Driver
    - Team
    - Circuit

---

## 2. Modeling

### 2.1 Baselines

- Naive baseline:
    - `LapTime(t+1) = LapTime(t)`
    - Rolling mean baseline
    - ARIMA-style models (per driver, per race)

---

### 2.2 Machine Learning Models

#### Feedforward Neural Network (MLP)

- Input:
- Flattened previous N laps
- Static context features
- Output:
- Next lap time

Used as the first deep learning baseline.

---

#### Recurrent Models

- GRU
- LSTM

In 3 phases:
- Phase 1

  - Pure teacher forcing
  - Stint-based sequences
  - seq2seq  (1-5 laps)

- Phase 2

  - Full-race sequences
  - Teacher forcing
  - Auxiliary pit head and compound head
  - Autoregressive

- Phase 3

  - Partial free-running
  - Scheduled sampling


---

#### Transformer Models

- Decoder-only architecture (autoregressive)
- For phase 2 & 3

Notes:
- Causal masking (no future access)
- Requires careful regularization to avoid overfitting
- Used only once dataset size is sufficient

---

## 3. Evaluation Protocol

### Data Splitting

- Split **by race**, not by lap
- Recommended:
  - Train: older seasons
  - Validation: middle season(s)
  - Test: most recent season

This prevents information leakage across laps of the same race.

---

### Metrics

- Mean Absolute Error (MAE) in milliseconds
- MAE normalized by track average lap time
- Error analysis by:
  - Stint phase (early / mid / late)
  - Tyre compound
  - Track

---

## 4. Additional Experiments

### Value qualification and testing data
- Add a session type tag 
- some features are not comparable accros sessions: fuel, trafick, ...
- To train the embeddings and freeze them
- To learn tire degradation curves (on FP long stints)
- Define a pretaining task

### Circuit Representation

- Mean lap time / speed per circuit
- Learned circuit embeddings
- Circuit clustering

---

### Driver & Car Representation

- Driver embeddings
- Team / car-year embeddings
- Driver-specific output heads 

---

### Extensions (Future Work)

- Uncertainty estimation (predict mean + variance)
- Counterfactual simulation:
  - "What if the driver stayed out one more lap?"

- A GUI?
---



