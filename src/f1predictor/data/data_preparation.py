import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import json

DATA_PATH = Path("./data")

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class DataPreparation:
    """
    Raw -> Cleaned dataset preparation.
    """

    TIME_COLS = [
        "Time", "LapTime", "PitOutTime", "PitInTime",
        "Sector1Time", "Sector2Time", "Sector3Time",
        "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime",
        "LapStartTime",
    ]

    DROP_COLS = [
        "Time", "DriverNumber",
        "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "IsPersonalBest", "LapStartTime", "LapStartDate",
        "FastF1Generated", "IsAccurate",
        "Sector1Time", "Sector2Time", "Sector3Time",
        "Deleted", "DeletedReason",
        "PitOutTime", "PitInTime",
        "TrackStatus", "Compound"
    ]

    TRACK_LENGTH_REF = {
        "Melbourne": 5.28,
        "Sakhir": 5.39,
        "Shanghai": 5.43,
        "Budapest": 4.36,
        "Mexico City": 4.26,
        "São Paulo": 4.25,
        "Spa-Francorchamps": 6.96,
        "Spielberg": 4.29,
    }

    BOOL_COLUMNS = [
        "FreshTyre",
        "Rainfall"
    ]

    CATEGORICAL_COLUMNS = [
        "Driver",
        "Team",
        "Year",
        "Circuit"

    ]

    STINT_NORMS_FALLBACK = 25.0  # laps; used when (circuit, compound) unseen
    STINT_NORMS_MIN_GROUP_SIZE = 3  # drop tiny groups to avoid noisy medians

    def __init__(self, data_path: Path, vocab_years: List[int] = None):
        """
        Parameters
        ----------
        data_path : Path
            Root data directory containing raw_data/, clean_data/, vocabs/
        vocab_years : list of int, optional
            Years to use for building categorical vocabularies.
            Categories appearing ONLY in years outside this list will be
            mapped to <UNK> (index 0). If None, all available years are
            used (legacy behaviour — not recommended, causes data leakage).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_path = data_path
        self.vocabs_path = data_path / "vocabs"
        self.raw_data_path = data_path / "raw_data"
        self.save_dir = data_path / "clean_data"
        self.stint_norms_path = data_path / "stint_norms.json"
        self.schedules, self.years = self._load_schedules()
        self.vocab_years = set(vocab_years) if vocab_years is not None else None

        self.categories = {
            col: {"<UNK>": 0} for col in self.CATEGORICAL_COLUMNS
        }

        # (circuit, compound_UPPER) -> expected stint length; built lazily
        # from raw training-year laps. C6 uses this for stint_progress_pct.
        self.stint_norms: Dict[str, float] = self._load_or_build_stint_norms()

        self.pipeline = [
            self._add_metadata,
            self._join_weather,
            self._convert_times_to_ms,
            self._correct_track_length,
            self._compute_missing_laptimes,
            self._add_lap_type_flags,
            self._add_trackstatus_flag,
            self._compute_delta_to_car_ahead,
            self._add_traffic_rate,              # C4
            self._convert_bool,
            self._encode_compound,
            self._add_fuel_proxy,
            self._add_weather_deltas,            # C3
            self._to_categorical,
            self._add_stint_length,
            self._add_cumulative_stint_time,
            self._process_sector_times,          # C5
            self._add_rolling_laptime_trend,     # C1
            self._add_stint_deg_slope,           # C2
            self._add_stint_progress_pct,        # C6
            self._drop_useless_columns,
        ]

    # ---------------------------------------------------------------------
    # IO
    # ---------------------------------------------------------------------

    def _load_schedules(self) -> tuple[Dict[int, pd.DataFrame], List[int]]:
        schedules = {}
        years = []

        for fp in (self.raw_data_path / "schedule").glob("*.csv"):
            year = int(fp.stem.split("_")[-1])
            schedules[year] = pd.read_csv(fp)
            years.append(year)

        return schedules, sorted(years)

    def _load_laps(self, filepath: Path) -> pd.DataFrame:
        df = pd.read_csv(
            filepath,
            parse_dates=["LapStartDate"],
            low_memory=False,
            na_values=[""],
            keep_default_na=True,
        )

        for col in self.TIME_COLS:
            if col in df.columns:
                df[col] = pd.to_timedelta(df[col], errors="coerce")

        return df

    # ---------------------------------------------------------------------
    # Pipeline steps
    # ---------------------------------------------------------------------

    def _add_metadata(self, df, *, year, circuit, **_):
        df["Year"] = year
        df["Circuit"] = circuit
        return df

    def _join_weather(self, df, *, weather, **_):
        df = df.sort_values("Time")
        weather = weather.sort_values("Time")

        df = pd.merge_asof(
            df,
            weather,
            on="Time",
            direction="nearest",
        )

        return df.sort_values(["Driver", "LapNumber"])

    def _convert_times_to_ms(self, df, **_):
        for col in self.TIME_COLS:
            if col in df.columns and pd.api.types.is_timedelta64_dtype(df[col]):
                df[col] = df[col].dt.total_seconds() * 1000
        return df

    def _correct_track_length(self, df, **_):
        if df["TrackLength"].isna().any():
            circuit = df["Circuit"].iloc[0]
            ref = self.TRACK_LENGTH_REF.get(circuit)
            if ref is None:
                self.logger.warning(f"No reference track length for {circuit}")
            else:
                df["TrackLength"] = df["TrackLength"].fillna(ref)

        df["TrackLength"] = df["TrackLength"].round(2)
        return df

    def _compute_missing_laptimes(self, df, **_):
        mask = df["LapTime"].isna() & df["LapStartTime"].notna()
        df.loc[mask, "LapTime"] = df.loc[mask, "Time"] - df.loc[mask, "LapStartTime"]
        return df

    def _add_lap_type_flags(self, df, **_):
        df["is_outlap"] = (
            (df["LapNumber"] == 1) | df["PitOutTime"].notna()
        ).astype(int)

        df["is_inlap"] = df["PitInTime"].notna().astype(int)
        df["is_pitlap"] = (df["is_outlap"] | df["is_inlap"]).astype(int)
        df["is_normal_lap"] = (~df["is_pitlap"].astype(bool)).astype(int)

        return df

    def _add_trackstatus_flag(self, df, **_):
        status = df["TrackStatus"].fillna(0).astype(int)

        df["track_clear"] = (status == 1).astype(int)
        df["yellow_flag"] = (status == 2).astype(int)
        df["safety_car"] = (status == 4).astype(int)
        df["red_flag"] = (status == 5).astype(int)
        df["vsc"] = (status == 6).astype(int)
        df["vsc_ending"] = (status == 7).astype(int)

        return df
    

    def _compute_delta_to_car_ahead(self, df, **_):

        df = df.sort_values(["LapStartTime"])

        df["delta_to_car_ahead"] = df["LapStartTime"] - df["LapStartTime"].shift(1)

        df.loc[df["Position"] == 1, "delta_to_car_ahead"] = 0.0
        df["delta_to_car_ahead"] = df["delta_to_car_ahead"].fillna(0.0)
        return df
    

    def _convert_bool(self, df, **_):

        for col in self.BOOL_COLUMNS:
            df[col] =  df[col].astype(int)

        return df
    
    def _encode_compound(self, df, **_):
        df["Compound"] = df["Compound"].fillna("UNKNOWN")

        df["Compound"] = df["Compound"].str.upper()
        for c in ["ULTRASOFT", "SUPERSOFT", "HYPERSOFT", "SOFT", "MEDIUM", "HARD", "UNKNOWN"]:
            df[f"compound_{c.lower()}"] = (df["Compound"] == c).astype("Int64")
        return df


    def _add_fuel_proxy(self, df, **_):
        df["fuel_proxy"] = 1 - (df["LapNumber"] / df["RaceLaps"])
        return df
    
    def _drop_useless_columns(self, df, **_):
        return df.drop(columns=[c for c in self.DROP_COLS if c in df.columns])


    def _to_categorical(self, df, *, year=None, **_):
        """Encode categorical columns as integer indices.

        When ``self.vocab_years`` is set, new vocabulary entries are only
        created for rows belonging to those years.  Categories that appear
        exclusively in held-out years (val / test) are mapped to ``<UNK>``
        (index 0), preventing data leakage through the embedding layer.
        """
        # Determine whether this DataFrame comes from a vocab-building year
        allow_new_entries = (self.vocab_years is None) or (year in self.vocab_years)

        for col in self.CATEGORICAL_COLUMNS:
            vocab = self.categories[col]

            values = (
                df[col]
                .astype("string")
                .fillna("<UNK>")
            )

            encoded = []

            for v in values:
                if v not in vocab:
                    if allow_new_entries:
                        vocab[v] = len(vocab)
                    else:
                        # Map unseen category to <UNK>
                        encoded.append(vocab["<UNK>"])
                        continue
                encoded.append(vocab[v])

            df[col] = pd.Series(encoded, index=df.index).astype("Int64")

        return df
    
    def _add_stint_length(self, df, **_):
        df = df.sort_values(["Driver", "LapNumber"])

        is_new_stint = (
            (df["PitOutTime"].notna()) |
            (df["LapNumber"] == 1)
        )

        df["stint_id"] = is_new_stint.groupby(df["Driver"]).cumsum()

        mask = df["is_normal_lap"] == 1

        df.loc[mask, "stint_lap"] = (
            df[mask]
            .groupby(["Driver", "Stint"])
            .cumcount()
            + 1
        )
        df["stint_lap"] = df["stint_lap"].fillna(-1)
        return df

    def _add_cumulative_stint_time(self, df, **_):
        """Cumulative sum of LapTime within each driver-stint."""
        df = df.sort_values(["Driver", "LapNumber"])
        df["cumulative_stint_time"] = (
            df.groupby(["Driver", "stint_id"])["LapTime"].cumsum()
        )
        df["cumulative_stint_time"] = df["cumulative_stint_time"].fillna(0.0)
        return df

    # ---------------------------------------------------------------------
    # Part C — engineered trend / deg / weather / sector features
    # ---------------------------------------------------------------------

    def _add_traffic_rate(self, df, **_):
        """C4: first difference of delta_to_car_ahead per driver."""
        df = df.sort_values(["Driver", "LapNumber"])
        df["d_delta_to_car_ahead"] = (
            df.groupby("Driver")["delta_to_car_ahead"].diff().fillna(0.0)
        )
        return df

    def _add_weather_deltas(self, df, **_):
        """C3: weather deltas vs race start (per-race first-lap reference)."""
        df = df.sort_values(["Driver", "LapNumber"])
        for src, dst in [
            ("AirTemp", "d_airtemp_vs_start"),
            ("TrackTemp", "d_tracktemp_vs_start"),
            ("Humidity", "d_humidity_vs_start"),
        ]:
            if src in df.columns:
                ref = df[src].dropna()
                ref_val = float(ref.iloc[0]) if len(ref) else 0.0
                df[dst] = (df[src] - ref_val).fillna(0.0)
            else:
                df[dst] = 0.0
        return df

    def _process_sector_times(self, df, **_):
        """C5: rename sector times to *_ms, ffill, compute per-stint deltas.

        Originals (Sector{1,2,3}Time) are still dropped at the end by
        DROP_COLS; the renamed ms versions survive.
        """
        df = df.sort_values(["Driver", "LapNumber"])
        for i in (1, 2, 3):
            src = f"Sector{i}Time"
            dst = f"Sector{i}Time_ms"
            if src in df.columns:
                df[dst] = df[src]
            else:
                df[dst] = np.nan

            # Per-stint ffill + fallback to the stint's median
            df[dst] = (
                df.groupby(["Driver", "stint_id"])[dst]
                .transform(lambda s: s.ffill().fillna(s.median()))
            )
            # If the whole stint was NaN, fall back to the race-level median
            med = df[dst].median()
            df[dst] = df[dst].fillna(med if pd.notna(med) else 0.0)

            # Per-stint delta vs stint's first valid lap (shape signal)
            first_vals = (
                df.groupby(["Driver", "stint_id"])[dst].transform("first")
            )
            df[f"sector{i}_delta"] = (df[dst] - first_vals).fillna(0.0)
        return df

    def _add_rolling_laptime_trend(self, df, **_):
        """C1: EMA + deltas of LapTime within each driver-stint.

        Non-normal laps (in/out/pitlap) are masked and forward-filled so
        pit noise doesn't poison the trend.
        """
        df = df.sort_values(["Driver", "LapNumber"])
        masked = df["LapTime"].where(df["is_normal_lap"] == 1)
        masked = masked.groupby([df["Driver"], df["stint_id"]]).ffill()
        # If a stint starts with a non-normal lap, bfill so early rows are
        # not NaN; else fall back to raw LapTime.
        masked = masked.groupby([df["Driver"], df["stint_id"]]).bfill()
        masked = masked.fillna(df["LapTime"])

        grp = masked.groupby([df["Driver"], df["stint_id"]])
        df["laptime_ema_3"] = (
            grp.transform(lambda s: s.ewm(span=3, adjust=False).mean())
        ).fillna(0.0)
        df["laptime_ema_5"] = (
            grp.transform(lambda s: s.ewm(span=5, adjust=False).mean())
        ).fillna(0.0)
        df["laptime_delta_1"] = grp.transform(lambda s: s - s.shift(1)).fillna(0.0)
        df["laptime_delta_3"] = grp.transform(lambda s: s - s.shift(3)).fillna(0.0)
        return df

    def _add_stint_deg_slope(self, df, **_):
        """C2: rolling slope of LapTime on stint_lap over last 3 in-stint laps."""
        df = df.sort_values(["Driver", "LapNumber"])

        def _slope(window: pd.Series) -> float:
            if len(window) < 2:
                return 0.0
            y = window.values.astype(np.float64)
            x = np.arange(len(y), dtype=np.float64)
            if np.isnan(y).any():
                mask = ~np.isnan(y)
                if mask.sum() < 2:
                    return 0.0
                x, y = x[mask], y[mask]
            x_mean = x.mean()
            y_mean = y.mean()
            denom = ((x - x_mean) ** 2).sum()
            if denom == 0.0:
                return 0.0
            return float(((x - x_mean) * (y - y_mean)).sum() / denom)

        masked = df["LapTime"].where(df["is_normal_lap"] == 1)
        slopes = (
            masked.groupby([df["Driver"], df["stint_id"]])
            .transform(lambda s: s.rolling(window=3, min_periods=2).apply(_slope, raw=False))
        )
        df["stint_deg_slope_3"] = slopes.fillna(0.0)
        return df

    def _add_stint_progress_pct(self, df, *, circuit=None, **_):
        """C6: stint_lap / expected_stint_len for (circuit, compound).

        Uses self.stint_norms (built from training years) with a constant
        fallback so held-out circuits still produce finite values.
        """
        compound = df.get("Compound")
        if compound is None:
            df["stint_progress_pct"] = 0.0
            return df

        expected = compound.astype("string").str.upper().map(
            lambda c: self.stint_norms.get(
                f"{circuit}|{c}", self.STINT_NORMS_FALLBACK
            )
        ).astype(float)
        expected = expected.where(expected > 0, self.STINT_NORMS_FALLBACK)

        stint_lap = df["stint_lap"].astype(float)
        # -1 sentinel (non-normal laps) → 0.0 progress
        pct = np.where(stint_lap < 0, 0.0, stint_lap / expected)
        df["stint_progress_pct"] = pct.astype(np.float32)
        return df

    # ---------------------------------------------------------------------
    # Stint norms (C6 support)
    # ---------------------------------------------------------------------

    def _load_or_build_stint_norms(self) -> Dict[str, float]:
        """Return (circuit, compound) -> expected stint length.

        Scans raw training-year laps. Stint length = per-(Driver, Stint)
        max(LapNumber) - min(LapNumber) + 1. Aggregated as median per
        (circuit_name, Compound_UPPER). Cached to data/stint_norms.json.
        Held-out circuits/compounds use STINT_NORMS_FALLBACK at lookup.
        """
        if self.stint_norms_path.exists():
            try:
                with open(self.stint_norms_path) as f:
                    norms = json.load(f)
                self.logger.info(
                    f"Loaded stint_norms ({len(norms)} keys) from cache"
                )
                return {k: float(v) for k, v in norms.items()}
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning(f"Failed to load stint_norms cache: {e}")

        self.logger.info("Building stint_norms from raw training-year laps...")
        train_years = (
            sorted(self.vocab_years) if self.vocab_years else self.years
        )

        records = []
        for year in train_years:
            if year not in self.schedules:
                continue
            for _, race in self.schedules[year].iterrows():
                event = race["EventName"]
                circuit = race["Location"]
                laps_dir = self.raw_data_path / "laps" / str(year) / event
                race_csv = laps_dir / "Race.csv"
                if not race_csv.exists():
                    continue
                try:
                    df = pd.read_csv(
                        race_csv,
                        usecols=["Driver", "LapNumber", "Stint", "Compound"],
                        low_memory=False,
                    )
                except (ValueError, OSError):
                    continue
                df = df.dropna(subset=["Driver", "LapNumber", "Stint", "Compound"])
                if df.empty:
                    continue
                df["Compound"] = df["Compound"].astype(str).str.upper()
                grp = df.groupby(["Driver", "Stint", "Compound"])
                stint_lens = (grp["LapNumber"].max() - grp["LapNumber"].min() + 1)
                for (drv, stint, comp), length in stint_lens.items():
                    records.append((circuit, comp, float(length)))

        norms: Dict[str, float] = {}
        if records:
            rec_df = pd.DataFrame(records, columns=["circuit", "compound", "length"])
            agg = rec_df.groupby(["circuit", "compound"])
            counts = agg.size()
            medians = agg["length"].median()
            for (circuit, compound), med in medians.items():
                if counts[(circuit, compound)] < self.STINT_NORMS_MIN_GROUP_SIZE:
                    continue
                norms[f"{circuit}|{compound}"] = float(med)

        try:
            self.stint_norms_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.stint_norms_path, "w") as f:
                json.dump(norms, f, indent=2, sort_keys=True)
            self.logger.info(
                f"Built stint_norms ({len(norms)} keys) → {self.stint_norms_path}"
            )
        except OSError as e:
            self.logger.warning(f"Failed to cache stint_norms: {e}")

        return norms

    # ---------------------------------------------------------------------
    # Orchestration
    # ---------------------------------------------------------------------

    def prepare(self, skip_2018: bool = True):
        # Process vocab years first so the vocabulary is fully built
        # before encountering held-out years (which should use <UNK>).
        if self.vocab_years is not None:
            ordered = sorted(
                [y for y in self.years if (y != 2018 or not skip_2018)],
                key=lambda y: (y not in self.vocab_years, y),
            )
        else:
            ordered = [y for y in self.years if (y != 2018 or not skip_2018)]

        for year in ordered:

            self.logger.info(f"Preparing year {year}")

            for _, race in tqdm(self.schedules[year].iterrows(),
                                 total=len(self.schedules[year])):

                event = race["EventName"]
                circuit = race["Location"] 

                laps_dir = self.raw_data_path / "laps" / str(year) / event
                weather_dir = self.raw_data_path / "weather" / str(year) / event

                for lap_file in laps_dir.glob("*.csv"):
                    weather = pd.read_csv(weather_dir / lap_file.name)
                    weather["Time"] = pd.to_timedelta(weather["Time"])

                    df = self._load_laps(lap_file)
                    df = df.dropna(subset=["LapTime", "Driver", "LapNumber"])
                    for step in self.pipeline:
                        df = step(
                            df,
                            year=year,
                            circuit=circuit,
                            weather=weather,
                        )

                    self._save(df, year, event, lap_file.stem)

        self._save_vocab()

    def _save(self, df, year, event, session):
        out_dir = self.save_dir / f"{year}" / f"{event}"
        out_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_dir / f"{session}.csv", index=False)

    def _save_vocab(self):
        
        self.vocabs_path.mkdir(parents=True, exist_ok=True)
        for k in self.categories.keys():
            path = self.vocabs_path / k
            with open(f"{path}.json", "w") as f:
                json.dump(self.categories[k], f, indent=2, sort_keys=True)

if __name__ == "__main__":
    # Build vocabularies from training years only to prevent data leakage.
    # Validation/test year categories unseen in training will map to <UNK>.
    TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]
    DataPreparation(DATA_PATH, vocab_years=TRAIN_YEARS).prepare(skip_2018=False)
