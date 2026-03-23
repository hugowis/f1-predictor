import logging
from pathlib import Path
from typing import Dict, List

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

    def __init__(self, data_path: Path):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.vocabs_path = data_path / "vocabs"
        self.raw_data_path = data_path / "raw_data"
        self.save_dir = data_path / "clean_data"
        self.schedules, self.years = self._load_schedules()

        self.categories = {
            col: {"<UNK>": 0} for col in self.CATEGORICAL_COLUMNS
        }

        self.pipeline = [
            self._add_metadata,
            self._join_weather,
            self._convert_times_to_ms,
            self._correct_track_length,
            self._compute_missing_laptimes,
            self._add_lap_type_flags,
            self._add_trackstatus_flag,
            self._compute_delta_to_car_ahead,
            self._convert_bool,
            self._encode_compound,
            self._add_fuel_proxy,
            self._to_categorical,
            self._add_stint_length,
            self._add_cumulative_stint_time,
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


    def _to_categorical(self, df, **_):
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
                    vocab[v] = len(vocab)
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
    # Orchestration
    # ---------------------------------------------------------------------

    def prepare(self, skip_2018: bool = True):
        for year in self.years:
            if year == 2018 and skip_2018:
                continue

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
    DataPreparation(DATA_PATH).prepare(skip_2018=False)
