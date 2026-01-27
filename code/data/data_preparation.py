import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import os
import logging
from tqdm import tqdm

RAW_DATA_PATH  = Path("data")
CLEANED_DATA_PATH = Path("clean_data")

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class DataPreparation():

    def __init__(
        self,
        raw_data_path: Path,
        save_dir: Path,
    ):

        self.logger = logging.getLogger(self.__class__.__name__)
        self.raw_data_path = raw_data_path
        self.save_dir = save_dir
        self.schedules, self.years = self.load_schedules()

    def load_schedules(self) -> Tuple[Dict, List[int]]:
        schedules = dict()
        years = list()
        schedule_dir = self.raw_data_path / "schedule"

        for filepath in schedule_dir.glob("*.csv"):
            year = int(str(filepath).split("_")[-1].split(".")[0])
            years.append(year)
            schedules[year] = pd.read_csv(filepath)
        
        return schedules, years

    def load_racelaps(self, laps_filepath: str) -> pd.DataFrame:

        df = pd.read_csv(
            laps_filepath,

            parse_dates=["LapStartDate"],

            dtype={
                # strings
                "Driver": "string",
                "Compound": "string",
                "Team": "string",
                "DeletedReason": "string",

                # integers (nullable)
                "DriverNumber": "Int64",
                "TrackStatus": "Int64",
                "Position": "Int64",

                # floats (nullable)
                "LapNumber": "Float64",
                "Stint": "Float64",
                "TyreLife": "Float64",
                "SpeedI1": "Float64",
                "SpeedI2": "Float64",
                "SpeedFL": "Float64",
                "SpeedST": "Float64",
                "TrackLength": "Float64",
                "RaceLaps": "Float64",

                # booleans (nullable)
                "IsPersonalBest": "boolean",
                "FreshTyre": "boolean",
                "Deleted": "boolean",
                "FastF1Generated": "boolean",
                "IsAccurate": "boolean",
            },

            # Treat empty strings as NA
            na_values=[""],

            # Faster / safer defaults
            keep_default_na=True,
            low_memory=False,
        )
        timedelta_cols = [
            "Time", "LapTime", "PitOutTime", "PitInTime",
            "Sector1Time", "Sector2Time", "Sector3Time",
            "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime",
            "LapStartTime",
        ]

        for col in timedelta_cols:
            if col in df.columns:
                df[col] = pd.to_timedelta(df[col], errors="coerce")
        
        return df


    def prepare(self, skip_2018=True):

        for year in self.years:
            if year == 2018 and skip_2018:
                continue
            self.logger.info(f"Preparing data for year: {year}")
            for i, raw in tqdm(self.schedules[year].iterrows(), total=self.schedules[year].shape[0]):
                event_path = self.raw_data_path / "laps" / str(year) / raw['EventName']
                weather_path = self.raw_data_path / "weather" / str(year) / raw['EventName']
                session_filepaths = event_path.glob("*.csv")
                
                for session_file in session_filepaths:
                    session = self.load_racelaps(session_file)
                    weather = pd.read_csv(weather_path / session_file.name )
                    weather["Time"] = pd.to_timedelta(weather["Time"])
                    self.preparation_pipeline(session , year, raw["Location"], weather)


    def convert_to_ms(self, session: pd.DataFrame) -> pd.DataFrame:
        timedelta_cols = [
            "Time", "LapTime", "PitOutTime", "PitInTime",
            "Sector1Time", "Sector2Time", "Sector3Time",
            "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime",
            "LapStartTime",
        ]
        for col in timedelta_cols:
            session[col] = session[col].dt.total_seconds() * 1000

        return session

    def correct_track_length(self, session: pd.DataFrame) -> pd.DataFrame:
        reference_lengths = {
            "Melbourne": 5.28,
            "Sakhir": 5.39,
            "Shanghai": 5.43,
            "Budapest": 4.36,
            "Mexico City": 4.26,
            "São Paulo": 4.25,
            "Spa-Francorchamps":6.96,
            "Spielberg": 4.29
        }
        if session["TrackLength"].notna().all() == False:
            session["TrackLength"] = reference_lengths[session['circuit'].iloc[0]]

        session["TrackLength"] = session["TrackLength"].round(2)
        return session

    def join_weather_data(self, session: pd.DataFrame , weather: pd.DataFrame) -> pd.DataFrame:
        
        session = session.sort_values("Time")
        weather = weather.sort_values("Time")
        session = pd.merge_asof(
            session,
            weather,
            on="Time",
            direction="backward"
        )

        # resort session
        session = session.sort_values(
            by=["Driver", "LapNumber"],
            ascending=[True, True]
        )
        return session 

    def add_lap_type_flags(self, session: pd.DataFrame) -> pd.DataFrame:
        session["is_outlap"] = ~session['PitOutTime'].isna()
        session["is_inlap"] = ~session['PitInTime'].isna()
        session["is_pitlap"] = session["is_outlap"] | session["is_inlap"] 
        session["is_normal_lap"] = ~ session["is_pitlap"]

        session["is_outlap"] = session["is_outlap"].astype(int)
        session["is_inlap"] = session["is_inlap"].astype(int)
        session["is_pitlap"] = session["is_pitlap"].astype(int)
        session["is_normal_lap"] = session["is_normal_lap"].astype(int)
        return session


    def preparation_pipeline(self, session: pd.DataFrame, year: int, circuit: str, weather: pd.DataFrame) -> None:
        
        # Adding schedule data
        self.logger.debug("Joining Schedule data")
        session["year"] = year
        session["circuit"] = circuit
        # Adding weather data
        self.logger.debug("Joining weather data")
        session = self.join_weather_data(session, weather)
        # Convert all times to ms
        self.logger.debug("Converting times to ms.")
        session = self.convert_to_ms(session)
        # Correct track lengths
        self.logger.debug("Correcting track lengths.")
        session = self.correct_track_length(session)
        # Calculate missing Laptimes
        self.logger.debug("Calculating missing Laptimes")
        session["LapTime"] = session["LapTime"].fillna(session["Time"] - session["LapStartTime"])
        # Add lap type flags
        session = self.add_lap_type_flags(session)


        # Drop useless columns
        self.logger.debug("Dropping useless columns")
        to_drop = ["Time", "DriverNumber", "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "IsPersonalBest", "LapStartTime", "LapStartDate", "FastF1Generated", "IsAccurate", "Sector1Time", "Sector2Time", "Sector3Time", "Position", "Deleted", "DeletedReason", "PitOutTime", "PitInTime"]
        session = session.drop(columns=to_drop)



        # TODO save df


if __name__ == "__main__":
    data_prep = DataPreparation(RAW_DATA_PATH, CLEANED_DATA_PATH)
    data_prep.prepare()
