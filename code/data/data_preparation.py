import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Int

RAW_DATA_PATH  = Path("data")
CLEANED_DATA_PATH = Path("clean_data")


class DataPreparation():

    def __init__(
        self,
        raw_data_path: Path,
        save_dir: Path,
    ):
        self.raw_data_path = raw_data_path
        self.save_dir = save_dir
        self.schedules, self.years = self.load_schedules()
         


    def load_schedules(self) -> Tuple[Dict, List[Int]]:
        schedules = dict()
        years = list()
        schedule_dir = self.raw_data_path / "schedule"
        for filepath in os.listdir():
            complete_filepath = schedule_dir / filepath
            year = int(filepath.split("_")[-1].split(".")[0])
            years.append(year)

            schedules[year] = pd.read_csv(complete_filepath)
        
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

if __name__ == "__main__":
    DataPreparation(RAW_DATA_PATH, CLEANED_DATA_PATH)