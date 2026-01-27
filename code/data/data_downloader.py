"""
FastF1 raw data downloader.

Downloads lap and weather data and stores them in data/raw_data.
"""

import time
import shutil
import argparse
import logging
from pathlib import Path

import fastf1
from tqdm import tqdm

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RAW_DATA_DIR = Path("data/raw_data")
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

MAX_RETRIES = 3
RETRY_DELAY = 30

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

for name in ["fastf1", "fastf1.core", "fastf1.api", "urllib3", "requests", "requests_cache"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

fastf1.set_log_level("CRITICAL")
fastf1.Cache.enable_cache(CACHE_DIR)

# ---------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------


class FastF1Downloader:
    """
    Downloads raw FastF1 data (laps + weather).
    """

    def __init__(
        self,
        raw_data_dir: Path,
        start_year: int,
        end_year: int,
        overwrite: bool = False,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.raw_data_dir = raw_data_dir
        self.start_year = start_year
        self.end_year = end_year
        self.overwrite = overwrite

        self.schedule_dir = raw_data_dir / "schedule"
        self.laps_dir = raw_data_dir / "laps"
        self.weather_dir = raw_data_dir / "weather"

        self.schedules = {}
        self.total_sessions = 0

        if overwrite and raw_data_dir.exists():
            self.logger.warning("Deleting existing raw data directory")
            shutil.rmtree(raw_data_dir)

        self._init_directories()

    # -----------------------------------------------------------------

    def _init_directories(self):
        for p in [self.schedule_dir, self.laps_dir, self.weather_dir]:
            p.mkdir(parents=True, exist_ok=True)

        for year in range(self.start_year, self.end_year + 1):
            (self.laps_dir / str(year)).mkdir(exist_ok=True)
            (self.weather_dir / str(year)).mkdir(exist_ok=True)

    # -----------------------------------------------------------------

    def fetch_schedules(self):
        for year in range(self.start_year, self.end_year + 1):
            self.logger.info("Fetching schedule %d", year)
            schedule = fastf1.get_event_schedule(year, include_testing=False)

            self.schedules[year] = schedule
            schedule.to_csv(self.schedule_dir / f"schedule_{year}.csv", index=False)

            n_sessions = schedule[
                ["Session1", "Session2", "Session3", "Session4", "Session5"]
            ].notna().sum().sum()

            self.total_sessions += n_sessions

    # -----------------------------------------------------------------

    def _already_downloaded(self, year, event, session):
        lap_fp = self.laps_dir / str(year) / event / f"{session}.csv"
        weather_fp = self.weather_dir / str(year) / event / f"{session}.csv"
        return lap_fp.exists() and weather_fp.exists()

    # -----------------------------------------------------------------

    def _load_session(self, year, event, session_name):
        last_exc = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                session = fastf1.get_session(year, event, session_name)
                session.load(laps=True, weather=True, telemetry=True, messages=False)
                session.laps  # force eval
                return session

            except Exception as e:
                last_exc = e
                self.logger.warning(
                    "Retry %d/%d failed: %d | %s | %s (%s)",
                    attempt, MAX_RETRIES, year, event, session_name, type(e).__name__,
                )
                time.sleep(RETRY_DELAY)

        raise last_exc

    # -----------------------------------------------------------------

    def _extract_track_length_km(self, session):
        try:
            lap = session.laps.pick_fastest()
            tel = lap.get_telemetry()
            return tel["Distance"].max() / 1000
        except Exception:
            return None

    # -----------------------------------------------------------------

    def download_event(self, year, event_name, sessions, pbar):
        laps_out = self.laps_dir / str(year) / event_name
        weather_out = self.weather_dir / str(year) / event_name
        laps_out.mkdir(exist_ok=True)
        weather_out.mkdir(exist_ok=True)

        for session_name in sessions:
            if not session_name or session_name == "None":
                pbar.update(1)
                continue

            if self._already_downloaded(year, event_name, session_name):
                pbar.update(1)
                continue

            try:
                session = self._load_session(year, event_name, session_name)

                laps = session.laps.copy()
                laps["TrackLength"] = self._extract_track_length_km(session)
                laps["RaceLaps"] = session.total_laps

                laps.to_csv(laps_out / f"{session_name}.csv", index=False)
                session.weather_data.to_csv(
                    weather_out / f"{session_name}.csv", index=False
                )

            except Exception as e:
                self.logger.exception(
                    "Failed: %d | %s | %s (%s)",
                    year, event_name, session_name, type(e).__name__,
                )

            finally:
                pbar.update(1)

    # -----------------------------------------------------------------

    def run(self):
        self.fetch_schedules()

        with tqdm(total=self.total_sessions, desc="Downloading FastF1 data") as pbar:
            for year, schedule in self.schedules.items():
                for _, event in schedule.iterrows():
                    sessions = [
                        event[c]
                        for c in ["Session1", "Session2", "Session3", "Session4", "Session5"]
                    ]
                    self.download_event(year, event["EventName"], sessions, pbar)

        self.logger.info("Raw data download completed")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser("FastF1 raw data downloader")

    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    downloader = FastF1Downloader(
        raw_data_dir=RAW_DATA_DIR,
        start_year=args.start_year,
        end_year=args.end_year,
        overwrite=args.overwrite,
    )

    downloader.run()
