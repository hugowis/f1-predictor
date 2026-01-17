"""
FastF1 data downloader CLI tool.

This script downloads lap and weather data for Formula 1 sessions
using the FastF1 library.

Typical usage:
    python downloader.py --start-year 2020 --end-year 2023
"""

import time
import shutil
import fastf1
from tqdm import tqdm
from pathlib import Path
import logging
import argparse

SAVE_DIR = Path("./data")
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"

MAX_RETRIES = 3
RETRY_DELAY = 30

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Silence FastF1 and noisy dependencies
for name in ["fastf1", "fastf1.core", "fastf1.api", "urllib3", "requests", "requests_cache"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

fastf1.set_log_level("CRITICAL")
fastf1.Cache.enable_cache(CACHE_DIR)


class DataDownloader:
    """
    Download and persist FastF1 lap and weather data.

    The downloader:
    - Fetches event schedules for a given year range
    - Downloads session lap and weather data
    - Skips sessions already present on disk
    - Retries transient API failures
    - Logs progress and errors
    """

    def __init__(
        self,
        save_dir: Path,
        starting_year: int = 2018,
        end_year: int = 2025,
        delete_previous_data: bool = False,
    ):
        """
        Initialize the downloader.

        Args:
            save_dir: Base directory where data will be stored.
            starting_year: First season to download.
            end_year: Last season to download.
            delete_previous_data: If True, deletes existing data directory.
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.starting_year = starting_year
        self.end_year = end_year
        self.save_dir = save_dir

        self.schedule_dir = save_dir / "schedule"
        self.laps_dir = save_dir / "laps"
        self.weather_dir = save_dir / "weather"

        self.schedule = {}
        self.sessions_count = 0

        if delete_previous_data and save_dir.exists():
            self.logger.warning("Deleting existing data directory: %s", save_dir)
            shutil.rmtree(save_dir)

        self._create_directories()

    def _create_directories(self) -> None:
        """
        Create the directory structure used to store schedules,
        lap data, and weather data.
        """
        for path in [self.schedule_dir, self.laps_dir, self.weather_dir]:
            path.mkdir(parents=True, exist_ok=True)

        for year in range(self.starting_year, self.end_year + 1):
            (self.laps_dir / str(year)).mkdir(exist_ok=True)
            (self.weather_dir / str(year)).mkdir(exist_ok=True)

    def get_schedules(self) -> None:
        """
        Fetch and store the event schedules for all configured seasons.

        Also counts the total number of sessions to initialize
        the progress bar accurately.
        """
        for year in range(self.starting_year, self.end_year + 1):
            self.logger.info("Fetching schedule for %d", year)

            schedule = fastf1.get_event_schedule(year, include_testing=False)
            self.schedule[year] = schedule
            schedule.to_csv(self.schedule_dir / f"schedule_{year}.csv", index=False)

            count = schedule[
                ["Session1", "Session2", "Session3", "Session4", "Session5"]
            ].notna().sum().sum()

            self.sessions_count += count
            self.logger.debug("Year %d: %d sessions found", year, count)

    def _session_already_downloaded(
        self, year: int, event_name: str, session_name: str
    ) -> bool:
        """
        Check whether a session's lap and weather data already exist on disk.

        Args:
            year: Season year.
            event_name: Grand Prix name.
            session_name: Session name (e.g. 'Practice 1').

        Returns:
            True if both lap and weather CSV files exist, False otherwise.
        """
        laps_file = self.laps_dir / str(year) / event_name / f"{session_name}.csv"
        weather_file = self.weather_dir / str(year) / event_name / f"{session_name}.csv"

        return laps_file.exists() and weather_file.exists()

    def _load_session_with_retry(self, year: int, event: str, session_name: str):
        """
        Load a FastF1 session with retry logic for transient failures.

        Args:
            year: Season year.
            event: Grand Prix name.
            session_name: Session name.

        Returns:
            A loaded FastF1 session object.

        Raises:
            Exception: Re-raises the last exception if all retries fail.
        """
        last_exc = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                session = fastf1.get_session(year, event, session_name)
                session.load(
                    laps=True,
                    weather=True,
                    telemetry=True,
                    messages=False,
                )
                session.laps  # force evaluation
                return session

            except Exception as e:
                last_exc = e
                self.logger.warning(
                    "Attempt %d/%d failed: %d | %s | %s (%s)",
                    attempt,
                    MAX_RETRIES,
                    year,
                    event,
                    session_name,
                    type(e).__name__,
                )

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        raise last_exc

    def download_event(
        self,
        year: int,
        event_name: str,
        sessions: list[str],
        pbar: tqdm,
    ) -> None:
        """
        Download all sessions for a single event.

        Args:
            year: Season year.
            event_name: Grand Prix name.
            sessions: List of session names.
            pbar: Shared tqdm progress bar.
        """
        self.logger.info("Loading event: %d | %s", year, event_name)

        laps_folder = self.laps_dir / str(year) / event_name
        weather_folder = self.weather_dir / str(year) / event_name

        laps_folder.mkdir(exist_ok=True)
        weather_folder.mkdir(exist_ok=True)

        for session_name in sessions:
            if not session_name or session_name == "None":
                pbar.update(1)
                continue

            if self._session_already_downloaded(year, event_name, session_name):
                self.logger.debug(
                    "Skipping existing session: %d | %s | %s",
                    year,
                    event_name,
                    session_name,
                )
                pbar.update(1)
                continue

            try:
                self.logger.debug(
                    "Loading session: %d | %s | %s",
                    year,
                    event_name,
                    session_name,
                )

                session = self._load_session_with_retry(
                    year, event_name, session_name
                )

                laps = session.laps
                # Add track length
                try:
                    fastest_lap = session.laps.pick_fastest()
                    tel = fastest_lap.get_telemetry()
                    track_length = tel['Distance'].max()
                    laps['TrackLength'] = track_length / 1000
                except:
                    self.logger.warning(
                        "Unknown track length for session: %d | %s | %s",
                        year,
                        event_name,
                        session_name,
                    )
                    laps['TrackLength'] = None
                # Add number of race laps
                laps['RaceLaps'] = session.total_laps

                laps.to_csv(
                    laps_folder / f"{session_name}.csv", index=False
                )
                session.weather_data.to_csv(
                    weather_folder / f"{session_name}.csv", index=False
                )

            except fastf1.core.DataNotLoadedError:
                self.logger.error(
                    "Data not loaded: %d | %s | %s",
                    year,
                    event_name,
                    session_name,
                )

            except Exception as e:
                self.logger.exception(
                    "Unexpected error: %d | %s | %s -> %s",
                    year,
                    event_name,
                    session_name,
                    e,
                )

            finally:
                pbar.update(1)

    def download(self) -> None:
        """
        Execute the full download process for all configured seasons.
        """
        self.logger.info(
            "Starting download from %d to %d",
            self.starting_year,
            self.end_year,
        )

        self.get_schedules()

        with tqdm(total=self.sessions_count, desc="Downloading sessions") as pbar:
            for year, schedule in self.schedule.items():
                for _, event in schedule.iterrows():
                    event_name = event["EventName"]
                    sessions = [
                        event[c]
                        for c in ["Session1", "Session2", "Session3", "Session4", "Session5"]
                    ]
                    self.download_event(year, event_name, sessions, pbar)

        self.logger.info("Download completed successfully")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the downloader CLI.

    Returns:
        argparse.Namespace containing parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download FastF1 lap and weather data",
    )

    parser.add_argument(
        "--save-dir",
        type=Path,
        default=SAVE_DIR,
        help="Directory where data will be stored (default: ./data)",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="First season to download (default: 2018)",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last season to download (default: 2025)",
    )

    parser.add_argument(
        "--delete-previous-data",
        action="store_true",
        help="Delete existing data directory before downloading",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.getLogger().setLevel(args.log_level)

    downloader = DataDownloader(
        save_dir=args.save_dir,
        starting_year=args.start_year,
        end_year=args.end_year,
        delete_previous_data=args.delete_previous_data,
    )

    try:
        downloader.download()
    except KeyboardInterrupt:
        downloader.logger.warning(
            "Download interrupted by user (Ctrl+C). Exiting cleanly."
        )
