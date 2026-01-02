import fastf1
import os
import shutil
from tqdm import tqdm

SAVE_DIR = "./data"
CACHE_DIR = ".cache"

fastf1.set_log_level('ERROR')
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)
    fastf1.Cache.enable_cache(CACHE_DIR)


class DataDownloader():
    

    def __init__(self, save_dir,  starting_year=2018, end_year=2025 , delete_previous_data=False):

        self.starting_year = starting_year
        self.end_year = end_year
        self.save_dir = save_dir
        self.schedule_dir = os.path.join(save_dir, "schedule")
        self.laps_dir = os.path.join(save_dir, "laps")
        self.weather_dir = os.path.join(save_dir, "weather")
        self.schedule = dict()
        self.sessions_count = 0

        # create data folder architecture
        if delete_previous_data and os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
            os.mkdir(self.schedule_dir)
            os.mkdir(self.weather_dir)
            os.mkdir(self.laps_dir)
            for year in range(self.starting_year, end_year + 1):
                os.mkdir(os.path.join(self.laps_dir, str(year)))
                os.mkdir(os.path.join(self.weather_dir, str(year)))


    def get_schedules(self):

        for year in range(self.starting_year, self.end_year+1):
            schedule = fastf1.get_event_schedule(year, include_testing = False)
            self.schedule[year] = schedule
            schedule.to_csv(os.path.join(self.schedule_dir, f"schedule_{year}.csv"), sep=",", index=False)
            self.sessions_count += 5 * schedule.shape[0]

        
    def download_event(self, year:int, event_name: str, sessions: list, pbar: tqdm):

        laps_foler = os.path.join(self.laps_dir, str(year), event_name)
        os.mkdir(laps_foler)

        weather_folder = os.path.join(self.weather_dir, str(year), event_name)
        os.mkdir(weather_folder)

        for s in sessions:
            if s != 'None' and s != '':
                session = fastf1.get_session(year, event_name, s)
                session.load(laps=True, weather=True, telemetry=False, messages=False)
                try:
                    session.laps.to_csv(os.path.join(laps_foler, f'{s}.csv') ,sep=",", index=False)
                    session.weather_data.to_csv(os.path.join(weather_folder, f'{s}.csv') ,sep=",", index=False)
                except :
                    print(f"Error while downloading {year} - {event_name} - {s}")

            pbar.update(1)
                

    def download(self):
        
        self.get_schedules()
        with tqdm(total=self.sessions_count) as pbar:
            for year in range(self.starting_year, self.end_year+1):
                for i, event in self.schedule[year].iterrows():
                    sessions = event[["Session1", "Session2", "Session3", "Session4", "Session5"]].to_list()
                    print(f"Downloading data from {year} - {event["EventName"]}")
                    self.download_event(year, event["EventName"], sessions, pbar)


if __name__ == "__main__":
    downloader = DataDownloader(save_dir=SAVE_DIR, starting_year=2018, end_year=2025, delete_previous_data=True)
    downloader.download()