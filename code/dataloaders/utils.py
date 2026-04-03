"""
Utility functions for dataloaders.
"""

import json
from pathlib import Path
from typing import Union, List
import numpy as np
import pandas as pd


def normalize_year_input(year: Union[int, List[int]]) -> List[int]:
    """
    Convert year input to a sorted list of integers.
    
    Parameters
    ----------
    year : int or list of int
        Single year (e.g., 2019) or list of years (e.g., [2019, 2020, 2021])
    
    Returns
    -------
    list of int
        Sorted list of years
        
    Examples
    --------
    >>> normalize_year_input(2019)
    [2019]
    >>> normalize_year_input([2021, 2019, 2020])
    [2019, 2020, 2021]
    """
    if isinstance(year, int):
        return [year]
    elif isinstance(year, (list, tuple)):
        return sorted(list(year))
    else:
        raise TypeError(f"year must be int or list of int, got {type(year)}")


def get_available_races(year: int, data_path: Path = None) -> List[str]:
    """
    List all available races for a given year.
    
    Parameters
    ----------
    year : int
        Season year
    data_path : Path, optional
        Path to data directory. Defaults to ./data
    
    Returns
    -------
    list of str
        Race names (circuit names)
    """
    if data_path is None:
        data_path = Path("./data")
    
    year_dir = data_path / "clean_data" / str(year)
    
    if not year_dir.exists():
        return []
    
    return sorted([d.name for d in year_dir.iterdir() if d.is_dir()])


def load_race_data(
    year: int,
    race: str,
    session: str = "Race",
    data_path: Path = None,
) -> pd.DataFrame:
    """
    Load race data from a single race CSV file.
    
    Parameters
    ----------
    year : int
        Season year
    race : str
        Race name (circuit name)
    session : str, optional
        Session type: "Race", "Qualifying", "Practice 1", etc. Default is "Race"
    data_path : Path, optional
        Path to data directory. Defaults to ./data
    
    Returns
    -------
    pd.DataFrame
        Lap data for the race
    """
    if data_path is None:
        data_path = Path("./data")
    
    filepath = data_path / "clean_data" / str(year) / race / f"{session}.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Race data not found: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def load_all_races(
    years: Union[int, List[int]],
    session: str = "Race",
    data_path: Path = None,
) -> pd.DataFrame:
    """
    Load and concatenate race data for all races across specified years.
    
    Parameters
    ----------
    years : int or list of int
        Single year or list of years to load
    session : str, optional
        Session type. Default is "Race"
    data_path : Path, optional
        Path to data directory. Defaults to ./data
    
    Returns
    -------
    pd.DataFrame
        Concatenated lap data from all races in all years
    """
    years_list = normalize_year_input(years)
    
    if data_path is None:
        data_path = Path("./data")
    
    all_data = []
    
    for year in years_list:
        races = get_available_races(year, data_path)
        
        for race in races:
            try:
                df = load_race_data(year, race, session, data_path)
                all_data.append(df)
            except FileNotFoundError:
                # Some races might not have all sessions (e.g., Qualifying might be missing)
                continue
    
    if not all_data:
        raise ValueError(f"No race data found for years {years_list} and session {session}")
    
    return pd.concat(all_data, ignore_index=True)


def load_vocabulary(vocab_name: str, data_path: Path = None) -> dict:
    """
    Load categorical vocabulary from JSON file.
    
    Parameters
    ----------
    vocab_name : str
        Vocabulary name: "Driver", "Team", "Circuit", "Year"
    data_path : Path, optional
        Path to data directory. Defaults to ./data
    
    Returns
    -------
    dict
        Mapping from category string to integer ID
    """
    if data_path is None:
        data_path = Path("./data")
    
    vocab_path = data_path / "vocabs" / f"{vocab_name}.json"
    
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    return vocab


def get_scaler_key(years: Union[int, List[int]]) -> str:
    """
    Generate a unique key for a scaler based on years.
    
    Parameters
    ----------
    years : int or list of int
        Years to generate key for
    
    Returns
    -------
    str
        Hash key based on sorted years, e.g., "2019_2020_2021"
    """
    years_list = normalize_year_input(years)
    return "_".join(map(str, years_list))


def get_numeric_columns() -> List[str]:
    """
    Get list of numeric columns that should be normalized.
    
    Returns
    -------
    list of str
        Column names for numeric features
    """
    return [
        "LapTime",
        "TyreLife",
        "Position",
        "TrackLength",
        "RaceLaps",
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "Pressure",
        "WindSpeed",
        "WindDirection",
        "delta_to_car_ahead",
        "fuel_proxy",
        "stint_lap",
        "cumulative_stint_time",
    ]


def get_categorical_columns() -> List[str]:
    """
    Get list of categorical columns that will be embedded.
    
    Returns
    -------
    list of str
        Column names for categorical features
    """
    return [
        "Driver",
        "Team",
        "Year",
        "Circuit",
    ]


def get_boolean_columns() -> List[str]:
    """
    Get list of boolean columns.
    
    Returns
    -------
    list of str
        Column names for boolean features
    """
    return [
        "FreshTyre",
        "Rainfall",
        "is_outlap",
        "is_inlap",
        "is_pitlap",
        "is_normal_lap",
        "track_clear",
        "yellow_flag",
        "safety_car",
        "red_flag",
        "vsc",
        "vsc_ending",
    ]


def get_compound_columns() -> List[str]:
    """
    Get list of compound one-hot encoded columns.
    
    Returns
    -------
    list of str
        Column names for compound features
    """
    return [
        "compound_soft",
        "compound_medium",
        "compound_hard",
        "compound_unknown",
    ]


def extract_lap_features_vectorized(
    laps: pd.DataFrame,
    numeric_columns: List[str] = None,
    categorical_columns: List[str] = None,
    boolean_columns: List[str] = None,
    compound_columns: List[str] = None,
) -> np.ndarray:
    """
    Extract feature matrix from multiple laps using vectorized operations.

    This is 10-50x faster than calling _get_lap_features() in an iterrows loop.

    Parameters
    ----------
    laps : pd.DataFrame
        DataFrame of laps (one row per lap)
    numeric_columns : list of str, optional
        Numeric column names. Defaults to get_numeric_columns().
    categorical_columns : list of str, optional
        Categorical column names. Defaults to get_categorical_columns().
    boolean_columns : list of str, optional
        Boolean column names. Defaults to get_boolean_columns().
    compound_columns : list of str, optional
        Compound one-hot column names. Defaults to get_compound_columns().

    Returns
    -------
    np.ndarray
        Shape (n_laps, n_features), dtype float32.
        Column order: [numeric, categorical, boolean, compound].
    """
    if numeric_columns is None:
        numeric_columns = get_numeric_columns()
    if categorical_columns is None:
        categorical_columns = get_categorical_columns()
    if boolean_columns is None:
        boolean_columns = get_boolean_columns()
    if compound_columns is None:
        compound_columns = get_compound_columns()

    return np.concatenate([
        laps[numeric_columns].values.astype(np.float32),
        laps[categorical_columns].values.astype(np.float32),
        laps[boolean_columns].values.astype(np.float32),
        laps[compound_columns].values.astype(np.float32),
    ], axis=1)
