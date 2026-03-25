import pandas as pd
import numpy as np 
from src.secrets import NREL_API_KEY, NREL_EMAIL

def fetch_and_encode_nsrdb_data(lat, lon, year):
    print(f"Fetching NSRDB data for Lat: {lat}, Lon: {lon}, Year: {year}...")

    attributes = "ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle"

    url = (
        f"https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv?"
        f"wkt=POINT({lon}%20{lat})&names={year}&leap_day=false&interval=60&utc=false"
        f"&full_name=Bare+Metal+Dev&email={NREL_EMAIL}&affiliation=Private"
        f"&mailing_list=false&reason=academic&api_key={NREL_API_KEY}&attributes={attributes}"
    )

    # Read directly from URL into Pandas. 
    # skiprows=2 is CRITICAL to bypass the NREL metadata rows.
    df = pd.read_csv(url, skiprows=2)

    # API returns Year, Month, Day, Hour, Minute.

    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df['day_of_year'] = df['datetime'].dt.dayofyear

    # Transform linear time into unit circle
    df['hour_sin'] = np.sin(2 * np.pi * df['Hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour']/24.0)

    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)

    # Drop the raw linear time columns to isolate feature matrix
    df = df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute', 'datetime', 'day_of_year'])

    df = df.rename(columns={
        'ghi': 'GHI',
        'dni': 'DNI',
        'dhi': 'DHI',
        'temp': 'Temperature',
        'wind_spd': 'Wind Speed',
        'solar_zenith_angle': 'Solar Zenith Angle'
    })
    
    return df

def load_data_from_csv(file_path="../data/phoenix_2024.csv"):
    print(f"Loading local dataset from {file_path}...")
    df = pd.read_csv(file_path, skiprows=2)
    features = ['GHI', 'DNI', 'DHI', 'Temperature', 'Wind Speed', 'Solar Zenith Angle']

    return df[features]


def get_nrel_data(source="csv", file_path="../data/phoenix_2024.csv", lat=33.45, lon=-112.04, year=2022):
    
    if source == "csv":
        return load_data_from_csv(file_path)
    elif source == "api":
        return fetch_and_encode_nsrdb_data(lat=lat, lon=lon, year=year)
    else:
        raise ValueError("Invalid source. Choose 'csv' or 'api'.")