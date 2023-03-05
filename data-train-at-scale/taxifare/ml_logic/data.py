import pandas as pd
from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from taxifare.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data by
    - assigning correct dtypes to each colummns
    - removing buggy or irrelevant transactions
    """
    # Compress raw_data by setting types to DTYPES_RAW
    df = df.astype(DTYPES_RAW)

    # remove buggy transactions
    df = df.drop_duplicates()
    df = df.dropna(how='any', axis=0)
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0) |
            (df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    df = df[df.passenger_count > 0]
    df = df[df.fare_amount > 0]

    # Remove geographically irrelevant transactions (rows)
    df = df[df.fare_amount < 400]
    df = df[df.passenger_count < 8]
    df = df[df["pickup_latitude"].between(left=40.5, right=40.9)]
    df = df[df["dropoff_latitude"].between(left=40.5, right=40.9)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-73.7)]
    df = df[df["dropoff_longitude"].between(left=-74.3, right=-73.7)]

    print("✅ data cleaned")

    return df
