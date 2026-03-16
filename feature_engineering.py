import numpy as np
import pandas as pd


def create_advanced_features(daily_df):
    # Expect: daily_df index = Date, columns include 'CrimeCount'
    df = daily_df.copy()

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')

    df = df.sort_index()

    # Temporal features
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week.astype(int)

    # Cyclical features
    df['sin_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['cos_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    df['sin_month'] = np.sin(2 * np.pi * df.index.month / 12)
    df['cos_month'] = np.cos(2 * np.pi * df.index.month / 12)
    df['sin_week'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['cos_week'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Special days
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)

    # Lag features
    for lag in [1, 2, 3, 4, 5, 6, 7]:
        df[f'lag_{lag}'] = df['CrimeCount'].shift(lag)

    # Rolling stats
    for window in [3, 5, 7]:
        if len(df) >= window:
            roll = df['CrimeCount'].rolling(window=window)
            df[f'rolling_mean_{window}'] = roll.mean()
            df[f'rolling_std_{window}'] = roll.std()
            df[f'rolling_min_{window}'] = roll.min()
            df[f'rolling_max_{window}'] = roll.max()
        else:
            expd = df['CrimeCount'].expanding()
            df[f'rolling_mean_{window}'] = expd.mean()
            df[f'rolling_std_{window}'] = expd.std()
            df[f'rolling_min_{window}'] = expd.min()
            df[f'rolling_max_{window}'] = expd.max()

    # EMA
    for span in [3, 7]:
        df[f'ema_{span}'] = df['CrimeCount'].ewm(span=span).mean()

    # Trend and change features
    df['trend'] = np.arange(len(df))
    df['trend_squared'] = df['trend'] ** 2
    df['trend_cubed'] = df['trend'] ** 3
    df['daily_change'] = df['CrimeCount'].diff()
    df['pct_change'] = df['CrimeCount'].pct_change()
    df['acceleration'] = df['daily_change'].diff()
    df['z_score'] = (df['CrimeCount'] - df['CrimeCount'].mean()) / df['CrimeCount'].std()

    # Handle missing values from shifts/rolling
    df = df.bfill().ffill().fillna(df.mean(numeric_only=True))

    return df
