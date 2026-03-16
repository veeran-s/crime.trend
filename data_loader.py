import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['cmplnt_fr_dt'] = pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce')
    df = df.dropna(subset=['cmplnt_fr_dt'])
    return df

def aggregate_data(df):
    daily_series = df.groupby(df['cmplnt_fr_dt'].dt.date).size().reset_index()
    daily_series.columns = ['Date', 'CrimeCount']
    daily_series['Date'] = pd.to_datetime(daily_series['Date'])
    daily_series = daily_series.sort_values('Date').set_index('Date')

    weekly_series = df.groupby(pd.Grouper(key='cmplnt_fr_dt', freq='W-MON')).size().reset_index()
    weekly_series.columns = ['Date', 'CrimeCount']
    weekly_series = weekly_series.set_index('Date')

    monthly_series = df.groupby(pd.Grouper(key='cmplnt_fr_dt', freq='ME')).size().reset_index()
    monthly_series.columns = ['Date', 'CrimeCount']
    monthly_series = monthly_series.set_index('Date')

    return daily_series, weekly_series, monthly_series
