import pandas as pd
import numpy as np


def generate_future_predictions(model, current_data, feature_columns, days_to_predict=30):
    predictions = []
    prediction_dates = []

    # current_data: index = Date, column 'CrimeCount' + all feature columns
    working_data = current_data.copy()
    last_date = working_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=days_to_predict,
                                 freq='D')

    for i, date in enumerate(future_dates):
        new_features = {}

        # --- basic date features ---
        new_features['day_of_week'] = date.dayofweek
        new_features['day_of_month'] = date.day
        new_features['month'] = date.month
        new_features['quarter'] = (date.month - 1) // 3 + 1
        new_features['day_of_year'] = date.timetuple().tm_yday
        new_features['week_of_year'] = date.isocalendar().week

        # --- cyclical features ---
        new_features['sin_day'] = np.sin(2 * np.pi * new_features['day_of_year'] / 365)
        new_features['cos_day'] = np.cos(2 * np.pi * new_features['day_of_year'] / 365)
        new_features['sin_month'] = np.sin(2 * np.pi * new_features['month'] / 12)
        new_features['cos_month'] = np.cos(2 * np.pi * new_features['month'] / 12)
        new_features['sin_week'] = np.sin(2 * np.pi * new_features['day_of_week'] / 7)
        new_features['cos_week'] = np.cos(2 * np.pi * new_features['day_of_week'] / 7)

        # --- special days ---
        new_features['is_weekend'] = 1 if new_features['day_of_week'] >= 5 else 0
        new_features['is_month_start'] = 1 if new_features['day_of_month'] == 1 else 0
        month_end = (date.replace(day=1) + pd.offsets.MonthEnd(0)).date()
        new_features['is_month_end'] = 1 if date.date() == month_end else 0

        # --- lag / rolling source series ---
        recent = working_data['CrimeCount']

        # lag features: lag_1 ... lag_7
        for lag in [1, 2, 3, 4, 5, 6, 7]:
            key = f'lag_{lag}'
            if len(recent) >= lag:
                new_features[key] = recent.iloc[-lag]
            else:
                new_features[key] = recent.iloc[-1]

        # rolling stats: windows 3,5,7
        for window in [3, 5, 7]:
            series = recent.tail(window) if len(recent) >= window else recent
            new_features[f'rolling_mean_{window}'] = series.mean()
            new_features[f'rolling_std_{window}'] = series.std()
            new_features[f'rolling_min_{window}'] = series.min()
            new_features[f'rolling_max_{window}'] = series.max()

        # EMA: 3, 7
        for span in [3, 7]:
            new_features[f'ema_{span}'] = recent.ewm(span=span).mean().iloc[-1]

        # trend / change
        n = len(working_data) + i
        new_features['trend'] = n
        new_features['trend_squared'] = n ** 2
        new_features['trend_cubed'] = n ** 3

        if len(recent) >= 2:
            last_val = recent.iloc[-1]
            prev_val = recent.iloc[-2]
            daily_change = last_val - prev_val
        elif len(recent) == 1:
            last_val = recent.iloc[-1]
            daily_change = 0
        else:
            last_val = 0
            daily_change = 0

        new_features['daily_change'] = daily_change
        new_features['pct_change'] = (daily_change / last_val) if last_val != 0 else 0

        if len(recent) >= 3:
            prev_change = recent.iloc[-1] - recent.iloc[-2]
            pre_prev_change = recent.iloc[-2] - recent.iloc[-3]
            new_features['acceleration'] = prev_change - pre_prev_change
        else:
            new_features['acceleration'] = 0

        # z_score – simple placeholder (model already learned from training z_scores)
        new_features['z_score'] = 0

        # --- build DataFrame with correct columns order ---
        pred_df = pd.DataFrame([new_features])[feature_columns]

        pred_value = model.predict(pred_df)[0]
        predictions.append(pred_value)
        prediction_dates.append(date)

        # Update working_data for next step
        new_row = new_features.copy()
        new_row['CrimeCount'] = pred_value
        working_data = pd.concat(
            [working_data, pd.DataFrame([new_row], index=[date])]
        )

    return prediction_dates, predictions
