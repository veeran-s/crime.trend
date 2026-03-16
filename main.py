from config import *
from data_loader import load_data, aggregate_data
from feature_engineering import create_advanced_features
from modeling import train_models
from prediction import generate_future_predictions
from save_results import save_results


# 1. Load data
df = load_data(DATA_FILE)
daily_series, weekly_series, monthly_series = aggregate_data(df)


# 2. Feature engineering
daily_enhanced = create_advanced_features(daily_series)

# Debug: check what columns are present after feature engineering
print("COLUMNS:", daily_enhanced.columns.tolist())

feature_columns = [col for col in daily_enhanced.columns if col != 'CrimeCount']


# 3. Prepare train/test
test_size = min(7, len(daily_enhanced) // 4)
X_train, X_test = daily_enhanced[feature_columns][:-test_size], daily_enhanced[feature_columns][-test_size:]
y_train, y_test = daily_enhanced['CrimeCount'][:-test_size], daily_enhanced['CrimeCount'][-test_size:]


# 4. Train models and ensemble
individual_results, ensemble, ensemble_pred, ensemble_metrics = train_models(X_train, y_train, X_test, y_test)


# 5. Generate future predictions
future_dates, future_predictions = generate_future_predictions(
    ensemble,
    daily_enhanced,
    feature_columns,
    days_to_predict=30
)


# 6. Save results
save_results(
    ensemble,
    (future_dates, future_predictions),
    feature_columns,
    ensemble_metrics,
    daily_enhanced,
    ENSEMBLE_MODEL_FILE,
    PREDICTIONS_FILE,
    PERFORMANCE_FILE
)
