import joblib
import json
import pandas as pd
import numpy as np

def save_results(ensemble, predictions, feature_columns, ensemble_metrics, daily_enhanced, 
                 ensemble_file, predictions_file, performance_file):
    joblib.dump(ensemble, ensemble_file)
    future_dates, future_predictions = predictions
    predictions_df = pd.DataFrame({
        'date': future_dates,
        'predicted_crimes': future_predictions,
        'day_of_week': [d.strftime('%A') for d in future_dates],
        'month': [d.strftime('%B') for d in future_dates],
        'year': [d.year for d in future_dates]
    })
    predictions_df.to_csv(predictions_file, index=False)
    avg_prediction = np.mean(future_predictions)
    historical_avg = daily_enhanced['CrimeCount'].mean()
    performance_level = "EXCELLENT" if ensemble_metrics[4] < 5 else "GOOD" if ensemble_metrics[4] < 10 else "FAIR"
    performance_data = {
        'ensemble_r2': float(ensemble_metrics[3]),
        'ensemble_mape': float(ensemble_metrics[4]),
        'ensemble_rmse': float(ensemble_metrics[2]),
        'training_days': len(daily_enhanced) - 7,
        'testing_days': 7,
        'prediction_days': len(future_predictions),
        'average_prediction': float(avg_prediction),
        'historical_average': float(historical_avg),
        'performance_level': performance_level
    }
    with open(performance_file, 'w') as f:
        json.dump(performance_data, f, indent=2)
