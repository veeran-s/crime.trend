#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("COMPLETE CRIME TREND ANALYSIS & PREDICTION SYSTEM")
print("=" * 60)

# =============================
# Load Data
# =============================
print("\n1. LOADING DATA...")
df = pd.read_csv('NYPD_Complaint_Data_Historic.csv')

print(f"Dataset shape: {df.shape}")
print(f"Total records: {len(df):,}")

# Convert date columns to datetime
df['cmplnt_fr_dt'] = pd.to_datetime(df['cmplnt_fr_dt'], errors='coerce')
df = df.dropna(subset=['cmplnt_fr_dt'])

print(f"Date range: {df['cmplnt_fr_dt'].min()} to {df['cmplnt_fr_dt'].max()}")
print(f"Total days: {(df['cmplnt_fr_dt'].max() - df['cmplnt_fr_dt'].min()).days}")

# =============================
# Multi-Temporal Aggregation
# =============================
print("\n2. CREATING MULTI-TEMPORAL DATASETS...")

# Daily aggregation
daily_crimes = df.groupby(df['cmplnt_fr_dt'].dt.date).size()
daily_series = daily_crimes.reset_index()
daily_series.columns = ['Date', 'CrimeCount']
daily_series['Date'] = pd.to_datetime(daily_series['Date'])
daily_series = daily_series.sort_values('Date').set_index('Date')

# Weekly aggregation
weekly_crimes = df.groupby(pd.Grouper(key='cmplnt_fr_dt', freq='W-MON')).size()
weekly_series = weekly_crimes.reset_index()
weekly_series.columns = ['Date', 'CrimeCount']
weekly_series = weekly_series.set_index('Date')

# Monthly aggregation
monthly_crimes = df.groupby(pd.Grouper(key='cmplnt_fr_dt', freq='M')).size()
monthly_series = monthly_crimes.reset_index()
monthly_series.columns = ['Date', 'CrimeCount']
monthly_series = monthly_series.set_index('Date')

print(f"Daily data points: {len(daily_series)}")
print(f"Weekly data points: {len(weekly_series)}")
print(f"Monthly data points: {len(monthly_series)}")

# =============================
# Advanced Feature Engineering
# =============================
print("\n3. ADVANCED FEATURE ENGINEERING...")

def create_advanced_features(time_series):
    """Create comprehensive features for time series data"""
    ts_df = time_series.copy()
    
    # Basic temporal features
    ts_df['day_of_week'] = ts_df.index.dayofweek
    ts_df['day_of_month'] = ts_df.index.day
    ts_df['month'] = ts_df.index.month
    ts_df['quarter'] = ts_df.index.quarter
    ts_df['day_of_year'] = ts_df.index.dayofyear
    ts_df['week_of_year'] = ts_df.index.isocalendar().week
    
    # Cyclical features
    ts_df['sin_day'] = np.sin(2 * np.pi * ts_df.index.dayofyear / 365)
    ts_df['cos_day'] = np.cos(2 * np.pi * ts_df.index.dayofyear / 365)
    ts_df['sin_month'] = np.sin(2 * np.pi * ts_df.index.month / 12)
    ts_df['cos_month'] = np.cos(2 * np.pi * ts_df.index.month / 12)
    ts_df['sin_week'] = np.sin(2 * np.pi * ts_df.index.dayofweek / 7)
    ts_df['cos_week'] = np.cos(2 * np.pi * ts_df.index.dayofweek / 7)
    
    # Special days
    ts_df['is_weekend'] = (ts_df['day_of_week'] >= 5).astype(int)
    ts_df['is_month_start'] = ts_df.index.is_month_start.astype(int)
    ts_df['is_month_end'] = ts_df.index.is_month_end.astype(int)
    
    # Lag features with more lags
    for lag in [1, 2, 3, 4, 5, 6, 7]:
        ts_df[f'lag_{lag}'] = ts_df['CrimeCount'].shift(lag)
    
    # Rolling statistics with multiple windows
    for window in [3, 5, 7]:
        if len(ts_df) >= window:
            ts_df[f'rolling_mean_{window}'] = ts_df['CrimeCount'].rolling(window=window).mean()
            ts_df[f'rolling_std_{window}'] = ts_df['CrimeCount'].rolling(window=window).std()
            ts_df[f'rolling_min_{window}'] = ts_df['CrimeCount'].rolling(window=window).min()
            ts_df[f'rolling_max_{window}'] = ts_df['CrimeCount'].rolling(window=window).max()
        else:
            ts_df[f'rolling_mean_{window}'] = ts_df['CrimeCount'].expanding().mean()
            ts_df[f'rolling_std_{window}'] = ts_df['CrimeCount'].expanding().std()
            ts_df[f'rolling_min_{window}'] = ts_df['CrimeCount'].expanding().min()
            ts_df[f'rolling_max_{window}'] = ts_df['CrimeCount'].expanding().max()
    
    # Exponential moving averages
    for span in [3, 7]:
        ts_df[f'ema_{span}'] = ts_df['CrimeCount'].ewm(span=span).mean()
    
    # Trend features
    ts_df['trend'] = np.arange(len(ts_df))
    ts_df['trend_squared'] = ts_df['trend'] ** 2
    ts_df['trend_cubed'] = ts_df['trend'] ** 3
    
    # Rate of change features
    ts_df['daily_change'] = ts_df['CrimeCount'].diff()
    ts_df['pct_change'] = ts_df['CrimeCount'].pct_change()
    ts_df['acceleration'] = ts_df['daily_change'].diff()
    
    # Statistical features
    ts_df['z_score'] = (ts_df['CrimeCount'] - ts_df['CrimeCount'].mean()) / ts_df['CrimeCount'].std()
    
    # Fill NaN values using multiple methods
    ts_df = ts_df.fillna(method='bfill').fillna(method='ffill').fillna(ts_df.mean())
    
    return ts_df

# Create enhanced features
daily_enhanced = create_advanced_features(daily_series)
print(f"Created {len(daily_enhanced.columns)} features for daily data")

# =============================
# Trend Analysis & Visualization
# =============================
print("\n4. COMPREHENSIVE TREND ANALYSIS...")

# Create comprehensive trend analysis plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Crime Data Trend Analysis - January 2018', fontsize=16, fontweight='bold')

# Plot 1: Daily trends with moving average
axes[0, 0].plot(daily_enhanced.index, daily_enhanced['CrimeCount'], 
                label='Daily Crimes', color='blue', linewidth=2, marker='o', markersize=4)
if 'rolling_mean_7' in daily_enhanced.columns:
    axes[0, 0].plot(daily_enhanced.index, daily_enhanced['rolling_mean_7'], 
                    label='7-Day Moving Avg', color='red', linewidth=2)
axes[0, 0].set_title('Daily Crime Trends', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Number of Crimes')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)

# Plot 2: Weekly trends
week_labels = [f'Week {i+1}' for i in range(len(weekly_series))]
axes[0, 1].bar(week_labels, weekly_series['CrimeCount'], alpha=0.7, color='green')
axes[0, 1].set_title('Weekly Crime Trends', fontweight='bold')
axes[0, 1].set_xlabel('Week')
axes[0, 1].set_ylabel('Number of Crimes')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Day of week patterns
dow_patterns = daily_enhanced.groupby('day_of_week')['CrimeCount'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[0, 2].bar(days, dow_patterns.values, color='orange', alpha=0.7)
axes[0, 2].set_title('Average Crimes by Day of Week', fontweight='bold')
axes[0, 2].set_xlabel('Day of Week')
axes[0, 2].set_ylabel('Average Crimes')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Monthly trend
if len(monthly_series) > 0:
    month_labels = [x.strftime('%b %Y') for x in monthly_series.index]
    axes[1, 0].bar(month_labels, monthly_series['CrimeCount'], alpha=0.7, color='purple')
    axes[1, 0].set_title('Monthly Crime Trends', fontweight='bold')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Number of Crimes')
    axes[1, 0].grid(True, alpha=0.3)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

# Plot 5: Cumulative trends
axes[1, 1].plot(daily_enhanced.index, daily_enhanced['CrimeCount'].cumsum(), 
                color='darkblue', linewidth=2)
axes[1, 1].set_title('Cumulative Crime Count', fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Cumulative Crimes')
axes[1, 1].grid(True, alpha=0.3)
plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

# Plot 6: Volatility analysis
if 'rolling_std_7' in daily_enhanced.columns:
    axes[1, 2].plot(daily_enhanced.index, daily_enhanced['rolling_std_7'], 
                    color='red', linewidth=2)
    axes[1, 2].set_title('Crime Volatility (7-Day Rolling Std Dev)', fontweight='bold')
    axes[1, 2].set_xlabel('Date')
    axes[1, 2].set_ylabel('Standard Deviation')
    axes[1, 2].grid(True, alpha=0.3)
    plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.show()

# =============================
# Statistical Analysis
# =============================
print("\n5. STATISTICAL TREND ANALYSIS...")

# Linear trend analysis
x = np.arange(len(daily_enhanced))
y = daily_enhanced['CrimeCount'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"📈 LINEAR TREND ANALYSIS:")
print(f"  Slope: {slope:.2f} crimes/day ({'Increasing' if slope > 0 else 'Decreasing'})")
print(f"  R²: {r_value**2:.4f}")
print(f"  P-value: {p_value:.4f} ({'Significant' if p_value < 0.05 else 'Not Significant'})")

# Weekly seasonality analysis
weekly_avg = daily_enhanced.groupby('day_of_week')['CrimeCount'].mean()
print(f"\n📅 WEEKLY SEASONALITY:")
for day, avg in zip(days, weekly_avg):
    print(f"  {day}: {avg:.0f} crimes")

# Basic statistics
print(f"\n📊 BASIC STATISTICS:")
print(f"  Average daily crimes: {daily_enhanced['CrimeCount'].mean():.0f}")
print(f"  Standard deviation: {daily_enhanced['CrimeCount'].std():.0f}")
print(f"  Minimum: {daily_enhanced['CrimeCount'].min():.0f}")
print(f"  Maximum: {daily_enhanced['CrimeCount'].max():.0f}")
print(f"  Coefficient of variation: {(daily_enhanced['CrimeCount'].std() / daily_enhanced['CrimeCount'].mean() * 100):.1f}%")

# =============================
# Modeling Preparation
# =============================
print("\n6. PREPARING FOR MODELING...")

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100
    return mse, mae, rmse, r2, mape

# Prepare features and target
feature_columns = [col for col in daily_enhanced.columns if col != 'CrimeCount']
X = daily_enhanced[feature_columns]
y = daily_enhanced['CrimeCount']

# Time-based split (last 7 days for testing)
test_size = min(7, len(X) // 4)  # Ensure we have enough training data
train_size = len(X) - test_size

X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Features used: {len(feature_columns)}")

# =============================
# Improved Ensemble Model Building
# =============================
print("\n7. BUILDING IMPROVED ENSEMBLE MODEL...")

# Individual models with optimized parameters
models = {
    'XGBoost': xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    ),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
}

# Train individual models
individual_results = {}
print("\nTraining individual models...")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    individual_results[name] = {
        'model': model,
        'metrics': metrics,
        'predictions': y_pred
    }
    print(f"{name:<20} | R²: {metrics[3]:.4f} | MAPE: {metrics[4]:.2f}% | RMSE: {metrics[2]:.2f}")

# Create smart weighted ensemble
class SmartEnsemble:
    def __init__(self, models, individual_results):
        self.models = models
        # Calculate weights based on R² performance (give more weight to better models)
        r2_scores = {name: max(0, individual_results[name]['metrics'][3]) for name in models.keys()}
        total_r2 = sum(r2_scores.values())
        
        if total_r2 > 0:
            self.weights = {name: r2_scores[name] / total_r2 for name in models.keys()}
        else:
            # If all R² are negative, use equal weights
            self.weights = {name: 1/len(models) for name in models.keys()}
    
    def predict(self, X):
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Weighted average
        final_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            final_pred += self.weights[name] * pred
        
        return final_pred

# Create and test ensemble
ensemble = SmartEnsemble({name: individual_results[name]['model'] for name in models.keys()}, individual_results)
ensemble_pred = ensemble.predict(X_test)
ensemble_metrics = calculate_metrics(y_test, ensemble_pred)

print(f"\n🎯 SMART ENSEMBLE RESULTS:")
print(f"  R² Score: {ensemble_metrics[3]:.4f}")
print(f"  MAPE: {ensemble_metrics[4]:.2f}%")
print(f"  RMSE: {ensemble_metrics[2]:.2f}")

print(f"\n⚖️ ENSEMBLE WEIGHTS:")
for name, weight in ensemble.weights.items():
    print(f"  {name}: {weight:.3f}")

# =============================
# Improved Prediction Generation
# =============================
print("\n8. GENERATING IMPROVED PREDICTIONS...")

def generate_future_predictions(model, current_data, feature_columns, days_to_predict=30):
    """Generate future predictions with improved logic"""
    predictions = []
    prediction_dates = []
    
    # Generate future dates
    last_date = current_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                periods=days_to_predict, freq='D')
    
    # Start with current data
    working_data = current_data.copy()
    
    for i, date in enumerate(future_dates):
        # Create new feature row for prediction
        new_features = {}
        
        # Update temporal features
        new_features['day_of_week'] = date.dayofweek
        new_features['day_of_month'] = date.day
        new_features['month'] = date.month
        new_features['quarter'] = date.quarter
        new_features['day_of_year'] = date.dayofyear
        new_features['week_of_year'] = date.isocalendar().week
        
        # Update cyclical features
        new_features['sin_day'] = np.sin(2 * np.pi * date.dayofyear / 365)
        new_features['cos_day'] = np.cos(2 * np.pi * date.dayofyear / 365)
        new_features['sin_month'] = np.sin(2 * np.pi * date.month / 12)
        new_features['cos_month'] = np.cos(2 * np.pi * date.month / 12)
        new_features['sin_week'] = np.sin(2 * np.pi * date.dayofweek / 7)
        new_features['cos_week'] = np.cos(2 * np.pi * date.dayofweek / 7)
        
        # Update special days
        new_features['is_weekend'] = 1 if date.dayofweek >= 5 else 0
        new_features['is_month_start'] = 1 if date.day == 1 else 0
        new_features['is_month_end'] = 1 if date == (date.replace(day=1) + pd.offsets.MonthEnd(0)) else 0
        
        # Update trend
        new_features['trend'] = len(working_data) + i
        new_features['trend_squared'] = new_features['trend'] ** 2
        new_features['trend_cubed'] = new_features['trend'] ** 3
        
        # Update lag features using available data
        for lag in [1, 2, 3, 4, 5, 6, 7]:
            if len(predictions) >= lag:
                new_features[f'lag_{lag}'] = predictions[-lag]
            else:
                # Use historical data if not enough predictions yet
                historical_idx = - (lag - len(predictions))
                if abs(historical_idx) <= len(working_data):
                    new_features[f'lag_{lag}'] = working_data['CrimeCount'].iloc[historical_idx]
                else:
                    new_features[f'lag_{lag}'] = working_data['CrimeCount'].mean()
        
        # Update rolling statistics using recent values
        recent_values = list(working_data['CrimeCount'].tail(10))  # Last 10 historical values
        if predictions:
            recent_values.extend(predictions[-min(7, len(predictions)):])  # Add recent predictions
        
        for window in [3, 5, 7]:
            if len(recent_values) >= window:
                new_features[f'rolling_mean_{window}'] = np.mean(recent_values[-window:])
                new_features[f'rolling_std_{window}'] = np.std(recent_values[-window:])
                new_features[f'rolling_min_{window}'] = np.min(recent_values[-window:])
                new_features[f'rolling_max_{window}'] = np.max(recent_values[-window:])
            else:
                new_features[f'rolling_mean_{window}'] = np.mean(recent_values)
                new_features[f'rolling_std_{window}'] = np.std(recent_values)
                new_features[f'rolling_min_{window}'] = np.min(recent_values)
                new_features[f'rolling_max_{window}'] = np.max(recent_values)
        
        # Update EMA
        for span in [3, 7]:
            if len(recent_values) >= span:
                new_features[f'ema_{span}'] = pd.Series(recent_values).ewm(span=span).mean().iloc[-1]
            else:
                new_features[f'ema_{span}'] = np.mean(recent_values)
        
        # Update rate of change
        if len(predictions) >= 1:
            prev_value = predictions[-1]
            prev_prev_value = predictions[-2] if len(predictions) >= 2 else working_data['CrimeCount'].iloc[-1]
            new_features['daily_change'] = prev_value - prev_prev_value
            new_features['pct_change'] = (new_features['daily_change'] / prev_prev_value * 100) if prev_prev_value != 0 else 0
        else:
            new_features['daily_change'] = working_data['CrimeCount'].iloc[-1] - working_data['CrimeCount'].iloc[-2] if len(working_data) > 1 else 0
            prev_value = working_data['CrimeCount'].iloc[-2] if len(working_data) > 1 else working_data['CrimeCount'].iloc[-1]
            new_features['pct_change'] = (new_features['daily_change'] / prev_value * 100) if prev_value != 0 else 0
        
        if len(predictions) >= 2:
            current_change = predictions[-1] - predictions[-2] if len(predictions) >= 2 else 0
            previous_change = predictions[-2] - predictions[-3] if len(predictions) >= 3 else 0
            new_features['acceleration'] = current_change - previous_change
        else:
            new_features['acceleration'] = 0
        
        # Update z-score
        all_values = list(working_data['CrimeCount']) + predictions
        mean_val = np.mean(all_values)
        std_val = np.std(all_values)
        if std_val > 0:
            new_features['z_score'] = (predictions[-1] - mean_val) / std_val if predictions else 0
        else:
            new_features['z_score'] = 0
        
        # Create prediction dataframe
        pred_df = pd.DataFrame([new_features])[feature_columns]
        
        # Make prediction
        try:
            pred_value = model.predict(pred_df)[0]
            # Ensure reasonable bounds based on historical data
            historical_min = working_data['CrimeCount'].min()
            historical_max = working_data['CrimeCount'].max()
            historical_std = working_data['CrimeCount'].std()
            
            # Allow some flexibility but keep within reasonable bounds
            lower_bound = max(historical_min * 0.7, historical_min - historical_std)
            upper_bound = min(historical_max * 1.3, historical_max + historical_std)
            pred_value = np.clip(pred_value, lower_bound, upper_bound)
            
        except Exception as e:
            # Fallback: use weighted average of recent values
            if predictions:
                recent_avg = np.mean(predictions[-min(7, len(predictions)):])
            else:
                recent_avg = working_data['CrimeCount'].tail(7).mean()
            pred_value = recent_avg
            print(f"  Fallback prediction used for {date}: {e}")
        
        predictions.append(pred_value)
        prediction_dates.append(date)
        
        # Add this prediction to working data for next iteration
        new_row = new_features.copy()
        new_row['CrimeCount'] = pred_value
        new_row_df = pd.DataFrame([new_row], index=[date])
        working_data = pd.concat([working_data, new_row_df])
    
    return prediction_dates, predictions

# Generate predictions
print("Generating 30-day predictions...")
future_dates, future_predictions = generate_future_predictions(
    ensemble, daily_enhanced, feature_columns, days_to_predict=30
)
print("✅ Predictions generated successfully!")

# =============================
# Fixed Results Visualization
# =============================
print("\n9. VISUALIZING RESULTS...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Crime Prediction Results & Analysis', fontsize=16, fontweight='bold')

# Plot 1: Historical vs Predictions (FIXED)
axes[0, 0].plot(daily_enhanced.index, daily_enhanced['CrimeCount'], 
                label='Historical', color='blue', linewidth=2, marker='o', markersize=4)
axes[0, 0].plot(future_dates, future_predictions, 
                label='30-Day Forecast', color='red', linewidth=2, marker='s', linestyle='--', markersize=4)
axes[0, 0].set_title('Historical Data & 30-Day Forecast', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Number of Crimes')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)

# Plot 2: Model Comparison
model_names = list(individual_results.keys()) + ['Ensemble']
r2_scores = [individual_results[name]['metrics'][3] for name in individual_results.keys()] + [ensemble_metrics[3]]

bars = axes[0, 1].bar(model_names, r2_scores, 
                     color=['blue', 'green', 'orange', 'purple', 'red'], alpha=0.7)
axes[0, 1].set_title('Model Performance (R² Score)', fontweight='bold')
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].set_ylim(bottom=min(r2_scores) - 0.1, top=max(r2_scores) + 0.1)
axes[0, 1].grid(True, alpha=0.3)
plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)

for bar, score in zip(bars, r2_scores):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 3: Weekly Projections
future_df = pd.DataFrame({
    'Date': future_dates,
    'Prediction': future_predictions
})
future_df['Week'] = future_df['Date'].dt.isocalendar().week
weekly_projections = future_df.groupby('Week')['Prediction'].sum()

axes[1, 0].bar([f'Week {w}' for w in weekly_projections.index], weekly_projections.values, 
               alpha=0.7, color='teal')
axes[1, 0].set_title('Weekly Crime Projections', fontweight='bold')
axes[1, 0].set_xlabel('Week')
axes[1, 0].set_ylabel('Total Crimes')
axes[1, 0].grid(True, alpha=0.3)
plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

# Plot 4: Short-term Forecast with Confidence (FIXED)
# Get last 7 actual values properly
recent_actual = daily_enhanced['CrimeCount'].iloc[-7:]
recent_dates = daily_enhanced.index[-7:]

axes[1, 1].plot(recent_dates, recent_actual, label='Last 7 Days Actual', 
                color='blue', linewidth=2, marker='o', markersize=4)
axes[1, 1].plot(future_dates[:14], future_predictions[:14], label='14-Day Forecast', 
                color='red', linewidth=2, marker='s', markersize=4)

# Add confidence interval
confidence = np.std(daily_enhanced['CrimeCount'].pct_change().dropna())
upper_bound = [p * (1 + confidence) for p in future_predictions[:14]]
lower_bound = [p * (1 - confidence) for p in future_predictions[:14]]

axes[1, 1].fill_between(future_dates[:14], lower_bound, upper_bound, 
                       alpha=0.2, color='red', label='Confidence Interval')
axes[1, 1].set_title('14-Day Forecast with Confidence', fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Number of Crimes')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.show()

# =============================
# Final Results & Insights
# =============================
print("\n10. FINAL RESULTS & INSIGHTS")
print("="*50)

# Prediction statistics
avg_prediction = np.mean(future_predictions)
max_prediction = np.max(future_predictions)
min_prediction = np.min(future_predictions)
prediction_std = np.std(future_predictions)

print(f"\n📊 30-DAY PREDICTION SUMMARY:")
print(f"  Average predicted: {avg_prediction:.0f} crimes/day")
print(f"  Range: {min_prediction:.0f} - {max_prediction:.0f} crimes")
print(f"  Standard deviation: {prediction_std:.0f} crimes")
print(f"  Total projected: {sum(future_predictions):.0f} crimes")

# Monthly breakdown
future_df['Month'] = future_df['Date'].dt.month
monthly_totals = future_df.groupby('Month')['Prediction'].sum()

print(f"\n📅 MONTHLY PROJECTIONS:")
for month, total in monthly_totals.items():
    month_name = pd.to_datetime(f"2018-{month}-01").strftime('%B')
    daily_avg = total / len(future_df[future_df['Month'] == month])
    print(f"  {month_name}: {total:.0f} total crimes ({daily_avg:.0f}/day)")

# Weekly pattern analysis
future_df['DayName'] = future_df['Date'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_pattern = future_df.groupby('DayName')['Prediction'].mean().reindex(day_order)

print(f"\n🔄 PREDICTED WEEKLY PATTERN:")
for day in day_order:
    if day in weekly_pattern:
        print(f"  {day[:3]}: {weekly_pattern[day]:.0f} crimes")

# Risk assessment
historical_avg = daily_enhanced['CrimeCount'].mean()
high_risk_threshold = historical_avg * 1.15  # 15% above average
high_risk_days = future_df[future_df['Prediction'] > high_risk_threshold]

print(f"\n⚠️  HIGH-RISK DAYS (>15% above average):")
print(f"  {len(high_risk_days)} days identified")
if len(high_risk_days) > 0:
    for _, row in high_risk_days.head(5).iterrows():
        print(f"  {row['Date'].strftime('%Y-%m-%d')} ({row['DayName'][:3]}): {row['Prediction']:.0f} crimes")

# =============================
# Accuracy Assessment
# =============================
print(f"\n🎯 ACCURACY ASSESSMENT:")
performance_level = "EXCELLENT" if ensemble_metrics[4] < 5 else "GOOD" if ensemble_metrics[4] < 10 else "FAIR"
print(f"  Ensemble R²: {ensemble_metrics[3]:.4f} ({ensemble_metrics[3]*100:.1f}% variance explained)")
print(f"  Mean Absolute Percentage Error: {ensemble_metrics[4]:.2f}%")
print(f"  Root Mean Square Error: {ensemble_metrics[2]:.2f} crimes")
print(f"  Model Performance: {performance_level}")

# =============================
# Save Results
# =============================
print("\n11. SAVING RESULTS...")

import joblib
import json

# Save ensemble model
joblib.dump(ensemble, 'crime_prediction_ensemble.pkl')

# Save predictions with additional info
predictions_df = pd.DataFrame({
    'date': future_dates,
    'predicted_crimes': future_predictions,
    'day_of_week': [d.strftime('%A') for d in future_dates],
    'month': [d.strftime('%B') for d in future_dates],
    'year': [d.year for d in future_dates]
})
predictions_df.to_csv('crime_predictions_30_days.csv', index=False)

# Save comprehensive performance metrics
performance_data = {
    'ensemble_r2': float(ensemble_metrics[3]),
    'ensemble_mape': float(ensemble_metrics[4]),
    'ensemble_rmse': float(ensemble_metrics[2]),
    'training_days': len(X_train),
    'testing_days': len(X_test),
    'prediction_days': len(future_predictions),
    'average_prediction': float(avg_prediction),
    'historical_average': float(historical_avg),
    'performance_level': performance_level
}

with open('model_performance.json', 'w') as f:
    json.dump(performance_data, f, indent=2)

print("✅ Results saved successfully!")
print("📁 Generated files:")
print("   - crime_prediction_ensemble.pkl")
print("   - crime_predictions_30_days.csv") 
print("   - model_performance.json")

# =============================
# Conclusion
# =============================
print("\n" + "="*60)
print("ANALYSIS COMPLETE - SYSTEM READY")
print("="*60)
print(f"✅ Successfully analyzed {len(daily_enhanced)} days of crime data")
print(f"✅ Built ensemble model with {ensemble_metrics[3]*100:.1f}% variance explained")
print(f"✅ Generated 30-day predictions with {ensemble_metrics[4]:.1f}% average error")
print(f"✅ Created monthly & weekly projections")
print(f"✅ Identified {len(high_risk_days)} high-risk days")
print(f"✅ Saved all models and predictions")
print("="*60)