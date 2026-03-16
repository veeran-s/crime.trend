from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import numpy as np
from utils import calculate_metrics

class SmartEnsemble:
    def __init__(self, models, individual_results):
        self.models = models
        r2_scores = {name: max(0, individual_results[name]['metrics'][3]) for name in models.keys()}
        total_r2 = sum(r2_scores.values())
        self.weights = {name: r2_scores[name]/total_r2 for name in models.keys()} if total_r2>0 else {name:1/len(models) for name in models.keys()}
    
    def predict(self, X):
        final_pred = np.zeros(len(X))
        for name, model in self.models.items():
            final_pred += self.weights[name] * model.predict(X)
        return final_pred

def train_models(X_train, y_train, X_test, y_test):
    models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }

    individual_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        individual_results[name] = {'model': model, 'metrics': metrics, 'predictions': y_pred}

    ensemble = SmartEnsemble({name: individual_results[name]['model'] for name in models.keys()}, individual_results)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_metrics = calculate_metrics(y_test, ensemble_pred)
    return individual_results, ensemble, ensemble_pred, ensemble_metrics
