# Crime Trend Prediction Web App

A Streamlit‑based web application that analyzes historical crime data and predicts future crime trends for New York City using machine learning models.

## 🔗 Live Demo

App link: https://crimetrend-acnhssrjpsntcgz7jqc2yf.streamlit.app/

## ✨ Features

- Uploads and processes NYPD historic crime data.
- Performs feature engineering and model training using multiple ML models.
- Loads pre‑trained models (`.pkl` / `.h5`) for fast predictions.
- Visualizes crime trends and predictions in an interactive Streamlit UI.

## 🧠 Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Scikit‑learn, XGBoost
- TensorFlow / Keras (for LSTM models)
- Matplotlib / Seaborn / Plotly for visualizations

## 🗂 Project Structure (main files)

- `app.py` – Main Streamlit app.
- `main.py`, `model.py`, `modeling.py` – Model training and orchestration scripts.
- `data_loader.py`, `feature_engineering.py`, `prediction.py`, `save_results.py`, `utils.py` – Data loading, preprocessing, prediction utilities.
- `NYPD_Complaint_Data_Historic.csv` – Sample historic crime dataset.
- `crime_predictions_30_days.csv` – Example prediction outputs.
- `crime_lstm_model.h5`, `enhanced_crime_lstm_model.h5`, `*.pkl` – Saved ML models.

## 🚀 How to Run Locally

```bash
# 1) Clone the repo
git clone https://github.com/veeran-s/crime.trend.git
cd crime.trend

# 2) (Optional) Create & activate a virtual environment

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the Streamlit app
streamlit run app.py
