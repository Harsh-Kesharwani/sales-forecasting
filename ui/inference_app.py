import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import pickle
import mlflow
import xgboost as xgb

sys.path.insert(0, "/usr/local/airflow/include")

from ml_models.ensemble_model import EnsembleModel
from ml_models.model_visualization import ModelVisualizer
from ml_models.model_comparison import ModelComparison

class SimpleModelLoader:
    def __init__(self):
        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.models = {}
        self.scalers = None
        self.encoders = None
        self.feature_cols = None
        self.loaded = False

    def load_models_from_run(self, run_id: str) -> bool:
        try:
            client = mlflow.tracking.MlflowClient()
            local_dir = f"/tmp/mlflow_models/{run_id}"
            os.makedirs(local_dir, exist_ok=True)
            artifacts_path = client.download_artifacts(run_id, "", dst_path=local_dir)

            scalers_path = os.path.join(artifacts_path, "scalers.pkl")
            if os.path.exists(scalers_path):
                try:
                    self.scalers = joblib.load(scalers_path)
                except Exception:
                    with open(scalers_path, 'rb') as f:
                        self.scalers = pickle.load(f)

            encoders_path = os.path.join(artifacts_path, "encoders.pkl")
            if os.path.exists(encoders_path):
                try:
                    self.encoders = joblib.load(encoders_path)
                except Exception:
                    with open(encoders_path, 'rb') as f:
                        self.encoders = pickle.load(f)

            feature_cols_path = os.path.join(artifacts_path, "feature_cols.pkl")
            if os.path.exists(feature_cols_path):
                try:
                    self.feature_cols = joblib.load(feature_cols_path)
                except Exception:
                    with open(feature_cols_path, 'rb') as f:
                        self.feature_cols = pickle.load(f)

            models_dir = os.path.join(artifacts_path, "models")
            xgb_json_path = os.path.join(models_dir, "xgboost", "xgboost_model.json")
            xgb_pkl_path = os.path.join(models_dir, "xgboost", "xgboost_model.pkl")
            if os.path.exists(xgb_json_path):
                xgb_model = xgb.XGBModel()
                xgb_model.load_model(xgb_json_path)
                self.models['xgboost'] = xgb_model
            elif os.path.exists(xgb_pkl_path):
                try:
                    self.models['xgboost'] = joblib.load(xgb_pkl_path)
                except Exception:
                    with open(xgb_pkl_path, 'rb') as f:
                        self.models['xgboost'] = pickle.load(f)

            lgb_path = os.path.join(models_dir, "lightgbm", "lightgbm_model.pkl")
            if os.path.exists(lgb_path):
                try:
                    from lightgbm import Booster
                    self.models['lightgbm'] = joblib.load(lgb_path)
                except Exception:
                    with open(lgb_path, 'rb') as f:
                        self.models['lightgbm'] = pickle.load(f)

            ensemble_path = os.path.join(models_dir, "ensemble", "ensemble_model.pkl")
            if os.path.exists(ensemble_path):
                try:
                    self.models['ensemble'] = joblib.load(ensemble_path)
                except Exception:
                    with open(ensemble_path, 'rb') as f:
                        self.models['ensemble'] = pickle.load(f)

            self.loaded = len(self.models) > 0
            return self.loaded
        except Exception as e:
            st.error(f"Model load error: {e}")
            return False

    def get_latest_run(self):
        try:
            exp = mlflow.get_experiment_by_name("sales_forecasting")
            if not exp:
                return None
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1)
            if len(runs) > 0:
                return runs.iloc[0]['run_id']
            return None
        except Exception:
            return None

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        preds = []
        if 'xgboost' in self.models:
            preds.append(self.models['xgboost'].predict(X))
        if 'lightgbm' in self.models:
            preds.append(self.models['lightgbm'].predict(X))
        if preds:
            return np.mean(preds, axis=0)
        raise ValueError("No models available for prediction")

    def predict(self, X: np.ndarray, model_type: str = 'ensemble') -> np.ndarray:
        if model_type == 'ensemble':
            if 'ensemble' in self.models:
                return self.models['ensemble'].predict(X)
            else:
                return self.predict_ensemble(X)
        elif model_type in self.models:
            return self.models[model_type].predict(X)
        else:
            raise ValueError(f"Model type '{model_type}' not available")


class SimplePredictor:
    def __init__(self, model_loader):
        self.model_loader = model_loader

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter
            df['weekofyear'] = df['date'].dt.isocalendar().week
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        if 'sales' in df.columns:
            for lag in [1, 2, 3, 7, 14, 21, 30]:
                df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
            for window in [3, 7, 14, 21, 30]:
                df[f'sales_rolling_{window}_mean'] = df['sales'].rolling(window).mean()
                df[f'sales_rolling_{window}_std'] = df['sales'].rolling(window).std()
                df[f'sales_rolling_{window}_min'] = df['sales'].rolling(window).min()
                df[f'sales_rolling_{window}_max'] = df['sales'].rolling(window).max()
                df[f'sales_rolling_{window}_median'] = df['sales'].rolling(window).median()
            sales_mean = df['sales'].mean()
            for col in df.columns:
                if 'sales_lag' in col or 'sales_rolling' in col:
                    if 'std' in col:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(sales_mean)
        defaults = {
            'quantity_sold': 100,
            'profit': 1000,
            'has_promotion': 0,
            'customer_traffic': 500,
            'is_holiday': 0,
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val
        return df

    def predict(self, input_data: pd.DataFrame, model_type: str = 'ensemble', forecast_days: int = 30):
        try:
            if not self.model_loader.loaded:
                return {'success': False, 'error': 'Models not loaded'}
            historical_df = self.prepare_features(input_data)
            last_date = pd.to_datetime(input_data['date']).max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D'
            )
            future_df = pd.DataFrame({
                'date': future_dates,
                'store_id': input_data['store_id'].iloc[-1] if 'store_id' in input_data.columns else 'store_001'
            })
            future_df = self.prepare_features(future_df)

            if len(historical_df) > 0 and 'sales' in historical_df.columns:
                recent_sales = historical_df['sales'].tail(30).values
                sales_mean = historical_df['sales'].mean()
                for lag in [1, 2, 3, 7, 14, 21, 30]:
                    future_df[f'sales_lag_{lag}'] = recent_sales[-lag] if len(recent_sales) >= lag else sales_mean
                for window in [3, 7, 14, 21, 30]:
                    if len(recent_sales) >= window:
                        window_data = recent_sales[-window:]
                        future_df[f'sales_rolling_{window}_mean'] = np.mean(window_data)
                        future_df[f'sales_rolling_{window}_std'] = np.std(window_data)
                        future_df[f'sales_rolling_{window}_min'] = np.min(window_data)
                        future_df[f'sales_rolling_{window}_max'] = np.max(window_data)
                        future_df[f'sales_rolling_{window}_median'] = np.median(window_data)
                    else:
                        future_df[f'sales_rolling_{window}_mean'] = sales_mean
                        future_df[f'sales_rolling_{window}_std'] = 0
                        future_df[f'sales_rolling_{window}_min'] = sales_mean
                        future_df[f'sales_rolling_{window}_max'] = sales_mean
                        future_df[f'sales_rolling_{window}_median'] = sales_mean

            if 'store_id' in future_df.columns and future_df['store_id'].dtype == 'object':
                if self.model_loader.encoders and 'store_id' in self.model_loader.encoders:
                    encoder = self.model_loader.encoders['store_id']
                    known_stores = list(encoder.classes_)
                    future_df['store_id'] = future_df['store_id'].apply(lambda x: x if x in known_stores else known_stores[0])
                    future_df['store_id'] = encoder.transform(future_df['store_id'])
                else:
                    if future_df['store_id'].str.contains('store_').any():
                        future_df['store_id'] = future_df['store_id'].str.extract('(\d+)').astype(int)
                    else:
                        future_df['store_id'] = 1

            if self.model_loader.feature_cols:
                for col in self.model_loader.feature_cols:
                    if col not in future_df.columns:
                        future_df[col] = 0
                X = future_df[self.model_loader.feature_cols].values
            else:
                feature_cols = ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend', 'sales_lag_1', 'sales_lag_7']
                X = future_df[[col for col in feature_cols if col in future_df.columns]].values

            if self.model_loader.scalers and 'features' in self.model_loader.scalers:
                try:
                    X = self.model_loader.scalers['features'].transform(X)
                except Exception:
                    pass

            predictions = self.model_loader.predict(X, model_type=model_type)

            if self.model_loader.scalers and 'target' in self.model_loader.scalers:
                try:
                    predictions = self.model_loader.scalers['target'].inverse_transform(predictions.reshape(-1, 1)).flatten()
                except Exception:
                    pass

            results_df = pd.DataFrame({
                'date': future_dates,
                'predicted_sales': predictions,
                'lower_bound': predictions * 0.9,
                'upper_bound': predictions * 1.1
            })

            summary = {
                'total_predicted_sales': predictions.sum(),
                'average_daily_sales': predictions.mean(),
                'max_daily_sales': predictions.max(),
                'min_daily_sales': predictions.min(),
                'forecast_days': forecast_days
            }

            return {'success': True, 'predictions': results_df, 'summary': summary, 'model_type': model_type}
        except Exception as e:
            return {'success': False, 'error': str(e)}


st.set_page_config(page_title="Sales Forecast Inference", page_icon="🔮", layout="wide")

if 'input_data' not in st.session_state:
    st.session_state.input_data = None

if 'model_loader' not in st.session_state:
    st.session_state.model_loader = SimpleModelLoader()
    st.session_state.predictor = SimplePredictor(st.session_state.model_loader)
    st.session_state.models_loaded = False
    st.session_state.run_id = None

st.title("🔮 Sales Forecast Inference")
st.markdown("Generate sales predictions using trained ML models")

with st.sidebar:
    st.header("📦 Model Configuration")
    if not st.session_state.models_loaded:
        st.warning("⚠️ No models loaded")
    else:
        st.success("✅ Models loaded")
        st.info(f"Models: {', '.join(st.session_state.model_loader.models.keys())}")
        if st.session_state.run_id:
            st.caption(f"Run ID: {st.session_state.run_id[:8]}...")
    if st.button("🔄 Load/Reload Models", type="primary", use_container_width=True):
        with st.spinner("Loading models..."):
            run_id = st.session_state.model_loader.get_latest_run()
            if not run_id:
                run_id = "f4b632f644f742ceab8397bccac14da8"
                st.info(f"Using fallback run ID: {run_id[:8]}...")
            if run_id and st.session_state.model_loader.load_models_from_run(run_id):
                st.session_state.models_loaded = True
                st.session_state.run_id = run_id
                st.success("✅ Models loaded!")
                st.experimental_rerun()
            else:
                st.error("❌ Failed to load models")
    st.markdown("---")
    model_type = st.selectbox("Model Type", ["ensemble", "xgboost", "lightgbm"], help="Ensemble combines multiple models")
    forecast_days = st.slider("Forecast Days", min_value=1, max_value=90, value=30)

tab1, tab2 = st.tabs(["📤 Upload Data", "🎲 Sample Data"])

input_data = st.session_state.input_data

with tab1:
    st.markdown("### Upload Historical Sales Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], help="File should contain: date, sales, and optionally store_id")
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.session_state.input_data = input_data
        st.success(f"✅ Loaded {len(input_data)} records")
        with st.expander("Data Preview"):
            st.dataframe(input_data.head())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=input_data['date'], y=input_data['sales'], mode='lines', name='Uploaded Sales Data'))
        fig.update_layout(title="Uploaded Sales Data", xaxis_title="Date", yaxis_title="Sales ($)", height=300)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Generate Sample Data")
    c1, c2, c3 = st.columns(3)
    with c1:
        sample_days = st.number_input("Historical Days", value=60, min_value=7)
    with c2:
        avg_sales = st.number_input("Average Daily Sales", value=5000, min_value=100)
    with c3:
        volatility = st.slider("Volatility (%)", 0, 50, 20)
    if st.button("Generate Sample Data"):
        dates = pd.date_range(end=datetime.now(), periods=sample_days, freq='D')
        trend = np.linspace(0, avg_sales * 0.1, sample_days)
        seasonal = avg_sales * 0.2 * np.sin(2 * np.pi * np.arange(sample_days) / 7)
        noise = np.random.normal(0, avg_sales * volatility / 100, sample_days)
        sales = avg_sales + trend + seasonal + noise
        sales = np.maximum(sales, 0)
        sample_df = pd.DataFrame({'date': dates, 'store_id': 'store_001', 'sales': sales})
        st.session_state.input_data = sample_df  # persist
        input_data = sample_df  # local assign for immediate use
        st.success("✅ Sample data generated")
        with st.expander("Sample Data Preview"):
            st.dataframe(sample_df.head())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample_df['date'], y=sample_df['sales'], mode='lines', name='Sample Sales Data'))
        fig.update_layout(title="Generated Sample Data", xaxis_title="Date", yaxis_title="Sales ($)", height=300)
        st.plotly_chart(fig, use_container_width=True)

if input_data is not None:
    st.markdown("---")
    st.header("📊 Generate Forecast")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🚀 Run Prediction"):
            with st.spinner("Generating forecast..."):
                results = st.session_state.predictor.predict(input_data, model_type=model_type, forecast_days=forecast_days)
                if results['success']:
                    st.success("✅ Forecast generated successfully!")
                    st.markdown("### 📈 Forecast Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Forecast", f"${results['summary']['total_predicted_sales']:,.0f}")
                    col2.metric("Daily Average", f"${results['summary']['average_daily_sales']:,.0f}")
                    col3.metric("Forecast Period", f"{forecast_days} days")
                    col4.metric("Model Used", model_type.upper())
                    st.markdown("### 📊 Forecast Visualization")
                    predictions_df = results['predictions']
                    historical_mask = predictions_df.index < len(input_data)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=predictions_df[historical_mask]['date'], y=input_data['sales'], mode='lines', name='Historical', line=dict(color='blue', width=2)))
                    fig.add_trace(go.Scatter(x=predictions_df[~historical_mask]['date'], y=predictions_df[~historical_mask]['predicted_sales'], mode='lines', name='Forecast', line=dict(color='green', width=3)))
                    fig.add_trace(go.Scatter(x=predictions_df[~historical_mask]['date'], y=predictions_df[~historical_mask]['upper_bound'], fill=None, mode='lines', line_color='rgba(0,255,0,0)', showlegend=False))
                    fig.add_trace(go.Scatter(x=predictions_df[~historical_mask]['date'], y=predictions_df[~historical_mask]['lower_bound'], fill='tonexty', mode='lines', line_color='rgba(0,255,0,0.2)', name='95% Confidence'))
                    fig.update_layout(title="Sales Forecast", xaxis_title="Date", yaxis_title="Sales ($)", hovermode='x unified', height=500, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                    col1, = st.columns(1)
                    with col1:
                        export_df = predictions_df[~historical_mask].copy().round(2)
                        csv = export_df.to_csv(index=False)
                        st.download_button(label="📥 Download Forecast (CSV)", data=csv, file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
                else:
                    st.error(f"❌ Prediction failed: {results['error']}")
else:
    st.warning("⚠️ Please load models using the sidebar before making predictions.")
    st.info("👈 Click 'Load/Reload Models' in the sidebar to begin")
    with st.expander("ℹ️ No models found? Here's what to do:", expanded=True):
        st.markdown("""
        ### First Time Setup
        1. **Open Airflow UI**: [http://localhost:8080](http://localhost:8080)
            - Username: `admin`
            - Password: `admin`
        2. **Run the Training DAG**:
            - Find `sales_forecast_training` in the DAG list
            - Click the play button (▶️) to trigger it
            - Wait for training to complete (5-10 minutes)
        3. **Come back here**:
            - Click "Load/Reload Models" again
            - Models should load successfully

        ### Quick Check
        - **MLflow UI**: [http://localhost:5001](http://localhost:5001)
        - **MinIO UI**: [http://localhost:9001](http://localhost:9001)
            - Username: `minioadmin`
            - Password: `minioadmin`
        """)
