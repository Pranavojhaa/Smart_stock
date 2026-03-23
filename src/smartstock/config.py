from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

RAW_DATA_PATH = DATA_RAW_DIR / "synthetic_retail_demand.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "retail_demand_features.csv"
FORECASTS_PATH = DATA_PROCESSED_DIR / "latest_forecasts.csv"
RECOMMENDATIONS_PATH = DATA_PROCESSED_DIR / "latest_recommendations.csv"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
MODEL_BUNDLE_PATH = MODELS_DIR / "smartstock_bundle.joblib"

DEFAULT_HORIZONS = (7, 28)
DEFAULT_RANDOM_SEED = 42

