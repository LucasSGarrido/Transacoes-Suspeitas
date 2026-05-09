from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPLANATIONS_DIR = REPORTS_DIR / "explanations"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"

RAW_DATA_PATH = RAW_DIR / "creditcard.csv"
SCORED_DATA_PATH = PROCESSED_DIR / "scored_transactions.csv"
METRICS_PATH = MODELS_DIR / "metrics_summary.csv"
TEMPORAL_METRICS_PATH = MODELS_DIR / "temporal_metrics_summary.csv"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
THRESHOLD_PATH = MODELS_DIR / "threshold_config.json"
PR_CURVE_PATH = FIGURES_DIR / "precision_recall_curve.csv"
FEATURE_IMPORTANCE_PATH = EXPLANATIONS_DIR / "feature_importance.csv"
LOCAL_EXPLANATIONS_PATH = EXPLANATIONS_DIR / "local_explanations.csv"
EXPLANATION_SUMMARY_PATH = EXPLANATIONS_DIR / "explanation_summary.json"
LATEST_EXPERIMENT_PATH = EXPERIMENTS_DIR / "latest_experiment.json"

TARGET_COLUMN = "Class"
RANDOM_STATE = 42


def ensure_project_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, EXPLANATIONS_DIR, EXPERIMENTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
