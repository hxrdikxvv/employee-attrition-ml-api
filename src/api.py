import os
import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR   = Path(os.getenv("MODEL_DIR", str(ROOT / "results" / "full_pipeline" / "models")))
TABLES_DIR  = ROOT / "results" / "full_pipeline" / "tables"
FIGURES_DIR = ROOT / "results" / "full_pipeline" / "figures"
FRONTEND_DIST = ROOT / "frontend" / "dist"

# ── Validate model directory ────────────────────────────────────────────────
if not MODEL_DIR.is_dir():
    raise FileNotFoundError(
        f"Model directory not found: {MODEL_DIR}. "
        "Set MODEL_DIR env var or ensure results/full_pipeline/models/ exists."
    )

MODEL_FILES: dict[str, Path] = {
    path.stem: path for path in sorted(MODEL_DIR.glob("*.pkl"))
}
if not MODEL_FILES:
    raise FileNotFoundError(f"No .pkl model files found in {MODEL_DIR}")

# ── Load pipeline metadata ───────────────────────────────────────────────────
METADATA_PATH = TABLES_DIR / "pipeline_metadata.json"
if not METADATA_PATH.is_file():
    raise FileNotFoundError(f"pipeline_metadata.json not found at {METADATA_PATH}")

PIPELINE_METADATA: dict = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

# Champion is stored as a short code (e.g. "LR") — map to artifact filename stem
_CHAMPION_SHORT = str(PIPELINE_METADATA.get("champion_model", "lr"))
_SHORT_TO_STEM: dict[str, str] = {
    "LR":  "lr_base",
    "RF":  "rf_base",
    "XGB": "xgb_base",
    "GB":  "gb_base",
    "SVM": "svm_base",
    "ET":  "et_base",
}
DEFAULT_MODEL = _SHORT_TO_STEM.get(_CHAMPION_SHORT.upper(), f"{_CHAMPION_SHORT.lower()}_base")

# Fail closed: silently serving another model would break the paper's LR champion contract.
if DEFAULT_MODEL not in MODEL_FILES:
    raise FileNotFoundError(
        f"Configured champion '{DEFAULT_MODEL}' is missing from {MODEL_DIR}. "
        "Deployment stopped to prevent silently serving a different model."
    )

FEATURES: list[str] = list(PIPELINE_METADATA.get("selected_features", []))
if not FEATURES:
    raise FileNotFoundError(
        f"'selected_features' is empty or missing in {METADATA_PATH}. "
        "Re-run the training pipeline to regenerate pipeline_metadata.json."
    )

THRESHOLD: float = float(PIPELINE_METADATA.get("serving_threshold", 0.50))
TRAINING_NUMERIC_RANGES: dict = PIPELINE_METADATA.get("training_numeric_ranges", {})

# ── Validate that all expected OHE features can be built from form inputs ───
NUMERIC_INPUTS: set[str] = {
    "JobInvolvement", "JobLevel", "MonthlyIncome", "StockOptionLevel",
    "WorkLifeBalance", "YearsWithCurrManager", "JobSatisfaction",
    "EnvironmentSatisfaction", "RelationshipSatisfaction",
}
CATEGORICAL_INPUTS: set[str] = {"BusinessTravel", "Department", "JobRole", "OverTime"}

unsupported_features = [
    name for name in FEATURES
    if name not in NUMERIC_INPUTS
    and not any(name.startswith(f"{cat}_") for cat in CATEGORICAL_INPUTS)
]
if unsupported_features:
    raise ValueError(
        f"Model expects features not collected by the prediction form: {unsupported_features}. "
        "Update NUMERIC_INPUTS / CATEGORICAL_INPUTS in api.py or retrain with compatible features."
    )


# ── Pydantic schema ──────────────────────────────────────────────────────────
class Employee(BaseModel):
    BusinessTravel: Literal["Non-Travel", "Travel_Frequently", "Travel_Rarely"]
    Department: Literal["Human Resources", "Research & Development", "Sales"]
    JobInvolvement: int = Field(ge=1, le=4)
    JobLevel: int = Field(ge=1, le=5)
    JobRole: Literal[
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative",
    ]
    MonthlyIncome: float = Field(ge=0)
    OverTime: Literal["Yes", "No"]
    StockOptionLevel: int = Field(ge=0, le=3)
    WorkLifeBalance: int = Field(ge=1, le=4)
    YearsWithCurrManager: int = Field(ge=0, le=50)
    JobSatisfaction: int = Field(ge=1, le=4)
    EnvironmentSatisfaction: int = Field(ge=1, le=4)
    RelationshipSatisfaction: int = Field(ge=1, le=4)


# ── Auth ─────────────────────────────────────────────────────────────────────
def require_api_key(authorization: str | None = Header(default=None)) -> None:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="API_KEY environment variable is not configured")
    if authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Model loading — no lru_cache to avoid caching HTTPException on failure ──
_model_cache: dict[str, object] = {}

def load_model(model_name: str) -> object:
    """Load and cache a model artifact by name.

    Unlike lru_cache, this cache only stores successfully loaded models, so a
    transient I/O error does not permanently block future calls for the same key.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = MODEL_FILES.get(model_name)
    if model_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{model_name}'. Use GET /models for available IDs."
        )
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load model '{model_name}': {exc}"
        ) from exc

    _model_cache[model_name] = model
    return model


# ── Feature engineering ──────────────────────────────────────────────────────
def build_feature_row(employee: Employee) -> pd.DataFrame:
    values = employee.model_dump()
    row = pd.DataFrame([values])
    categorical = row.select_dtypes(include=["object"]).columns.tolist()
    encoded = pd.get_dummies(row, columns=categorical, drop_first=False)
    # Boolean columns from get_dummies must be cast to int for sklearn compatibility
    bool_cols = encoded.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        encoded[bool_cols] = encoded[bool_cols].astype(int)
    return encoded.reindex(columns=FEATURES, fill_value=0)


def predict_with_model(model_name: str, employee: Employee) -> dict:
    model = load_model(model_name)
    row = build_feature_row(employee)
    try:
        probability = float(model.predict_proba(row)[0, 1])  # type: ignore[union-attr]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed for model '{model_name}': {exc}"
        ) from exc

    values = employee.model_dump()
    out_of_range_fields = [
        field for field, bounds in TRAINING_NUMERIC_RANGES.items()
        if field in values and (
            values[field] < bounds["min"] or values[field] > bounds["max"]
        )
    ]
    return {
        "model": model_name,
        "prediction": "Yes" if probability >= THRESHOLD else "No",
        "attrition_probability": round(probability, 6),
        "threshold": THRESHOLD,
        "out_of_training_range_fields": out_of_range_fields,
    }


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Employee Attrition Prediction API",
    version="1.3.0",
    description=(
        "Research/demo API. Serves pre-trained attrition models from the IBM HR Analytics dataset. "
        "Do not use for actual employment decisions."
    ),
)

# CORS — allow the Vite dev server and any same-origin prod request
_allowed_origins = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Static assets
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")
if FIGURES_DIR.is_dir():
    app.mount("/result-figures", StaticFiles(directory=FIGURES_DIR), name="result-figures")


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    index_file = FRONTEND_DIST / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return FileResponse(ROOT / "src" / "dashboard.html")


def read_table(filename: str) -> list[dict]:
    table_path = TABLES_DIR / filename
    if not table_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Evaluation data unavailable: {filename}"
        )
    return json.loads(pd.read_csv(table_path).to_json(orient="records"))  # type: ignore[return-value]


@app.get("/health")
def health() -> dict:
    if not os.getenv("API_KEY"):
        raise HTTPException(status_code=503, detail="API_KEY is not configured")
    try:
        load_model(DEFAULT_MODEL)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Default model failed to load: {exc}") from exc
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "available_models": len(MODEL_FILES),
    }


@app.get("/models")
def list_models() -> dict:
    return {
        "default_model": DEFAULT_MODEL,
        "threshold": THRESHOLD,
        "models": [{"id": name, "artifact": path.name} for name, path in MODEL_FILES.items()],
    }


@app.get("/api/performance")
def performance() -> dict:
    holdout = read_table("final_summary.csv")
    # Champion short name (e.g. "XGB") must match Model column in final_summary.csv
    champion_short = str(PIPELINE_METADATA.get("champion_model", _CHAMPION_SHORT))
    champion = next((row for row in holdout if row.get("Model") == champion_short), None)
    return {
        "criterion": str(
            PIPELINE_METADATA.get(
                "selection_basis",
                "cross-validation selection; holdout shown separately"
            )
        ),
        "champion": champion,
        "holdout": holdout,
        "all_models": read_table("holdout_all_models.csv"),
        "cross_validation": read_table("cv_5x5_results.csv"),
        "threshold_sweep": read_table("threshold_sweep.csv"),
        "calibration": read_table("calibration.csv"),
        "statistical_tests": read_table("statistical_tests.csv"),
        "figures": {
            "roc_pr":             "/result-figures/roc_pr_all_models.png",
            "confusion_matrices": "/result-figures/confusion_matrices_all.png",
            "decision_curve":     "/result-figures/decision_curve_analysis.png",
        },
    }


@app.get("/api/insights")
def insights() -> dict:
    return {
        "feature_importance": read_table("feature_importance.csv"),
        "shap_importance":    read_table("shap_importance.csv"),
        "subgroup_fairness":  read_table("subgroup_fairness.csv"),
        "error_analysis":     read_table("error_analysis.csv"),
        "selected_features":  read_table("selected_features.csv"),
    }


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(employee: Employee) -> dict:
    return predict_with_model(DEFAULT_MODEL, employee)


@app.post("/predict/{model_name}", dependencies=[Depends(require_api_key)])
def predict_selected(model_name: str, employee: Employee) -> dict:
    return predict_with_model(model_name, employee)


@app.get("/{frontend_path:path}", include_in_schema=False)
def frontend_fallback(frontend_path: str) -> FileResponse:
    index_file = FRONTEND_DIST / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(index_file)
