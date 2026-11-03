import os
import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT / "results" / "full_pipeline" / "models")))
TABLES_DIR = ROOT / "results" / "full_pipeline" / "tables"

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

if DEFAULT_MODEL not in MODEL_FILES:
    raise FileNotFoundError(
        f"Configured champion '{DEFAULT_MODEL}' is missing from {MODEL_DIR}."
    )

FEATURES: list[str] = list(PIPELINE_METADATA.get("selected_features", []))
if not FEATURES:
    raise FileNotFoundError(
        f"'selected_features' is empty or missing in {METADATA_PATH}."
    )

THRESHOLD: float = float(PIPELINE_METADATA.get("serving_threshold", 0.50))
TRAINING_NUMERIC_RANGES: dict = PIPELINE_METADATA.get("training_numeric_ranges", {})

# ── Supported form inputs ────────────────────────────────────────────────────
NUMERIC_INPUTS: set[str] = {
    "Age", "JobInvolvement", "JobLevel", "MonthlyIncome", "StockOptionLevel",
    "WorkLifeBalance", "YearsWithCurrManager", "JobSatisfaction",
    "EnvironmentSatisfaction", "RelationshipSatisfaction",
}
CATEGORICAL_INPUTS: set[str] = {
    "BusinessTravel", "Department", "JobRole", "MaritalStatus", "OverTime",
}

unsupported_features = [
    name for name in FEATURES
    if name not in NUMERIC_INPUTS
    and not any(name.startswith(f"{cat}_") for cat in CATEGORICAL_INPUTS)
]
if unsupported_features:
    raise ValueError(
        f"Model expects features not collected by the prediction form: {unsupported_features}."
    )


# ── Pydantic schema with original parameter set ─────────────────────────────
class Employee(BaseModel):
    Age: int = Field(ge=18, le=100)
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
    MaritalStatus: Literal["Divorced", "Married", "Single"]
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
        return
    if authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Model loading ────────────────────────────────────────────────────────────
_model_cache: dict[str, object] = {}

def load_model(model_name: str) -> object:
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
    version="1.0.0",
    description="Simple employee attrition prediction service.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    dashboard_file = ROOT / "src" / "dashboard.html"
    if not dashboard_file.is_file():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(dashboard_file)


@app.get("/health")
def health() -> dict:
    try:
        load_model(DEFAULT_MODEL)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Default model failed to load: {exc}") from exc
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "available_models": len(MODEL_FILES),
        "api_key_required": bool(os.getenv("API_KEY")),
    }


@app.get("/models")
def list_models() -> dict:
    return {
        "default_model": DEFAULT_MODEL,
        "threshold": THRESHOLD,
        "models": [{"id": name, "artifact": path.name} for name, path in MODEL_FILES.items()],
    }


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(employee: Employee) -> dict:
    return predict_with_model(DEFAULT_MODEL, employee)


@app.post("/predict/{model_name}", dependencies=[Depends(require_api_key)])
def predict_selected(model_name: str, employee: Employee) -> dict:
    return predict_with_model(model_name, employee)
