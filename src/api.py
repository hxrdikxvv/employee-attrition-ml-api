import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT / "results" / "full_pipeline" / "models")))
TABLES_DIR = ROOT / "results" / "full_pipeline" / "tables"
FIGURES_DIR = ROOT / "results" / "full_pipeline" / "figures"
FRONTEND_DIST = ROOT / "frontend" / "dist"
MODEL_FILES = {path.stem: path for path in sorted(MODEL_DIR.glob("*.pkl"))}
DEFAULT_MODEL = "lr_base"
FEATURES = [
    "OverTime_Yes",
    "MaritalStatus_Single",
    "IsSingle",
    "OverTime_No",
    "StockOptionLevel",
    "MonthlyIncome",
    "SatisfactionScore",
    "YearsWithCurrManager",
    "JobRole_Laboratory Technician",
    "JobInvolvement",
    "HasOvertime",
    "Department_Sales",
    "WorkLifeBalance",
    "Age",
    "BusinessTravel_Travel_Frequently",
    "RiskScore",
]
THRESHOLD = 0.50

if DEFAULT_MODEL not in MODEL_FILES:
    raise FileNotFoundError(f"Default model not found in {MODEL_DIR}: {DEFAULT_MODEL}.pkl")
if not MODEL_FILES:
    raise FileNotFoundError(f"No model files found in {MODEL_DIR}")

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


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="API_KEY is not configured")
    if authorization != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@lru_cache(maxsize=1)
def load_model(model_name: str):
    model_path = MODEL_FILES.get(model_name)
    if model_path is None:
        raise HTTPException(status_code=404, detail="Unknown model. Use GET /models to list available models.")
    return joblib.load(model_path)


def build_feature_row(employee: Employee) -> pd.DataFrame:
    values = employee.model_dump()
    overtime = int(values["OverTime"] == "Yes")
    single = int(values["MaritalStatus"] == "Single")
    low_stock = int(values["StockOptionLevel"] == 0)
    junior = int(values["JobLevel"] <= 1)
    features = {
        "OverTime_Yes": overtime,
        "MaritalStatus_Single": single,
        "IsSingle": single,
        "OverTime_No": 1 - overtime,
        "StockOptionLevel": values["StockOptionLevel"],
        "MonthlyIncome": values["MonthlyIncome"],
        "SatisfactionScore": (
            values["JobSatisfaction"]
            + values["EnvironmentSatisfaction"]
            + values["RelationshipSatisfaction"]
            + values["WorkLifeBalance"]
        ),
        "YearsWithCurrManager": values["YearsWithCurrManager"],
        "JobRole_Laboratory Technician": int(values["JobRole"] == "Laboratory Technician"),
        "JobInvolvement": values["JobInvolvement"],
        "HasOvertime": overtime,
        "Department_Sales": int(values["Department"] == "Sales"),
        "WorkLifeBalance": values["WorkLifeBalance"],
        "Age": values["Age"],
        "BusinessTravel_Travel_Frequently": int(
            values["BusinessTravel"] == "Travel_Frequently"
        ),
        "RiskScore": 3 * overtime + 2 * single + 2 * low_stock + junior,
    }
    return pd.DataFrame([[features[name] for name in FEATURES]], columns=FEATURES)


def predict_with_model(model_name: str, employee: Employee) -> dict[str, str | float]:
    model = load_model(model_name)
    row = build_feature_row(employee)
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "model": model_name,
        "prediction": "Yes" if probability >= THRESHOLD else "No",
        "attrition_probability": round(probability, 6),
        "threshold": THRESHOLD,
    }


app = FastAPI(title="Employee Attrition Prediction API", version="1.2.0")

if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")
if FIGURES_DIR.is_dir():
    app.mount("/result-figures", StaticFiles(directory=FIGURES_DIR), name="result-figures")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    index_file = FRONTEND_DIST / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return FileResponse(ROOT / "src" / "dashboard.html")


def read_table(filename: str) -> list[dict[str, object]]:
    table_path = TABLES_DIR / filename
    if not table_path.is_file():
        raise HTTPException(status_code=503, detail=f"Evaluation data is unavailable: {filename}")
    return json.loads(pd.read_csv(table_path).to_json(orient="records"))


@app.get("/health")
def health() -> dict[str, str | int]:
    if not os.getenv("API_KEY"):
        raise HTTPException(status_code=503, detail="API_KEY is not configured")
    try:
        load_model(DEFAULT_MODEL)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Default model failed to load") from exc
    return {"status": "ok", "default_model": DEFAULT_MODEL, "available_models": len(MODEL_FILES)}


@app.get("/models")
def list_models() -> dict[str, object]:
    return {
        "default_model": DEFAULT_MODEL,
        "threshold": THRESHOLD,
        "models": [{"id": name, "artifact": path.name} for name, path in MODEL_FILES.items()],
    }


@app.get("/api/performance")
def performance() -> dict[str, object]:
    holdout = read_table("final_summary.csv")
    return {
        "criterion": "Highest F1-score on the held-out test set",
        "champion": holdout[0] if holdout else None,
        "holdout": holdout,
        "all_models": read_table("holdout_all_models.csv"),
        "cross_validation": read_table("cv_5x5_results.csv"),
        "threshold_sweep": read_table("threshold_sweep.csv"),
        "calibration": read_table("calibration.csv"),
        "statistical_tests": read_table("statistical_tests.csv"),
        "figures": {
            "roc_pr": "/result-figures/roc_pr_all_models.png",
            "confusion_matrices": "/result-figures/confusion_matrices_all.png",
            "decision_curve": "/result-figures/decision_curve_analysis.png",
        },
    }


@app.get("/api/insights")
def insights() -> dict[str, object]:
    return {
        "feature_importance": read_table("feature_importance.csv"),
        "shap_importance": read_table("shap_importance.csv"),
        "subgroup_fairness": read_table("subgroup_fairness.csv"),
        "error_analysis": read_table("error_analysis.csv"),
        "selected_features": read_table("selected_features.csv"),
    }


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(employee: Employee) -> dict[str, str | float]:
    return predict_with_model(DEFAULT_MODEL, employee)


@app.post("/predict/{model_name}", dependencies=[Depends(require_api_key)])
def predict_selected(model_name: str, employee: Employee) -> dict[str, str | float]:
    return predict_with_model(model_name, employee)


@app.get("/{frontend_path:path}", include_in_schema=False)
def frontend_fallback(frontend_path: str) -> FileResponse:
    index_file = FRONTEND_DIST / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(index_file)
