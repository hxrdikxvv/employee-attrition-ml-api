import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT / "results" / "full_pipeline" / "models")))
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

app = FastAPI(title="Employee Attrition Prediction API", version="1.0.0")


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


app = FastAPI(title="Employee Attrition Prediction API", version="1.1.0")


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "default_model": DEFAULT_MODEL, "available_models": len(MODEL_FILES)}


@app.get("/models")
def list_models() -> dict[str, object]:
    return {
        "default_model": DEFAULT_MODEL,
        "threshold": THRESHOLD,
        "models": [{"id": name, "artifact": path.name} for name, path in MODEL_FILES.items()],
    }


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(employee: Employee) -> dict[str, str | float]:
    return predict_with_model(DEFAULT_MODEL, employee)


@app.post("/predict/{model_name}", dependencies=[Depends(require_api_key)])
def predict_selected(model_name: str, employee: Employee) -> dict[str, str | float]:
    return predict_with_model(model_name, employee)