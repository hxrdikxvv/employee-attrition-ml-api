# Employee Attrition Model API Deployment

The deployment serves a simple prediction dashboard (`dashboard.html`) and REST API backed by model artifacts in `results/full_pipeline/models/`.

Logistic Regression (`lr_base.pkl`) is the default champion model. The service accepts an employee profile with standard HR parameters, builds the feature representations, and returns an attrition probability, classification prediction ("Yes" / "No"), and threshold.

## Parameters Accepted by the Prediction Endpoint

The prediction API and dashboard accept the following 15 employee attributes:

| Parameter | Type / Allowed Values | Description |
|-----------|------------------------|-------------|
| `Age` | Integer (18–100) | Employee age |
| `MaritalStatus` | `"Single"`, `"Married"`, `"Divorced"` | Marital status |
| `BusinessTravel` | `"Non-Travel"`, `"Travel_Frequently"`, `"Travel_Rarely"` | Travel frequency |
| `Department` | `"Sales"`, `"Research & Development"`, `"Human Resources"` | Department |
| `JobRole` | 9 standard job roles (e.g. `"Sales Executive"`, `"Research Scientist"`) | Position / Role |
| `JobLevel` | Integer (1–5) | Job level |
| `MonthlyIncome` | Number (>= 0) | Monthly salary |
| `StockOptionLevel` | Integer (0–3) | Stock option level |
| `OverTime` | `"Yes"`, `"No"` | Overtime status |
| `JobInvolvement` | Integer (1–4) | Job involvement (1=Low, 4=High) |
| `WorkLifeBalance` | Integer (1–4) | Work-life balance (1=Low, 4=High) |
| `YearsWithCurrManager` | Integer (0–50) | Years under current manager |
| `JobSatisfaction` | Integer (1–4) | Job satisfaction rating |
| `EnvironmentSatisfaction` | Integer (1–4) | Work environment rating |
| `RelationshipSatisfaction` | Integer (1–4) | Relationship satisfaction rating |

## Local Execution

From `Employee_Attrition_ML`:

```powershell
# 1. Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install deployment dependencies
pip install -r requirements-deploy.txt

# 3. Set a local API key
$env:API_KEY = "local-secret-key"

# 4. Start the server
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

- Open `http://127.0.0.1:8000/` for the prediction dashboard.
- Open `http://127.0.0.1:8000/docs` for Swagger UI.

## Routes Summary

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | — | Prediction web dashboard (`dashboard.html`) |
| `GET` | `/health` | — | Health check & model status |
| `GET` | `/models` | — | Available model IDs and champion info |
| `POST` | `/predict` | Bearer token | Predict attrition using the default champion model |
| `POST` | `/predict/{model_name}` | Bearer token | Predict attrition using a specific model artifact |

## Example Prediction Request (PowerShell)

```powershell
$headers = @{
    Authorization = "Bearer $env:API_KEY"
}

$body = @{
    Age                     = 35
    MaritalStatus           = "Single"
    BusinessTravel          = "Travel_Rarely"
    Department              = "Sales"
    JobInvolvement          = 3
    JobLevel                = 2
    JobRole                 = "Sales Executive"
    MonthlyIncome           = 5993
    OverTime                = "Yes"
    StockOptionLevel        = 0
    WorkLifeBalance         = 1
    YearsWithCurrManager    = 5
    JobSatisfaction         = 4
    EnvironmentSatisfaction = 2
    RelationshipSatisfaction = 1
} | ConvertTo-Json

Invoke-RestMethod http://localhost:8000/predict `
    -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

## Docker & Cloud Deployment (Render)

The simplified `Dockerfile` uses a single Python 3.13-slim stage without Node.js or multi-page build steps.

```powershell
docker build -t employee-attrition-api .
docker run --rm -d -p 8000:8000 -e API_KEY="your-secret-key" employee-attrition-api
```
