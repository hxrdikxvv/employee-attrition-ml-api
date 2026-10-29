# Employee Attrition Model API Deployment

The API serves the model artifacts in `results/full_pipeline/models/`. Logistic Regression (`lr_base.pkl`) is the deployment champion, as designated by `RESEARCH_PAPER_FINAL`. It rebuilds the
15 OHE + numeric input features from HR fields and returns an attrition probability, predicted
label, and threshold. Models are loaded on demand and cached in memory after the first request;
a failed load does not permanently block future retries. The `/health` readiness check verifies
that `API_KEY` is configured and the default model can be loaded.

## Local API

From the `Employee_Attrition_ML` directory, create a virtual environment, install the pinned API
dependencies, build the frontend, set a local API key, and start the service:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-deploy.txt
Push-Location frontend
npm ci
npm run build
Pop-Location
$env:API_KEY = "local-development-secret"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Browse to `http://127.0.0.1:8000/` for the AttritionIQ workspace or
`http://127.0.0.1:8000/docs` for the interactive API documentation.

**Routes summary:**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | — | Readiness check; returns `{"status":"ok",...}` |
| `GET` | `/models` | — | Lists available model IDs and serving threshold |
| `GET` | `/api/performance` | — | Holdout and CV evaluation tables + figure URLs |
| `GET` | `/api/insights` | — | Feature importance, SHAP, subgroup, error analysis |
| `POST` | `/predict` | Bearer token | Predict with the champion model (`lr_base`) |
| `POST` | `/predict/{model_id}` | Bearer token | Predict with a specific model artifact |

Both prediction routes require `Authorization: Bearer <API_KEY>`.

### Example prediction call (PowerShell)

```powershell
$headers = @{ Authorization = "Bearer $env:API_KEY" }
$body = @{
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

> **Note:** The model does not accept `Age`, `MaritalStatus`, `Gender`, or any fields not listed
> above. Those attributes were excluded from the feature set during training.

### CORS for separate frontend origins

If running the Vite dev server on `localhost:5173` against the API on `localhost:8000`, the API
already allows `http://localhost:5173` and `http://127.0.0.1:5173` by default. For a deployed
frontend on a different domain, set the `CORS_ORIGINS` environment variable to a
comma-separated list of allowed origins:

```powershell
$env:CORS_ORIGINS = "https://your-frontend.onrender.com"
```

The existing generated threshold, calibration, subgroup, error-analysis, and statistical-test tables are still from XGB; do not interpret them as LR results. They need to be regenerated from the exact paper split before drawing LR conclusions. The saved LR holdout row also differs from the paper's reported LR results, so the artifact bundle and paper are not a reproducibly matched evaluation snapshot.

## Render Free Deployment

The included `render.yaml` defines a free Docker web service and `/health` health check.

1. Create a **private** GitHub repository with `Employee_Attrition_ML` as its root and push the
   project. The included `.gitignore` excludes the raw HR dataset while retaining all 17 model
   artifacts and the evaluation tables/figures served by the app.
2. In Render, create a Blueprint from that repository and select `render.yaml`.
3. Set the `API_KEY` environment variable in Render to a long, random secret.
   **Do not commit the key.**
4. Wait for the Docker build to finish, then open the service root for the prediction workspace;
   `/docs` remains available for technical API documentation.

> The free Render instance **spins down after 15 minutes of inactivity**. The first request
> after a cold start can take 30–60 seconds. This is expected on the free tier.

Keep the API key private and share it only with authorized users. Only send employee data that
you are authorized to process. This single-key setup is intended for a small trusted audience;
use per-user authentication before offering the site to a broad or public audience.

## Docker image

The Dockerfile builds the Vite frontend in a Node stage, then serves its compiled assets from
FastAPI alongside the model API in a Python 3.13-slim image.

```
Node 22-alpine  →  npm ci && npm run build
Python 3.13-slim
  ├── src/api.py
  ├── frontend/dist/          (Vite build output)
  ├── models/                 (17 .pkl artifacts, ~186 MB total)
  ├── results/full_pipeline/tables/
  └── results/full_pipeline/figures/
```

Inference libraries included: scikit-learn, XGBoost, LightGBM, CatBoost.
`libgomp1` is installed for native booster runtimes on Debian slim.

Set `MODEL_DIR` only if artifacts are mounted at a non-default location.

## Deployment verification

From the project root, build and run the same image used by Render. The API fails closed if the
configured champion artifact is missing; it will not silently substitute another model.

```powershell
docker build -t employee-attrition-api .
docker run --rm -d --name employee-attrition-api -p 8000:8000 `
  -e API_KEY="local-development-secret" employee-attrition-api

$health = Invoke-RestMethod http://localhost:8000/health
if ($health.status -ne "ok" -or $health.default_model -ne "lr_base") {
    throw "Deployment health check failed: expected LR champion, got $($health.default_model)"
}
Invoke-RestMethod http://localhost:8000/models
docker stop employee-attrition-api
```

The frontend production build can also be checked without Docker:

```powershell
Push-Location frontend
npm ci
npm run build
Pop-Location
```

The frontend build completed successfully in the current workspace. The Docker image and live
API still need to be checked in an environment with Docker and Python available; no live Render
service URL is configured in this project.
