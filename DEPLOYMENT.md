# Employee Attrition Model API Deployment

The API serves the model artifacts in `results/full_pipeline/models/`. It rebuilds the selected 16 input features from HR fields and returns an attrition probability, predicted label, and threshold. Models are loaded on demand, with only the most recently requested model cached to limit memory use. The `/health` readiness check verifies that `API_KEY` is configured and the default model can be loaded.

## Local API

From the `Employee_Attrition_ML` directory, create a virtual environment, install the pinned API dependencies, build the frontend, then set a local API key and start the service:

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

Browse to `http://127.0.0.1:8000/` for the AttritionIQ workspace or `http://127.0.0.1:8000/docs` for technical API documentation. The workspace lets you enter an employee profile, select a model, and view the estimated likelihood. It asks for the API key at request time and does not save it. `GET /health` reports readiness; `GET /models` lists model IDs; `GET /api/performance` and `GET /api/insights` return saved evaluation results. `POST /predict` uses the champion `lr_base`; `POST /predict/{model_id}` selects another artifact. Both prediction routes require `Authorization: Bearer <API_KEY>`.

Example model-specific call:

```powershell
$headers = @{ Authorization = "Bearer $env:API_KEY" }
Invoke-RestMethod http://localhost:8000/predict/xgb_base -Method Post -Headers $headers -ContentType "application/json" -Body '{"Age":41,"BusinessTravel":"Travel_Rarely","Department":"Sales","JobInvolvement":3,"JobLevel":2,"JobRole":"Sales Executive","MaritalStatus":"Single","MonthlyIncome":5993,"OverTime":"Yes","StockOptionLevel":0,"WorkLifeBalance":1,"YearsWithCurrManager":5,"JobSatisfaction":4,"EnvironmentSatisfaction":2,"RelationshipSatisfaction":1}'
```

## Render Free Deployment

The included `render.yaml` defines a free Docker web service and `/health` health check. To publish it:

1. Create a private GitHub repository with `Employee_Attrition_ML` as its root and push the project. The included `.gitignore` excludes the raw HR dataset while retaining all 17 model artifacts and the evaluation tables/figures served by the app.
2. In Render, create a Blueprint from that repository and select `render.yaml`.
3. Set the `API_KEY` environment variable in Render to a long, random secret. Do not commit the key.
4. Wait for the Docker build, then open the service root for the prediction workspace; `/docs` remains available for technical API documentation.

The free instance may spin down while idle, so its first request after inactivity can be delayed. Keep the API key private, share it only with authorized users, and use HTTPS for hosted requests. Only send employee data that you are authorized to process. This single-key setup is intended for a small trusted audience; use per-user authentication before offering the site to a broad or public audience.

The Docker image builds the Vite frontend in a Node stage, then serves its compiled assets from FastAPI with the model API. The `.dockerignore` excludes the raw dataset, Git metadata, and local frontend dependencies from the build context. The image includes model artifacts, evaluation tables, and generated figures used by the performance and insights pages. Inference libraries include scikit-learn, XGBoost, LightGBM, and CatBoost. `libgomp1` is installed for native booster runtimes. Set `MODEL_DIR` only if the artifacts are mounted at a non-default location.
