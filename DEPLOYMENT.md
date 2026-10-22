# Employee Attrition Model API Deployment

The API serves all 17 trained model artifacts in `results/full_pipeline/models/`. It rebuilds the selected 16 input features from HR fields and returns an attrition probability, predicted label, and threshold. Models are loaded on demand, with only the most recently requested model cached to limit memory use.

## Local API

From the `Employee_Attrition_ML` directory, install `requirements-deploy.txt`, then set an API key and start the service:

```powershell
$env:API_KEY = "replace-with-a-long-random-secret"
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Browse to `http://localhost:8000/docs`. `GET /health` reports service status and the number of artifacts; `GET /models` lists model IDs. `POST /predict` uses the champion `lr_base`; `POST /predict/{model_id}` selects another artifact. Both prediction routes require `Authorization: Bearer <API_KEY>`.

Example model-specific call:

```powershell
$headers = @{ Authorization = "Bearer $env:API_KEY" }
Invoke-RestMethod http://localhost:8000/predict/xgb_base -Method Post -Headers $headers -ContentType "application/json" -Body '{"Age":41,"BusinessTravel":"Travel_Rarely","Department":"Sales","JobInvolvement":3,"JobLevel":2,"JobRole":"Sales Executive","MaritalStatus":"Single","MonthlyIncome":5993,"OverTime":"Yes","StockOptionLevel":0,"WorkLifeBalance":1,"YearsWithCurrManager":5,"JobSatisfaction":4,"EnvironmentSatisfaction":2,"RelationshipSatisfaction":1}'
```

## Render Free Deployment

The included `render.yaml` defines a free Docker web service and `/health` health check. To publish it:

1. Create a private GitHub repository with `Employee_Attrition_ML` as its root and push the project. The included `.gitignore` excludes the raw HR dataset and generated figures/tables while retaining all 17 model artifacts.
2. In Render, create a Blueprint from that repository and select `render.yaml`.
3. Set the `API_KEY` environment variable in Render to a long, random secret. Do not commit the key.
4. Wait for the Docker build, then verify the service's `/health` and `/docs` URLs.

The free instance may spin down while idle, so its first request after inactivity can be delayed. Keep the API key private and use HTTPS for hosted requests. Only send employee data that you are authorized to process.

The Docker image includes the model files and their required inference libraries: scikit-learn, XGBoost, LightGBM, and CatBoost. `libgomp1` is installed for native booster runtimes. Set `MODEL_DIR` only if the artifacts are mounted at a non-default location.