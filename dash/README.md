# Soft Power Intelligence

Interactive dashboard for national soft power: composite scores, trajectories,
rise/fall rankings, model-attributed drivers, and 5-year Kalman forecasts.

The frontend does not contain mock data. It fetches every view from the FastAPI
backend. Locally, the backend can serve directly from the real pipeline artifacts
in `../output/`; in production, the same API can read from PostgreSQL.

## Quick Start With Real Artifacts

Run the backend from `dash/backend`:

```bash
DATA_SOURCE=files uvicorn app.main:app --reload
```

On Windows PowerShell:

```powershell
$env:DATA_SOURCE = "files"
uvicorn app.main:app --reload
```

Then run the frontend from `dash/frontend`:

```bash
npm install
npm run dev
```

Open the local Vite URL. The frontend uses `http://localhost:8000` by default
through `frontend/.env.development`.

## Data Sources

`DATA_SOURCE=files` reads the actual modeling outputs:

- `output/master_soft_power_panel.csv`
- `output/kalman_results.csv`
- `output/kalman_summary.csv`
- `output/kalman_forecast_5yr.csv`
- `output/shap_country.csv`
- `output/shap_global.csv`
- `output/country_reference_table.csv`

`DATA_SOURCE=auto` is the default. It tries PostgreSQL first, then falls back to
the real files if the database or SQLAlchemy is unavailable.

`DATA_SOURCE=db` requires PostgreSQL and fails if the database cannot be reached.

## PostgreSQL Mode

Bring up Postgres and the API:

```bash
docker compose up -d
docker compose exec api python db/load_data.py
```

The loader imports `backend/seed_data/*.json` into PostgreSQL. Use this mode for
deployments where you want a managed database behind the API.

## API Endpoints

- `GET /health`
- `GET /api/countries`
- `GET /api/latest`
- `GET /api/timeseries?iso3=USA&iso3=CHN&yStart=2010&yEnd=2024`
- `GET /api/deltas?yStart=2010&yEnd=2024`
- `GET /api/drivers/{iso3}`
- `GET /api/forecast/{iso3}`
- `GET /api/global-importance`

## Frontend

The dashboard is React + Vite + Recharts. Controls are interactive: country
multi-select, comparison window sliders, metric selector, riser/faller tabs,
country-focused driver attribution, and forecast panels.
