# Soft Power Intelligence

Interactive dashboard for national soft power: composite scores, trajectories,
rise/fall rankings, model-attributed drivers, and 5-year Kalman forecasts.

The frontend does not contain mock data. It fetches every view from the FastAPI
backend. Locally, the backend can serve directly from the real pipeline artifacts
in `../output/`; in production, the same API can read from PostgreSQL.

The same backend also serves the Soft Power **agent** endpoint, `POST /agent/query`,
which the project's orchestrator calls. It answers in the shared agent envelope
used by all four agents (see [Agent Endpoint](#agent-endpoint)). The dashboard
does not use it.

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
- `output/country_embeddings.parquet` and `output/artifacts/` (peer similarity)

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
- `GET /api/peers/{iso3}?limit=8`
- `GET /api/global-importance`
- `POST /agent/query` (orchestrator-facing, see below)

## Agent Endpoint

`POST /agent/query` returns the shared envelope that every agent in the project
uses, so the orchestrator can fan out a question and the insight-fusion agent can
rank and corroborate the answers. The contract lives in
`backend/app/envelope.py` and depends only on Pydantic, so other agents can copy
it unchanged. The logic is in `backend/app/agent.py`.

The endpoint always reads from the files in `../output/`, whatever `DATA_SOURCE`
is set to, because it needs per-year Kalman CIs and KPI coverage that the
PostgreSQL schema does not store.

### Request

```json
{"query_type": "profile", "iso3": ["IND", "CHN"], "year": 2024}
{"query_type": "peers",   "iso3": ["KOR"], "limit": 5}
```

| Field | Required | Notes |
|---|---|---|
| `query_type` | yes | `profile` (score, rank, 5-year rank change) or `peers` (most similar countries) |
| `iso3` | yes | 1–25 ISO3 codes. Case-insensitive. The orchestrator resolves names to ISO3 first. |
| `year` | no | `profile` only. Defaults to each country's latest year (2024). |
| `limit` | no | `peers` only. Peers per country, 1–25, default 5. |

### Response

```json
{
  "agent": "soft_power",
  "metadata": {
    "query_type": "profile",
    "year": 2024,
    "entities": ["IND"],
    "data_quality": {"year_coverage": [2000, 2024]},
    "model_version": "kalman-xgb-1.1",
    "data_source": "files"
  },
  "insights": [
    {
      "entity_iso3": "IND",
      "claim": "India ranks #31 of 194 in soft power in 2024 (score 66.5/100), up 5 places since 2019",
      "score": 0.6646,
      "confidence": 0.6535,
      "reason": "Kalman-smoothed composite of 23 KPIs across 5 dimensions; 95% CI width 3.2 pts; 27% of raw KPIs imputed",
      "evidence": {"raw_score": 66.46, "rank": 31, "n_ranked": 194, "rank_change_5y": 5}
    }
  ]
}
```

`evidence` is shortened here. The full version also has `ci_95`, the five
dimension scores, `stability_class`, `trend_slope` and `confidence_factors`.

- `score` is normalised to 0–1 so fusion can compare it with other agents' scores.
  For `profile` it is the soft power score / 100. For `peers` it is the cosine
  similarity. The raw values are in `evidence`.
- For `peers`, `entity_iso3` is the country that was asked about and
  `evidence.peer_iso3` is the peer.
- Unknown codes do not cause an error. They come back in
  `metadata.data_quality.unknown_iso3` with no insights, so one bad code does not
  fail a multi-agent query. A `year` with no data is listed under
  `data_quality.no_data_for_year`.
- Invalid requests (unsupported `query_type`, empty `iso3`, `limit` outside 1–25)
  return `422`.

### Confidence

Confidence is calculated from data quality, not hard-coded:

```
confidence = base × (1 − 0.5 × kpi_missing_share) × (1 − min(0.8, ci_width / 20))
```

- `base` is 0.9 for `profile` (the rankings agree with four external indices in
  `../trust_report.md`) and 0.75 for `peers` (31 of 54 embedding inputs have
  VIF > 10). `peers` skips the CI term.
- `kpi_missing_share` is the share of the 22 raw KPIs that were missing before
  imputation for that country-year.
- `ci_width` is the Kalman 95% CI width in score points.

Each insight's `evidence.confidence_factors` shows the three factors. For example,
in 2024 Germany gets 0.73 (CI width 1.8, 23% imputed) and Monaco gets 0.18
(CI width 13.4, 77% imputed), even though Monaco has the higher score.

### Trying It Out

With the backend running, open `http://127.0.0.1:8000/docs`, expand
`POST /agent/query`, click **Try it out**, and send a request body. The first
request takes about 15 seconds while the data loads; later requests are fast.

From PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/agent/query `
  -ContentType "application/json" `
  -Body '{"query_type":"peers","iso3":["KOR"],"limit":3}' | ConvertTo-Json -Depth 6
```

## Testing

Install the test dependencies once, from `working/`:

```powershell
.venv\Scripts\python.exe -m pip install -r dash\backend\requirements-dev.txt
```

Run the agent endpoint's contract tests (no server or database needed):

```powershell
.venv\Scripts\python.exe -m pytest dash\backend\tests -v
```

These check that every response parses as the shared envelope, that scores and
confidences are between 0 and 1, that unknown countries and years degrade
gracefully, and a few sanity anchors (for example, Japan is among Korea's closest
peers). The pipeline and model checks live one level up in
`../test_pipeline_integrity.py` and `../model_trust_suite.py`.

## Frontend

The dashboard is React + Vite + Recharts. Controls are interactive: country
multi-select, comparison window sliders, metric selector, riser/faller tabs,
country-focused driver attribution, and forecast panels.
