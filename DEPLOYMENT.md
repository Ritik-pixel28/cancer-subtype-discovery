# Vercel Deployment & Production Operations Guide

> **Project:** Cancer Subtype Discovery
> **Target Platform:** Vercel (Python Serverless Runtime + Edge CDN)
> **Architecture:** Decoupled FastAPI Backend (`/api/*`) + Responsive Static SPA (`/`)

---

## 1. Architecture Overview

```
                          ┌───────────────────────────┐
                          │     User / Client         │
                          └─────────────┬─────────────┘
                                        │
                         https://<project>.vercel.app
                                        │
                   ┌────────────────────┴────────────────────┐
                   │                                         │
        Static Assets (CDN)                       API Requests (/api/*)
                   │                                         │
                   ▼                                         ▼
        ┌─────────────────────┐                   ┌─────────────────────┐
        │  frontend/          │                   │  api/index.py       │
        │  index.html         │                   │  FastAPI (Python    │
        │  (Plotly.js + CSS)  │                   │  Serverless Function)│
        └─────────────────────┘                   └──────────┬──────────┘
                                                             │
                                                  ┌──────────┴──────────┐
                                                  │ reports/ Artifacts  │
                                                  │ - PCA coordinates   │
                                                  │ - Clusters (k=3)    │
                                                  │ - Biomarkers        │
                                                  │ - ORA Pathways      │
                                                  │ - Phase 8 Summary   │
                                                  └─────────────────────┘
```

### Why Decoupled FastAPI + Static Frontend?
1. **Serverless Execution:** Vercel functions execute in short-lived serverless containers. Traditional Streamlit requires a continuous WebSocket server process, making native Vercel hosting technically invalid for Streamlit.
2. **Speed & Efficiency:** Static UI assets load instantly from Vercel's global CDN (edge locations).
3. **Reproducible Research Preservation:** The FastAPI backend directly queries validated research artifacts (`reports/`), preserving all 9 research phases without rerunning computationally expensive pipelines.

---

## 2. Verified API Endpoints

All endpoints include a mandatory scientific disclaimer in the response.

| Method | Endpoint | Description | Query Parameters |
|---|---|---|---|
| `GET` | `/api/health` | Service health status and disclaimer | None |
| `GET` | `/api/overview` | Cohort metrics, cluster counts, variance | None |
| `GET` | `/api/dataset` | Probe dimensions, sample counts, quality | None |
| `GET` | `/api/pca` | 30-component coordinates and variance | `limit` (int, default 200) |
| `GET` | `/api/clusters` | 72 sample consensus assignments & summary | None |
| `GET` | `/api/validation` | Silhouette, ARI, bootstrap, PAC, gap | None |
| `GET` | `/api/biomarkers` | Candidate biomarkers with FDR & effect size | `cluster`, `max_fdr`, `min_effect`, `limit` |
| `GET` | `/api/pathways` | Phase 7 ORA pathway enrichment results | `cluster`, `limit` |
| `GET` | `/api/phase8` | Subtype characterization & cross-cluster data | None |
| `GET` | `/api/reports` | Inventory of all validated research artifacts | None |
| `GET` | `/api/methods` | Complete statistical methodology & limitations | None |

---

## 3. Deployment Instructions

### Method A: Automated Deployment via GitHub Integration (Recommended)

1. Push your repository to GitHub:
   ```bash
   git push origin main
   ```
2. Navigate to [vercel.com/new](https://vercel.com/new).
3. Select and import the `cancer-subtype-discovery` repository.
4. **Project Settings:**
   - **Framework Preset:** `Other`
   - **Root Directory:** `./`
   - **Build Command:** Leave blank (no build step needed for pure static + Python)
   - **Output Directory:** `public` (or leave default; handled automatically by `vercel.json`)
5. Click **Deploy**.
6. Vercel automatically:
   - Detects `vercel.json` routing configuration.
   - Installs Python dependencies from `api/requirements.txt` (or root `requirements.txt`).
   - Packages the FastAPI handler into an AWS Lambda container.
   - Deploys static assets to the global edge network.

---

### Method B: Deployment via Vercel CLI

1. Authenticate with Vercel:
   ```bash
   npx vercel login
   ```
2. Link project (first time only):
   ```bash
   npx vercel link
   ```
3. Deploy to production:
   ```bash
   npx vercel --prod
   ```
4. Follow the prompt; upon completion, Vercel will output your live URL:
   ```
   Production: https://cancer-subtype-discovery-<unique-hash>.vercel.app
   ```

---

## 4. Local Development & Verification

### Running the API Backend Locally
```bash
# Activate your environment
source .venv/bin/activate

# Launch FastAPI development server
uvicorn api.index:app --reload --port 8000
```
Test with curl or browser:
```bash
curl http://localhost:8000/api/health
```

### Running the Automated Test Suite
```bash
pytest -q
# Expected output: 156 passed
```

### Running the Legacy Streamlit Explorer
The original Streamlit application remains completely intact and functional:
```bash
python run.py
# or
streamlit run app/streamlit_app.py
```

---

## 5. Machine Learning & Research Boundary Audit

> [!IMPORTANT]
> **No Supervised Predictive Model Exists in this Repository.**
> - The project performs **unsupervised clustering and post-hoc subtype characterization**.
> - There are **no trained model weights, estimators, joblib/pickle classifiers, or diagnostic inference pipelines**.
> - The API and frontend explicitly present exploratory subgroup discovery findings. They do not accept patient sample uploads for clinical classification or diagnosis.

---

## 6. Scientific Disclaimer

*This application is intended strictly for research and educational purposes. It is not a clinical diagnostic tool and should not be used to make medical decisions, predict patient outcomes, or select therapeutic regimens.*
