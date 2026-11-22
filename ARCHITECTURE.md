# Deployment Architecture — Cancer Subtype Discovery

## Overview

This project deploys the Cancer Subtype Discovery research platform on Vercel using a
**FastAPI backend + static frontend** architecture.

```
Vercel Deployment
├── api/index.py          ← FastAPI (Python serverless function)
│   └── /api/* endpoints  ← Reads from reports/ CSV/JSON artifacts (bundled)
└── frontend/index.html   ← Static SPA research dashboard (Vercel CDN)
    └── Calls /api/* via fetch()
```

## Why Not Streamlit on Vercel

| Requirement | Streamlit | Vercel Python Runtime |
|---|---|---|
| Persistent process | ✅ Required | ❌ Serverless only |
| WebSocket connection | ✅ Required | ❌ Not persistent |
| Long-running session state | ✅ Required | ❌ Request-scoped |
| Static file serving | ⚠️ Via server | ✅ CDN native |
| Cold start acceptable | ❌ | ✅ |

Vercel's Python runtime executes **serverless functions** (stateless, request-scoped, max 300s on
Hobby plan). Streamlit requires a persistent WebSocket server. There is no supported path to run
Streamlit natively on Vercel.

## Architecture Decision: FastAPI + Static SPA

**Selected: Option A** — FastAPI backend + static HTML/CSS/JS frontend

**Rationale:**
- FastAPI is a valid ASGI application auto-detected by Vercel's Python runtime
- Research artifacts (reports/*.csv, *.json) are bundled at deployment (~8 MB total)
- The static frontend is served as Vercel CDN assets — zero cold-start cost on the UI layer
- All existing research data-loading logic is reused from `app/services/research.py`
- No external services, databases, or object storage required
- Total runtime bundle well within Vercel's 500 MB function limit

## Runtime Bundle Analysis

| Component | Approximate Size |
|---|---|
| `reports/` artifacts | ~8 MB |
| `api/requirements.txt` dependencies (installed) | ~45 MB |
| `frontend/index.html` | ~40 KB |
| `app/`, `src/` Python modules | ~200 KB |
| **Total** | **~53 MB** (well under 500 MB limit) |

**Excluded from deployment** (via `.vercelignore`):
- `.venv/` — Vercel installs from `api/requirements.txt`
- `data/raw/` — raw data not needed at runtime
- `data/annotation/` — pre-processed into biomarker CSVs already
- `reports/enrichment/enrichment_all_results.csv` — 1.8 MB; significant results used instead

## What Is Preserved from the Streamlit Application

All 12 research sections are reproduced in the static frontend:

| # | Section | Streamlit View | Vercel Implementation |
|---|---|---|---|
| 01 | Overview | `research.overview` | `/api/overview` → JS charts |
| 02 | Dataset | `research.dataset` | `/api/dataset` → HTML table |
| 03 | Preprocessing | `research.preprocessing` | Static content (config data) |
| 04 | PCA | `research.pca` | `/api/pca` → Plotly scatter |
| 05 | Clusters | `research.clusters` | `/api/clusters` → bar + table |
| 06 | Validation | `research.validation` | `/api/validation` → 4 tabs |
| 07 | Biomarkers | `discovery.biomarkers` | `/api/biomarkers` → volcano + table |
| 08 | Compare | `discovery.compare` | `/api/clusters` + `/api/phase8` |
| 09 | Pathways | `discovery.pathways` | `/api/pathways` → bubble + table |
| 10 | Explainability | `discovery.explainability` | "Not available" (no SHAP artifacts) |
| 11 | Reports | `reporting.reports` | `/api/reports` → artifact status |
| 12 | Methods | `reporting.methods` | `/api/methods` → static JSON |

## ML Model Audit

**Result: No trained prediction model files exist in this repository.**

The project is correctly classified as **unsupervised clustering / subtype discovery research**.
No `.pkl`, `.joblib`, or other model files were found outside the Python venv test fixtures.

The API never claims to perform clinical classification or new-sample prediction.
Explainability artifacts (SHAP) are documented as "not available" with explanation of what
would be required to generate them.

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/health` | Application health check |
| `GET /api/overview` | Dataset summary and key metrics |
| `GET /api/dataset` | Sample and probe information |
| `GET /api/pca` | PCA coordinates and variance |
| `GET /api/clusters` | Cluster assignments and summary |
| `GET /api/validation` | ARI, silhouette, PAC, bootstrap |
| `GET /api/biomarkers` | Candidate probes (filterable by cluster, FDR, effect) |
| `GET /api/pathways` | Pathway enrichment results (filterable) |
| `GET /api/phase8` | Phase 8 subtype characterization |
| `GET /api/reports` | Artifact availability status |
| `GET /api/methods` | Methodology documentation |

## Differences from Streamlit Application

| Feature | Streamlit App | Vercel Deployment |
|---|---|---|
| Interactive widgets | `st.selectbox`, `st.slider` | HTML `<select>`, JS event listeners |
| Caching | `@st.cache_data` | Module-level Python singleton |
| PDF generation | `reportlab` via `build_report()` | Not available (requires persistent server) |
| ZIP export | `zipfile` in-memory | Not available |
| Consensus heatmap | 72×72 matrix | Not present (file missing from repo) |
| Expression explorer | Full 72×2000 matrix | Via `/api/pca` (coordinates only) |
| Navigation | `st.query_params` | JavaScript SPA routing |
| Session state | `st.session_state` | JavaScript module state |

## Security

- CORS: `allow_origins=["*"]` — research project, public data, no auth needed
- No secrets in source code (`.gitignore` excludes `.env`, `secrets.toml`)
- No arbitrary filesystem access — only pre-approved report paths
- No unsafe file uploads
- Internal paths not exposed in API error messages
- Scientific disclaimer included in every API response

## Known Limitations

1. **No PDF/ZIP report generation** — requires persistent server process
2. **No interactive expression matrix exploration** — 878 KB file not served per-request
3. **No consensus matrix heatmap** — `consensus_k3_matrix.csv` missing from repository
4. **No explainability (SHAP)** — surrogate classifier artifacts not generated
5. **Cold starts** — first request after inactivity may take 2–5 seconds
6. **Stateless** — no session persistence between browser reloads

## Local Development

```bash
# Backend
cd cancer-subtype-discovery
pip install -r api/requirements.txt
uvicorn api.index:app --reload --port 8000

# Frontend — open in browser
open frontend/index.html
# Note: set API_BASE = 'http://localhost:8000' in frontend/index.html for local dev

# Original Streamlit app (unchanged)
python run.py
```

## Vercel Deployment

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy (from project root)
vercel --prod

# Or connect GitHub repo in Vercel dashboard:
# https://vercel.com/new → Import Git Repository
# → Ritik-pixel28/cancer-subtype-discovery
# → Framework: Other
# → Build: (none needed for static)
# → Output: frontend/
```
