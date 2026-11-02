# LandScan v2

Land due diligence reports for realtors and builders, powered by Google DeepMind's AlphaEarth Foundations
embeddings (Earth Engine dataset `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`).

Pin, draw, or upload a plot boundary and get risk flags, land history, and a PDF in seconds.

## Setup

1. `pip install -r requirements.txt`
2. Google Cloud project with the Earth Engine API enabled (free for non commercial use,
   check Earth Engine commercial terms before charging customers).
3. `earthengine authenticate`
4. `export EE_PROJECT=landscan-513009`
5. `uvicorn app:app --reload` then open http://localhost:8000

Without steps 3 and 4 the app runs on fake DEMO data (clearly bannered) so you can test the UI.

## Free Cloud Deployment (Render / Docker / Railway)

This repo includes `render.yaml`, `Dockerfile`, and `Procfile` for one-click deployment:

1. **Push this repo to GitHub** (`git add . && git commit -m "Deploy LandScan v2" && git push -f origin main`).
2. **Deploy on Render (Free)**:
   - Open the Render Dashboard → **New +** → **Blueprint** (auto-reads `render.yaml`) or **Web Service** (`uvicorn app:app --host 0.0.0.0 --port $PORT`).
   - `EE_PROJECT` is pre-configured to `landscan-513009`. Paste the contents of your downloaded Google Cloud Service Account JSON key into `EE_SERVICE_ACCOUNT_JSON` for live AlphaEarth satellite data (or leave blank for Demo Mode).

## What is new in v2

- **Plot geometry:** click a pin (radius slider), draw a polygon or rectangle, or upload KML / GeoJSON.
- **Water rings:** surface water on plot, within 30 m (buffer zone), within 300 m (FTL proximity).
- **Flood:** HAND (height above nearest drainage, MERIT Hydro) and Sentinel-1 radar monsoon vs dry season wetness.
- **Rules:** slope and small plot (under 25 pixels) warnings now active.
- **Model:** features use all 8 years (state, change, mid decade, peak jump, variance, land cover),
  each block normalised separately. Similarity weighted kNN.
- **Honest metrics:** leave one out cross validation at `/api/model-metrics`, shown in the Teach tab.
  Once 12+ spots are labeled and accuracy is 60%+, the flag threshold is calibrated from your data.
- **Speed:** all Earth Engine values come back in one round trip, cached for 1 hour. PDF download reuses it.
- **Storage:** labels in SQLite (`landscan.db`). Old `labels.json` is imported once, then press Reindex labels.
- **UI:** satellite / streets basemaps, place search (OpenStreetMap Nominatim, or paste `lat, lon`),
  edit and delete labeled spots.
- **PDF:** 2017 vs 2024 Sentinel-2 thumbnails with the plot outlined, evidence table.

## Files

- `ee_data.py` plot spec, single call EE fetch, thumbnails, demo data
- `ee_batch.py` labeled spot fetch and the 278 dim feature recipe
- `model.py` SQLite labels, kNN, cross validation, calibration
- `logic.py` flags and risk score
- `pdf.py` report, `app.py` API, `static/index.html` map UI

## API

`POST /api/report` (lat, lon, radius or polygon `[[lon, lat], ...]`), `GET /api/report.pdf?rid=`,
`GET/POST /api/label`, `PUT/DELETE /api/label/{id}`, `GET /api/model-metrics`, `POST /api/reindex`.

## Verified vs not verified

Tested in a sandbox without Earth Engine access: API, validation, caching, SQLite, cross validation,
PDF, and the browser UI (pin, draw inputs, KML and GeoJSON upload, search, label edit and delete).
**Not tested against live Earth Engine:** the dataset and band names for Sentinel-1, MERIT Hydro
(`MERIT/Hydro/v1_0_1`, band `hnd`), Sentinel-2 thumbnails, and the real AlphaEarth fetch.
Radar and HAND are wrapped so a failure there only adds a "Data note" instead of breaking the report.
Run a few plots you know first.

## Limits (say this to customers)

- Screening signal, not legal advice. Does not replace title search, survey, or site visit.
- 10 m pixels, annual data. Small plots are noisy. No real time changes.
- Water history is not a legal FTL or buffer boundary. Rules differ by authority.
- Lookalike scores mean something only after you label enough spots (aim 30+, check the leave one out score).
- Nominatim is for light use. Use a paid geocoder before you sell it.
