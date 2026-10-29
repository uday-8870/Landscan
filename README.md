# LandScan

Land due diligence reports for realtors and builders, powered by Google DeepMind's AlphaEarth Foundations
embeddings (Earth Engine dataset `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`).

Drop a pin, get risk flags, land cover history, and a PDF in seconds.

## Setup

1. `pip install -r requirements.txt`
2. Make a Google Cloud project with the Earth Engine API enabled (free for non commercial use;
   check Earth Engine commercial terms before charging customers).
3. `earthengine authenticate`
4. `export EE_PROJECT=your-gcp-project-id`
5. `uvicorn app:app --reload` then open http://localhost:8000

Without step 3 and 4 the app runs on fake DEMO data (clearly bannered) so you can test the UI.

## How it works

- `ee_data.py`: per plot, pulls 8 years (2017 to 2024) of AlphaEarth embeddings, plus Dynamic World land cover,
  JRC surface water, SRTM terrain.
- `ee_batch.py`: batch embedding fetch for labeled spots.
- `model.py`: lookalike model. Compares a plot's embedding trajectory (first year, last year, change)
  to spots you labeled (lake bed, encroachment, unapproved layout, clean, hotspot) using cosine kNN.
- `logic.py`: change detection, water, terrain, and growth flags and the 0 to 100 score.
- `pdf.py`: one page report.

## Teaching the AI

Open the "Teach the AI" tab, click spots you personally know are lake beds, encroachments, unapproved
layouts, clean plots, or boom areas, and save. Aim for 10+ per class. Labels live in `labels.json`.
Only label what you are sure about, since the model is only as good as these.

## Limits (say this to customers)

- Screening signal, not legal advice. Does not replace title search, survey, or site visit.
- 10 m pixels, annual data. Plots under a few hundred sq m are noisy. No real time changes.
- Lookalike scores are only meaningful after you have enough labeled spots.
