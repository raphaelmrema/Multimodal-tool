# Multimodal-tool
# Integrated Multimodal Transit (FastAPI)

FastAPI service that combines OSRM bike routing, Google Transit Directions, and (optionally) shapefile-based segment analysis.

## Run locally
```bash
python -m venv .venv
. .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
export GEOPANDAS_IO_ENGINE=pyogrio
export GOOGLE_API_KEY=YOUR_KEY
uvicorn integrated_multimodal_transit:app --reload
