# Watershed

This repository contains a simple full-stack demo with a Python API backend and a React frontend built with Vite.

## Backend (FastAPI)

```
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The API exposes a single endpoint at `/` which returns a hello message.

## Frontend (React + Vite)

```
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173` by default and is configured to interact with the API.


## Elevation Visualization

To visualize elevation tiles (`.hgt` or `.hgt.gz`) located in `backend/data`, install the backend dependencies and run the visualization script:

```
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python visualize_hgt.py
```

The script loads all `.hgt` tiles from the `data` folder and opens an interactive 3D surface plot using Plotly. Use the mouse to rotate along the x, y, and z axes.

Pass `--exaggeration` to control vertical exaggeration of the terrain when launching the script. Values less than `1.0` flatten the plot (default is `0.02`). The generated HTML now includes a slider so you can tweak the exaggeration interactively after the plot opens.

Use `--trees N` to overlay `N` exaggerated tree markers on the interactive plot. The trees are randomly distributed within the displayed area and scale with the vertical exaggeration level.

Files compressed with `gzip` (`.hgt.gz`) are supported and will be decompressed automatically.

## Infographic Generation

To create a simple infographic of the HVC area with illustrative trees, run:

```bash
cd backend
python create_infographic.py
```

The output image `hvc.jpg` will be written to `backend/results`.
