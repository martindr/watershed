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

To visualize `.hgt` files located in `backend/data`, install the backend dependencies and run the visualization script:

```
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python visualize_hgt.py
```

The script loads all `.hgt` tiles from the `data` folder and opens an interactive 3D surface plot using Plotly. Use the mouse to rotate along the x, y, and z axes.
