import os
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List

app = FastAPI()

# Allow CORS for development convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory to serve images
STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

LABELS_DIR = "labels"
if not os.path.exists(LABELS_DIR):
    os.makedirs(LABELS_DIR)

class LabelRequest(BaseModel):
    idx: int
    label: str  # "good", "bad", etc.

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    return FileResponse('index.html')

@app.get("/api/datasets")
def list_datasets():
    if not os.path.exists(STATIC_DIR):
        return []
    datasets = [d for d in os.listdir(STATIC_DIR) if os.path.isdir(os.path.join(STATIC_DIR, d))]
    return datasets

@app.get("/api/{dataset_name}/status")
def get_dataset_status(dataset_name: str):
    dataset_path = os.path.join(STATIC_DIR, dataset_name, "input")
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get all available indices from files
    # Assuming files are named {idx}.png
    files = os.listdir(dataset_path)
    indices = []
    for f in files:
        if f.endswith(".png"):
            try:
                idx = int(f.replace(".png", ""))
                indices.append(idx)
            except ValueError:
                pass
    
    indices.sort()
    
    # Load existing labels
    labels_file = os.path.join(LABELS_DIR, f"labels_{dataset_name}.json")
    labels = {}
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            try:
                labels = json.load(f)
            except json.JSONDecodeError:
                pass

    # Default missing labels to "bad"
    labels_changed = False
    for idx_int in indices:
        idx_str = str(idx_int)
        if idx_str not in labels:
            labels[idx_str] = "bad"
            labels_changed = True

    if labels_changed:
        with open(labels_file, "w") as f:
            json.dump(labels, f, indent=2)

    return {
        "dataset": dataset_name,
        "total": len(indices),
        "indices": indices,
        "labels": labels
    }

@app.post("/api/{dataset_name}/label")
def save_label(dataset_name: str, req: LabelRequest):
    labels_file = os.path.join(LABELS_DIR, f"labels_{dataset_name}.json")
    
    labels = {}
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            try:
                labels = json.load(f)
            except json.JSONDecodeError:
                pass
    
    # Update label
    labels[str(req.idx)] = req.label
    
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)
        
    return {"status": "success", "idx": req.idx, "label": req.label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
