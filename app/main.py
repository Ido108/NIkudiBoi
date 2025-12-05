import os
import sys
import logging
import tempfile
import json
import shutil
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoConfig

# Add source path
if os.path.exists("src"):
    sys.path.append(os.path.abspath("."))
else:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import DNikudModel, ModelConfig
from src.models_utils import predict
from src.utiles_data import NikudDataset, Nikud, extract_text_to_compare_nakdimon
from src.running_params import BATCH_SIZE, MAX_LENGTH_SEN

# --- Constants ---
MODELS_DIR = "models"
CONFIG_FILE = os.path.join(MODELS_DIR, "site_config.json")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nikudiboi")

# --- Default Config ---
DEFAULT_CONFIG = {
    "site_title": "Nikudiboi",
    "subtitle": "AI Powered Hebrew Diacritization",
    "primary_color": "#4f46e5",
    "welcome_message": "הדבק טקסט בעברית כאן...",
    "active_model": "model.pth"
}

# --- Global State ---
ml_models = {}
app_config = DEFAULT_CONFIG.copy()

def load_config():
    global app_config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                app_config.update(loaded)
        except Exception as e:
            logger.error(f"Error loading config: {e}")

def save_config():
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(app_config, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving config: {e}")

def reload_model():
    """Reloads the model based on the current active_model in app_config"""
    model_name = app_config.get("active_model")
    model_path = os.path.join(MODELS_DIR, model_name)
    
    logger.info(f"Reloading model: {model_path}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False

    try:
        # 1. Tokenizer (Cached)
        if "tokenizer" not in ml_models:
            ml_models["tokenizer"] = AutoTokenizer.from_pretrained("tau/tavbert-he")

        # 2. Config & Architecture
        # We assume standard TavBERT config for now
        base_model_name = "tau/tavbert-he"
        config = AutoConfig.from_pretrained(base_model_name)
        
        model = DNikudModel(config, 
                            len(Nikud.label_2_id["nikud"]), 
                            len(Nikud.label_2_id["dagesh"]), 
                            len(Nikud.label_2_id["sin"]), 
                            device=DEVICE).to(DEVICE)
        
        # 3. Load Weights
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        
        ml_models["model"] = model
        logger.info("Model reloaded successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    
    # 1. Ensure models dir exists (Critical for Volume mounting)
    if not os.path.exists(MODELS_DIR):
        try:
            os.makedirs(MODELS_DIR)
            logger.info(f"Created models directory at: {MODELS_DIR}")
        except Exception as e:
            logger.error(f"Failed to create models directory: {e}")

    # 2. Load Config (Now safe to do)
    load_config()
    
    # 3. Load Model
    reload_model()
    
    yield
    
    # Shutdown
    ml_models.clear()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# --- Schemas ---
class PredictRequest(BaseModel):
    text: str
    compare_nakdimon: bool = False

class SiteConfig(BaseModel):
    site_title: str
    subtitle: str
    primary_color: str
    welcome_message: str
    active_model: str

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": app_config
    })

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    # List models
    models = []
    if os.path.exists(MODELS_DIR):
        models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "config": app_config,
        "models": models
    })

@app.post("/api/config")
async def update_config(config: SiteConfig):
    global app_config
    old_model = app_config.get("active_model")
    app_config.update(config.dict())
    save_config()
    
    # Reload model if changed
    if config.active_model != old_model:
        success = reload_model()
        if not success:
            return JSONResponse(status_code=400, content={"message": "Config saved, but model failed to load."})
            
    return {"status": "success", "config": app_config}

@app.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith('.pth'):
        raise HTTPException(400, "Only .pth files are allowed")
        
    file_path = os.path.join(MODELS_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")
        
    return {"filename": file.filename, "status": "uploaded"}

@app.delete("/api/models/{filename}")
async def delete_model(filename: str):
    file_path = os.path.join(MODELS_DIR, filename)
    
    if filename == app_config.get("active_model"):
        raise HTTPException(400, "Cannot delete the active model")
        
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "deleted"}
    raise HTTPException(404, "File not found")

@app.post("/api/predict")
async def predict_text(request: PredictRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please configure a valid model in Admin.")

    try:
        with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False, suffix='.txt') as temp_file:
            temp_file.write(request.text)
            temp_file_path = temp_file.name

        try:
            dataset = NikudDataset(ml_models["tokenizer"], file=temp_file_path, logger=logger, max_length=MAX_LENGTH_SEN)
            dataset.prepare_data(name="prediction")
            dl = torch.utils.data.DataLoader(dataset.prepered_data, batch_size=BATCH_SIZE)
            all_labels = predict(ml_models["model"], dl, DEVICE)
            result_text = dataset.back_2_text(labels=all_labels)
            
            if request.compare_nakdimon:
                result_text = extract_text_to_compare_nakdimon(result_text)

            return {"result": result_text}
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))