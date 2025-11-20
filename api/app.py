# app.py
import os
import math
import json
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
from starlette.middleware.sessions import SessionMiddleware
from zoneinfo import ZoneInfo  # Python 3.9+
LOCAL_TZ = ZoneInfo("Africa/Lagos")


# Optional libs (installed via requirements)
from PIL import Image
import numpy as np

# ---------- Paths and setup ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
LATEST_RISK_PATH = os.path.join(DATA_DIR, "latest_risk.parquet")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="CHSG Multimodal Inference")
app.add_middleware(SessionMiddleware, secret_key="change-this-secret")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ---------- Utilities ----------

def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        # handle strings like "6.49" or " 6.49 "
        return float(str(value).strip())
    except Exception:
        return default

def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return int(value)
        return int(str(value).strip())
    except Exception:
        return default

def safe_str(value: Any, default: str = "Unknown") -> str:
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default

def compute_risk_score(lat: float, lon: float, unsafe_flag: int = 0, category_weight: float = 0.2) -> float:
    # Simple heuristic risk score combining location and unsafe water flag
    base = (abs(lat) + abs(lon)) / 200.0  # normalize lat/lon magnitude
    unsafe = 0.6 if unsafe_flag == 1 else 0.1
    score = min(1.0, base + unsafe + category_weight)
    return round(score, 3)

def write_latest_risk(record: Dict[str, Any]) -> None:
    df_new = pd.DataFrame([record])
    if os.path.exists(LATEST_RISK_PATH):
        try:
            df = pd.read_parquet(LATEST_RISK_PATH)
            df = pd.concat([df, df_new], ignore_index=True)
        except Exception:
            # If parquet is corrupt or missing deps, start fresh
            df = df_new
    else:
        df = df_new
    df.to_parquet(LATEST_RISK_PATH, index=False)

from typing import List, Dict, Any

def read_latest_risk(limit: int = 200) -> List[Dict[str, Any]]:
    if not os.path.exists(LATEST_RISK_PATH):
        return []
    try:
        df = pd.read_parquet(LATEST_RISK_PATH)
        if df.empty:
            return []

        # Clean invalid values before converting
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Sort by date descending
        df = df.sort_values(by="date", ascending=False)

        records = df.head(limit).to_dict(orient="records")

        # Ensure JSON-safe floats
        safe_records = []
        for r in records:
            safe = {}
            for k, v in r.items():
                if isinstance(v, float):
                    if math.isnan(v) or math.isinf(v):
                        safe[k] = 0.0
                    else:
                        safe[k] = float(v)
                else:
                    safe[k] = v
            safe_records.append(safe)

        return safe_records

    except Exception as e:
        print(f"Error reading latest risk: {e}")
        return []


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None

# ---------- Pydantic models ----------

class TextIngest(BaseModel):
    text: str = Field(..., description="Free text report")
    lat: Optional[float] = None
    lon: Optional[float] = None

    @validator("lat", "lon", pre=True, always=True)
    def coerce_float(cls, v):
        return to_float(v, default=0.0)

class CombinedPredict(BaseModel):
    # Use exact keys the frontend expects; map internally
    Household_Water_Source: Optional[str] = Field(None, alias="Household Water Source")
    Location_of_households_Latitude: Optional[float] = Field(None, alias="Location of households:Latitude")
    Location_of_households_Longitude: Optional[float] = Field(None, alias="Location of households:Longitude")
    UnsafeWater: Optional[int] = 0

    class Config:
        populate_by_name = True  # allow alias population

    @validator("Location_of_households_Latitude", "Location_of_households_Longitude", pre=True, always=True)
    def coerce_coords(cls, v):
        return to_float(v, default=0.0)

    @validator("UnsafeWater", pre=True, always=True)
    def coerce_unsafe(cls, v):
        return to_int(v, default=0)


# ---------- Pages ----------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {
        "request": request,
        "breadcrumb": "Home",
        "next_page": {"url": "/predict/combined", "label": "Make Prediction"}
    })

@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse("about.html", {
        "request": request,
        "breadcrumb": "About",
        "next_page": {"url": "/contact", "label": "Contact Us"}
    })

@app.get("/contact", response_class=HTMLResponse)
def contact(request: Request):
    return templates.TemplateResponse("contact.html", {
        "request": request,
        "breadcrumb": "Contact",
        "next_page": {"url": "/", "label": "Back to Home"}
    })

@app.get("/recommendations", response_class=HTMLResponse)
def recommendations(request: Request):
    records = read_latest_risk(limit=1000)
    df = pd.DataFrame(records)

    state_summary = []
    if not df.empty and "state" in df.columns:
        grouped = df.groupby("state").agg(
            avg_risk=("risk_score", "mean"),
            reports=("state", "count")
        ).reset_index()

        for _, row in grouped.iterrows():
            recs = []
            if row["avg_risk"] > 0.7:
                recs.append("Deploy rapid response teams")
            if row["avg_risk"] > 0.5:
                recs.append("Increase chlorine supply")
            if row["reports"] > 10:
                recs.append("Conduct community awareness campaigns")

            state_summary.append({
                "state": row["state"],
                "avg_risk": round(row["avg_risk"], 2),
                "reports": row["reports"],
                "recommendations": recs or ["Monitor closely"]
            })

    return templates.TemplateResponse(
        "recommendations.html",
        {"request": request, "state_summary": state_summary}
    )
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    records = read_latest_risk(limit=500)
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "breadcrumb": "Dashboard",
        "records": records,
        "next_page": {"url": "/about", "label": "About Us"}
    })

# --- Privacy Policy Page ---
@app.get("/privacy", response_class=HTMLResponse)
def privacy(request: Request):
    return templates.TemplateResponse(
        "privacy.html",
        {
            "request": request,
            "breadcrumb": "Privacy Policy",
            "next_page": {"url": "/terms", "label": "Terms of Service"}
        }
    )

# --- Terms of Service Page ---
@app.get("/terms", response_class=HTMLResponse)
def terms(request: Request):
    return templates.TemplateResponse(
        "terms.html",
        {
            "request": request,
            "breadcrumb": "Terms of Service",
            "next_page": {"url": "/contact", "label": "Contact Us"}
        }
    )

# --- Prediction pages (GET) ---

@app.get("/predict/combined", response_class=HTMLResponse)
def predict_combined_page(request: Request):
    return templates.TemplateResponse("predict_combined.html", {
        "request": request,
        "breadcrumb": "Local Prediction (Combined)",
        "next_page": {"url": "/dashboard", "label": "View Dashboard"}
    })

@app.get("/predict/ndhs", response_class=HTMLResponse)
def predict_ndhs_page(request: Request):
    return templates.TemplateResponse("predict_ndhs.html", {
        "request": request,
        "breadcrumb": "National Prediction (NDHS)",
        "next_page": {"url": "/dashboard", "label": "View Dashboard"}
    })
# --- Prediction pages (POST) ---

@app.post("/predict/combined/run")
def run_combined(
    Household_Water_Source: str = Form(...),
    Location_of_households_Latitude: float = Form(0.0),
    Location_of_households_Longitude: float = Form(0.0),
    UnsafeWater: int = Form(0),
):
    lat = to_float(Location_of_households_Latitude)
    lon = to_float(Location_of_households_Longitude)
    unsafe = to_int(UnsafeWater)

    risk_score = compute_risk_score(lat, lon, unsafe_flag=unsafe, category_weight=0.25)
    record = {
        "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "lat": lat,
        "lon": lon,
        "category": "combined_local",
        "is_risky": risk_score >= 0.5,
        "risk_score": risk_score,
        "source": "combined",
        "features": {
            "Household Water Source": Household_Water_Source,
            "UnsafeWater": unsafe
        }
    }
    write_latest_risk(record)
    predictions = {
        "rf_prediction": 1 if is_risky else 0,
        "xgb_prediction": 1 if (risk_score > 0.65) else 0,
    }
    # ✅ Decide response mode
    if mode == "json":
        return JSONResponse({"message": "Prediction complete", "record": record, "predictions": predictions})
    elif mode == "redirect":
        return RedirectResponse(url="/dashboard?msg=Prediction complete! Go to Dashboard to view.", status_code=303)
    else:  # default: HTML with pop‑up
        return templates.TemplateResponse(
            "predict_combined.html",
            {"request": request, "risk_score": risk_score, "is_risky": is_risky}
        )
# --- Ingestion pages (GET) ---

@app.get("/ingest/text", response_class=HTMLResponse)
def ingest_text_page(request: Request):
    return templates.TemplateResponse("ingest_text.html", {
        "request": request,
        "breadcrumb": "Text Reports",
        "next_page": {"url": "/dashboard", "label": "Dashboard"}
    })

@app.get("/ingest/voice", response_class=HTMLResponse)
def ingest_voice_page(request: Request):
    ffmpeg_ok = ffmpeg_available()
    return templates.TemplateResponse("ingest_voice.html", {
        "request": request,
        "breadcrumb": "Voice Reports",
        "next_page": {"url": "/dashboard", "label": "Dashboard"},
        "ffmpeg_available": ffmpeg_ok
    })

@app.get("/ingest/image", response_class=HTMLResponse)
def ingest_image_page(request: Request):
    return templates.TemplateResponse("ingest_image.html", {
        "request": request,
        "breadcrumb": "Image Reports",
        "next_page": {"url": "/dashboard", "label": "Dashboard"}
    })

# ---------- Health ----------

@app.get("/ping")
def ping():
    return {"status": "ok"}

from fastapi import Form
from fastapi.responses import JSONResponse

# ---------- Ingestion: Text (AJAX JSON) ----------
@app.post("/ingest/text")
def ingest_text(request: Request, payload: TextIngest, mode: str = "json"):
    try:
        text = safe_str(payload.text)
        lat = to_float(payload.lat)
        lon = to_float(payload.lon)

        lower = text.lower()
        if any(k in lower for k in ["cholera", "diarrhea", "diarrhoea"]):
            category, cat_weight = "waterborne_cholera", 0.4
        elif "typhoid" in lower:
            category, cat_weight = "waterborne_typhoid", 0.35
        elif any(k in lower for k in ["flood", "river overflow", "contamination"]):
            category, cat_weight = "environmental_risk", 0.25
        else:
            category, cat_weight = "general_report", 0.1

        risk_score = compute_risk_score(
            lat, lon,
            unsafe_flag=1 if "unsafe" in lower else 0,
            category_weight=cat_weight
        )
        is_risky = risk_score >= 0.5

        record = {
            "date": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "lat": lat,
            "lon": lon,
            "category": category,
            "is_risky": is_risky,
            "risk_score": risk_score,
            "source": "text",
            "text": text,
        }
        write_latest_risk(record)

        if mode == "html":
            return templates.TemplateResponse(
                "ingest_text.html",
                {"request": request, "risk_score": risk_score, "is_risky": is_risky}
            )
        else:
            return {"prediction": "Risky" if is_risky else "Safe", "record": record}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text prediction error: {e}")

# ---------- Ingestion: Voice (Transcript JSON) ----------
from fastapi import Body
@app.post("/ingest/voice")
async def ingest_voice(payload: dict = Body(...)):
    """
    Accepts a JSON payload with transcript text and optional lat/lon.
    Example:
    {
      "text": "Cholera outbreak near river",
      "lat": 9.0820,
      "lon": 8.6753,
      "source": "voice"
    }
    """
    text = safe_str(payload.get("text", ""))
    lat = to_float(payload.get("lat", 0.0))
    lon = to_float(payload.get("lon", 0.0))
    source = payload.get("source", "voice")

    # Simple heuristic: detect Nigerian languages keywords
    lower = text.lower()
    if any(k in lower for k in ["cholera", "diarrhea", "diarrhoea"]):
        category, cat_weight = "waterborne_cholera", 0.4
    elif "typhoid" in lower:
        category, cat_weight = "waterborne_typhoid", 0.35
    elif any(k in lower for k in ["flood", "river overflow", "contamination"]):
        category, cat_weight = "environmental_risk", 0.25
    else:
        category, cat_weight = "voice_report", 0.2

    # Risk score heuristic
    risk_score = compute_risk_score(lat, lon, unsafe_flag=1 if "unsafe" in lower else 0, category_weight=cat_weight)
    is_risky = risk_score >= 0.5

    record = {
        "date": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "lat": lat,
        "lon": lon,
        "category": category,
        "is_risky": is_risky,
        "risk_score": risk_score,
        "source": source,
        "text": text,
    }

    try:
        write_latest_risk(record)
        if mode == "html":
            return templates.TemplateResponse(
                "ingest_voice.html",
                {"request": request, "risk_score": risk_score, "is_risky": is_risky}
            )
        else:
            return {"prediction": "Risky" if is_risky else "Safe", "record": record}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice prediction error: {e}")

# ---------- Ingestion: Image (AJAX FormData
@app.post("/ingest/image")
async def ingest_image(
    request: Request,  
    file: UploadFile = File(...),
    lat: float = Form(0.0),
    lon: float = Form(0.0),
    mode: str = Form("json")
):
    tmp_path = os.path.join(DATADIR, f"upload{datetime.now(LOCAL_TZ).timestamp()}_{file.filename}")
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        img = Image.open(tmp_path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        brightness = float(arr.mean()) / 255.0

        category = "image_environment"
        cat_weight = 0.25 if brightness > 0.5 else 0.15
        lat_f, lon_f = to_float(lat), to_float(lon)   
        risk_score = compute_risk_score(lat_f, lon_f, unsafe_flag=0, category_weight=cat_weight) 
        is_risky = risk_score >= 0.5

        record = {
            "date": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "lat": lat_f,
            "lon": lon_f,
            "category": category,
            "is_risky": is_risky,
            "risk_score": risk_score,
            "source": "image",
            "meta": {"brightness": round(brightness, 3)},
        }
        write_latest_risk(record)

        if mode == "html":
            return templates.TemplateResponse(
                "ingest_image.html",
                {"request": request, "risk_score": risk_score, "is_risky": is_risky}
            )
        else:
            return {"prediction": "Risky" if is_risky else "Safe", "record": record}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image prediction error: {e}")

# ---------- Predictions: Combined (AJAX FormData) ----------

@app.post("/predict/combined/run")
async def run_combined(
    Household_Water_Source: str = Form(...),
    Location_of_households_Latitude: float = Form(0.0),
    Location_of_households_Longitude: float = Form(0.0),
    UnsafeWater: int = Form(0),
    mode: str = Form("json")
):
    lat = to_float(Location_of_households_Latitude)
    lon = to_float(Location_of_households_Longitude)
    unsafe = to_int(UnsafeWater)


    source_weight_map = {
        "Borehole": 0.15, "Tap": 0.1, "River": 0.35, "Well": 0.25, "Unknown": 0.2,
    }
    cat_weight = source_weight_map.get(Household_Water_Source or "Unknown", 0.2)

    risk_score = compute_risk_score(lat, lon, unsafe_flag=unsafe, category_weight=cat_weight)
    is_risky = risk_score >= 0.5

    record = {
        "date": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "lat": lat,
        "lon": lon,
        "category": f"combined_{Household_Water_Source or 'Unknown'}",
        "is_risky": is_risky,
        "risk_score": risk_score,
        "source": "combined",
        "features": {
            "Household Water Source": Household_Water_Source,
            "UnsafeWater": unsafe,
        },
    }
    write_latest_risk(record)
    

    predictions = {
        "rf_prediction": 1 if is_risky else 0,
        "xgb_prediction": 1 if (risk_score > 0.65) else 0,
    }
    # Decide response mode
    if mode == "json":
        return JSONResponse({"message": "Prediction complete", "record": record, "predictions": predictions})
    elif mode == "redirect":
        return RedirectResponse(url="/dashboard?msg=Prediction complete! Go to Dashboard to view.", status_code=303)
    else:  # default: HTML with pop‑up
        return templates.TemplateResponse(
            "predict_combined.html",
            {"request": request, "risk_score": risk_score, "is_risky": is_risky}
        )
# ---------- Predictions: NDHS (AJAX FormData) ----------
@app.post("/predict/ndhs")
async def run_ndhs(request: Request):
    try:
        data = await request.json()
        state = data.get("state")
        indicator = data.get("indicator")
        value = float(data.get("value"))

        if not state or not indicator:
            raise HTTPException(status_code=400, detail="State and indicator are required")

        # Simple heuristic for demo
        cat_weight = 0.3 if "unsafe" in indicator.lower() else 0.15
        risk_score = compute_risk_score(lat=0.0, lon=0.0, unsafe_flag=1, categoryweight=cat_weight)
        is_risky = risks_core >= 0.5

        record = {
            "date": datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "state": state,
            "indicator": indicator,
            "value": value,
            "category": "ndhs_prediction",
            "is_risky": is_risky,
            "risk_score": risk_score,
            "source": "ndhs",
        }

        writelatestrisk(record)
        if mode == "html":
            return templates.TemplateResponse(
                "predict_ndhs.html",
                {"request": request, "risk_score": risk_score, "is_risky": is_risky}
            )
        else:  # default JSON for AJAX
            return {"prediction": "Risky" if is_risky else "Safe", "record": record}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NDHS prediction error: {e}")


@app.get("/risk/latest")
def risk_latest():
    return read_latest_risk(limit=200)

# Optional: return all for debugging
@app.get("/risk/all")
def risk_all():
    return read_latest_risk(limit=1000)
# ---------- Error handlers ----------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Avoid exposing stack traces; provide friendly message
    return JSONResponse(status_code=500, content={"detail": f"Server error: {str(exc)}"})

