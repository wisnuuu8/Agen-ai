from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import joblib
import pandas as pd
import requests
import os

app = FastAPI(title="Fertilizer Recommendation API - Orbitstick & Open-Meteo")

# Memuat model langsung tanpa memerlukan model_columns.pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "random_forest_model.pkl")

model = joblib.load(model_path)

# Mengambil daftar kolom otomatis dari model (atau fallback ke 6 kolom sensor BPSI)
DEFAULT_COLUMNS = [
    'N (mg/kg)', 'P (mg/kg)', 'K (mg/kg)',
    'SOIL_PH (pH)', 'TEMP (°C)', 'RELATIVE_HUMIDITY (%)'
]
model_columns = list(getattr(model, "feature_names_in_", DEFAULT_COLUMNS))


# Fungsi Imputasi Cuaca Otomatis Open-Meteo berdasarkan Koordinat & Waktu Pengujian Alat
def fetch_open_meteo_precipitation(lat: float, lon: float, waktu_alat_str: Optional[str] = None) -> float:
    try:
        target_date = pd.to_datetime(waktu_alat_str).date() if waktu_alat_str else datetime.utcnow().date()
        start_date = target_date - timedelta(days=30)

        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&"
            f"start_date={start_date}&end_date={target_date}&"
            f"daily=precipitation_sum&timezone=auto"
        )

        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        precip_sums = data.get("daily", {}).get("precipitation_sum", [])
        total_precip = sum([p for p in precip_sums if p is not None])
        return round(float(total_precip), 2)
    except Exception as e:
        print(f"Gagal mengambil data Open-Meteo: {e}")
        return 0.0


class FarmInput(BaseModel):
    Temperature: float
    Moisture: float
    Rainfall: Optional[float] = None
    PH: float
    Nitrogen: float
    Phosphorous: float
    Potassium: float
    Carbon: float = 0.0
    Latitude: float = -7.444
    Longitude: float = 109.245
    Waktu_Alat: Optional[str] = None


@app.get("/")
def home():
    return {
        "message": "Fertilizer AI API with Open-Meteo is running successfully!",
        "expected_columns": model_columns
    }


@app.post("/predict")
def predict_fertilizer(data: FarmInput):
    try:
        curah_hujan = (
            data.Rainfall
            if (data.Rainfall is not None and data.Rainfall > 0.0)
            else fetch_open_meteo_precipitation(data.Latitude, data.Longitude, data.Waktu_Alat)
        )

        mapped_data = {
            # 1. Nama kolom sesuai expected_columns pada model .pkl Anda saat ini
            'N': data.Nitrogen,
            'P': data.Phosphorous,
            'K': data.Potassium,
            'SOIL_PH': data.PH,
            'TEMP': data.Temperature,
            'RELATIVE_HUMIDITY': data.Moisture,
            # 2. Nama kolom dengan satuan (jika nanti model .pkl diperbarui dari df_clean terbaru)
            'N (mg/kg)': data.Nitrogen,
            'P (mg/kg)': data.Phosphorous,
            'K (mg/kg)': data.Potassium,
            'SOIL_PH (pH)': data.PH,
            'TEMP (°C)': data.Temperature,
            'RELATIVE_HUMIDITY (%)': data.Moisture,
            # 3. Cadangan kompatibilitas
            'Rainfall': curah_hujan,
        }

        df_input = pd.DataFrame([mapped_data])
        df_encoded = pd.get_dummies(df_input)
        df_final = df_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(df_final)
        confidence = model.predict_proba(df_final).max()

        return {
            "status": "success",
            "waktu_pengujian_alat": data.Waktu_Alat or str(datetime.utcnow().date()),
            "latitude": data.Latitude,
            "longitude": data.Longitude,
            "rainfall_30_days_mm": curah_hujan,
            "recommended_fertilizer": str(prediction[0]),
            "confidence_score": round(float(confidence) * 100, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict-live")
def predict_live_orbitstick(
    token: Optional[str] = Query(default=None, description="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzY29wZSI6Im9yYmlzdGlja19wb3J0YWwiLCJwdiI6IjE2Y2RiNDAzZmY3MWM4NmYiLCJpYXQiOjE3OTMyOTcwNzAsImV4cCI6MTc5MzMyNTg3MH0.NO6LycQwKSV4-txyM0COkc9BYvAY6xi6ef-RnM3ldT0"),
    authorization: Optional[str] = Header(default=None)
):
    try:
        orbitstick_api_url = "https://orbistick.orbitani.com/api/orbistick-portal/readings?limit=1"
        
        # Mengambil token dari URL Query, Header, atau Environment Variable Vercel (ORBITSTICK_TOKEN)
        auth_token = token or os.environ.get("ORBITSTICK_TOKEN")
        
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        
        if authorization:
            headers["Authorization"] = authorization
        elif auth_token:
            # Jika token belum diawali kata 'Bearer ', tambahkan otomatis
            headers["Authorization"] = auth_token if auth_token.startswith("Bearer ") else f"Bearer {auth_token}"

        # Jika portal menggunakan Cookie (opsional, diset via env ORBITSTICK_COOKIE)
        orbit_cookie = os.environ.get("ORBITSTICK_COOKIE")
        if orbit_cookie:
            headers["Cookie"] = orbit_cookie

        api_response = requests.get(orbitstick_api_url, headers=headers, timeout=10)
        api_response.raise_for_status()

        json_data = api_response.json()
        latest = json_data[0] if isinstance(json_data, list) else json_data.get("data", [{}])[0]

        lat = float(latest.get("Latitude") or -7.444)
        lon = float(latest.get("Longitude") or 109.245)
        waktu_alat = latest.get("Waktu Alat (UTC)") or latest.get("Waktu Diterima (WIB)")
        raw_rain = latest.get("Curah Hujan 30 Hari (mm)")

        live_input = FarmInput(
            Nitrogen=float(latest.get("N Kalibrasi (mg/kg)") or latest.get("N Mentah (mg/kg)") or 0.0),
            Phosphorous=float(latest.get("P Kalibrasi (mg/kg)") or latest.get("P Mentah (mg/kg)") or 0.0),
            Potassium=float(latest.get("K Kalibrasi (mg/kg)") or latest.get("K Mentah (mg/kg)") or 0.0),
            PH=float(latest.get("pH") or 6.5),
            Temperature=float(latest.get("Suhu (°C)") or 25.0),
            Moisture=float(latest.get("Kelembapan (%, maks 90)") or 75.0),
            Rainfall=float(raw_rain) if (raw_rain is not None and str(raw_rain).strip() != "") else None,
            Latitude=lat,
            Longitude=lon,
            Waktu_Alat=waktu_alat
        )

        return predict_fertilizer(live_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))