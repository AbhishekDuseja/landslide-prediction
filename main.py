import numpy as np
import pickle
import json
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Load ML model
# -----------------------------
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Load JSON dataset
# -----------------------------
with open("output.json", "r") as f:
    dataset = json.load(f)

# -----------------------------
# Initialize app
# -----------------------------
app = FastAPI(title="Earthquake API")

# =============================
# ===== PREDICT ENDPOINT ======
# =============================

class InputData(BaseModel):
    Magnitude: float
    Rjb_km: float
    Vs30_m_s: float
    Hypo_Depth_km: float
    Critical_Accel_g: float
    PGA_g: float

def preprocess(data: InputData):
    log_Rjb = np.log1p(data.Rjb_km)
    log_Vs30 = np.log1p(data.Vs30_m_s)
    log_H = np.log1p(data.Hypo_Depth_km)
    log_Ac = np.log1p(data.Critical_Accel_g)
    log_PGA = np.log1p(data.PGA_g)
    M_R = data.Magnitude * data.Rjb_km

    return np.array([[
        data.Magnitude,
        log_Rjb,
        log_Vs30,
        log_H,
        log_Ac,
        log_PGA,
        M_R
    ]])

@app.post("/predict")
def predict(data: InputData):
    X = preprocess(data)
    log_pred = model.predict(X)[0]
    final_pred = np.expm1(log_pred)

    return {
        "Predicted_Max_RotD_Disp_cm": float(final_pred)
    }

# =============================
# ===== GET PARAMS ============
# =============================

class LocationInput(BaseModel):
    latitude: float
    longitude: float

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

@app.post("/getparams")
def get_params(loc: LocationInput):

    lat = loc.latitude
    lon = loc.longitude

    points = []

    # Compute distances
    for item in dataset:
        d = haversine(lat, lon, item["Latitude"], item["Longitude"])
        points.append((d, item))

    # Sort by distance
    points.sort(key=lambda x: x[0])

    # Take 6 nearest (like hexagon idea)
    k = 6
    nearest = points[:k]

    distances = np.array([p[0] for p in nearest])
    distances = np.where(distances == 0, 1e-6, distances)

    weights = 1 / distances
    weights = weights / weights.sum()

    keys = [
        "Magnitude", "Rjb_km", "Vs30_m_s",
        "Hypo_Depth_km", "Critical_Accel_g", "PGA_g"
    ]

    weighted_params = {}

    for key in keys:
        weighted_params[key] = float(sum(
            weights[i] * nearest[i][1]["Averages"][key]
            for i in range(k)
        ))

    # Return RSN + coords used
    rsn_list = " ".join(str(p[1]["RSN"]) for p in nearest)
    lat_list = " ".join(str(p[1]["Latitude"]) for p in nearest)
    lon_list = " ".join(str(p[1]["Longitude"]) for p in nearest)

    return {
        "Interpolated_Params": weighted_params,
        "RSNs_used": rsn_list,
        "Latitudes_used": lat_list,
        "Longitudes_used": lon_list
    }

# =============================
# ===== ROOT ==================
# =============================

@app.get("/")
def home():
    return {"message": "API running"}
