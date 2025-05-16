import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
import requests
import joblib

from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Load env vars
load_dotenv()
FITBIT_CLIENT_ID = os.getenv("FITBIT_CLIENT_ID")
FITBIT_CLIENT_SECRET = os.getenv("FITBIT_CLIENT_SECRET")
FITBIT_BASIC_AUTH = os.getenv("FITBIT_BASIC_AUTH")  # Base64 encoded client_id:client_secret
REDIRECT_URI = "http://localhost:8000/fitbit/callback"

scaler = joblib.load("models/heart_scaler.pkl")  # Save your original StandardScaler with joblib

app = FastAPI()

# Static and template config
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load models
image_model = tf.keras.models.load_model("models/MULTI_DISEASE_DETECTION_MODEL.keras")
heart_model = tf.keras.models.load_model("models/heart_disease_detection_model.keras")

image_class_map = [
    "COVID", "NORMAL", "PNEUMONIA",
    "basal cell carcinoma", "glioma tumor",
    "melanoma", "no tumor", "pituitary tumor"
]

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ===========================
# ✅ HEART DISEASE PREDICTION
# ===========================
@app.post("/predict-heart")
async def predict_heart(
    age: float = Form(...),
    sex: int = Form(...),  # 1 = Male, 0 = Female
    trestbps: float = Form(...),
    chol: float = Form(...),
    fbs: int = Form(...),  # 1 = FastingBS > 120 mg/dl, else 0
    thalach: float = Form(...),
    exang: int = Form(...),  # 1 = Yes, 0 = No
    oldpeak: float = Form(...),
    cp: str = Form(...),  # e.g., "ATA", "NAP", "TA", "ASY"
    restecg: str = Form(...),  # "Normal", "ST"
    slope: str = Form(...),  # "Flat", "Up"
):
    # Fill missing values using your functions
    trestbps = trestbps if trestbps else autofill_trestbps(age, sex)
    chol = chol if chol else autofill_chol(age, sex, trestbps)
    oldpeak = oldpeak if oldpeak else autofill_oldpeak(age, trestbps, chol)
    thalach = thalach if thalach else autofill_thalach(age, sex, chol)

    # Feature scaling
    scaled_values = scaler.transform([[trestbps, chol, thalach, oldpeak]])[0]
    scaled_trestbps, scaled_chol, scaled_thalach, scaled_oldpeak = scaled_values
    scaled_age = age / 100

    # Manual one-hot encoding
    cp_encoded = {
        "ChestPainType_ATA": 1 if cp == "ATA" else 0,
        "ChestPainType_NAP": 1 if cp == "NAP" else 0,
        "ChestPainType_TA": 1 if cp == "TA" else 0
    }
    restecg_encoded = {
        "RestingECG_Normal": 1 if restecg == "Normal" else 0,
        "RestingECG_ST": 1 if restecg == "ST" else 0
    }
    slope_encoded = {
        "ST_Slope_Flat": 1 if slope == "Flat" else 0,
        "ST_Slope_Up": 1 if slope == "Up" else 0
    }

    # Construct input vector (ensure correct feature order)
    input_vector = np.array([[
        scaled_age, sex, scaled_trestbps, scaled_chol,
        fbs, scaled_thalach, exang, scaled_oldpeak,
        cp_encoded["ChestPainType_ATA"],
        cp_encoded["ChestPainType_NAP"],
        cp_encoded["ChestPainType_TA"],
        restecg_encoded["RestingECG_Normal"],
        restecg_encoded["RestingECG_ST"],
        slope_encoded["ST_Slope_Flat"],
        slope_encoded["ST_Slope_Up"]
    ]])

    # Predict
    prediction = heart_model.predict(input_vector)
    predicted_class = int((prediction > 0.5).astype("int")[0][0])
    return {"heart_disease_prediction": predicted_class}


# ===========================
# ✅ IMAGE DISEASE PREDICTION
# ===========================
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((224, 224)).convert("RGB")
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = image_model.predict(image_array)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_class = image_class_map[predicted_index]
    confidence = float(np.max(prediction))

    return {
        "class_name": predicted_class,
        "confidence_score": confidence
    }

# =============================
# ✅ FITBIT OAUTH INTEGRATION
# =============================

@app.get("/fitbit/connect")
def connect_fitbit():
    scope = "heartrate activity profile"
    return RedirectResponse(
        url=(f"https://www.fitbit.com/oauth2/authorize?response_type=code"
             f"&client_id={FITBIT_CLIENT_ID}"
             f"&redirect_uri={REDIRECT_URI}"
             f"&scope={scope}")
    )

@app.get("/fitbit/callback")
def fitbit_callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        return {"error": "No authorization code received"}

    token_url = "https://api.fitbit.com/oauth2/token"
    headers = {
        "Authorization": f"Basic {FITBIT_BASIC_AUTH}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "client_id": FITBIT_CLIENT_ID,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
        "code": code
    }

    response = requests.post(token_url, headers=headers, data=data)
    token_data = response.json()

    if "access_token" not in token_data:
        return {"error": token_data}

    return {
        "message": "Successfully authenticated with Fitbit",
        "access_token": token_data["access_token"]
    }

# =============================
# ✅ FETCH HEART RATE FROM FITBIT
# =============================

@app.get("/fitbit/heart-rate")
def fetch_heart_data(token: str):
    url = "https://api.fitbit.com/1/user/-/activities/heart/date/today/1d.json"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    data = response.json()

    try:
        hr_data = data["activities-heart"][0]["value"]
        resting_hr = hr_data.get("restingHeartRate", "Not available")
    except Exception as e:
        return {"error": "Unable to fetch heart rate", "details": str(e)}

    return {
        "resting_heart_rate": resting_hr,
        "raw_data": hr_data
    }

# ===========================
# 🚀 AUTO-FILL LOGIC FUNCTIONS
# ===========================

def autofill_trestbps(age: float, sex: float) -> float:
    """Autofill Resting Blood Pressure based on age and sex."""
    # Example logic (you can modify this based on better data models)
    if sex == 0:  # Male
        return 120 + (age // 10) * 5
    else:  # Female
        return 110 + (age // 10) * 5

def autofill_chol(age: float, sex: float, trestbps: float) -> float:
    """Autofill Cholesterol based on age, sex, and resting blood pressure."""
    if sex == 0:  # Male
        return 220 + (age // 10) * 10 + trestbps // 2
    else:  # Female
        return 200 + (age // 10) * 8 + trestbps // 2

def autofill_oldpeak(age: float, trestbps: float, chol: float) -> float:
    """Autofill Oldpeak based on age, resting blood pressure, and cholesterol."""
    # A rough estimate based on typical medical insights (you can refine this)
    return (age // 10) * 0.5 + (trestbps // 100) * 0.3 + (chol // 50) * 0.2

def autofill_thalach(age: float, sex: float, chol: float) -> float:
    """Autofill Max Heart Rate based on age, sex, and cholesterol."""
    # MaxHR = 220 - age (a standard formula for estimated max heart rate)
    max_hr = 220 - age
    if chol > 250:
        max_hr -= 10  # Cholesterol adjustment for health conditions
    return max_hr
