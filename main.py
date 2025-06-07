import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# Enable CORS (adjust origin in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace with your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download and load model
MODEL_URL = "https://kitish-whatsapp-bot-media.s3.ap-south-1.amazonaws.com/documentMessage_1749283919637.bin"
MODEL_PATH = "diabetes_model.sav"

if not os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "wb") as f:
        f.write(requests.get(MODEL_URL).content)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Define input schema
class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    bloodpressure: float
    skin: float
    insulin: float
    bmi: float
    diabetespedigree: float
    age: int

@app.post("/predict")
def predict(data: DiabetesInput):
    try:
        features = [[
            data.pregnancies,
            data.glucose,
            data.bloodpressure,
            data.skin,
            data.insulin,
            data.bmi,
            data.diabetespedigree,
            data.age
        ]]
        prediction = model.predict(features)
        return {"result": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
