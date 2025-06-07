import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = "model.sav"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Input model
class InputData(BaseModel):
    pregnancies: int
    glucose: int
    bloodpressure: int
    skinthickness: int
    insulin: int
    bmi: float
    diabetespedigreefunction: float
    age: int

# Root
@app.get("/")
def read_root():
    return {"message": "Diabetes prediction API is running!"}

# Predict route
@app.post("/predict")
def predict(data: InputData):
    try:
        features = [[
            data.pregnancies,
            data.glucose,
            data.bloodpressure,
            data.skinthickness,
            data.insulin,
            data.bmi,
            data.diabetespedigreefunction,
            data.age
        ]]
        prediction = model.predict(features)
        result = int(prediction[0])
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
