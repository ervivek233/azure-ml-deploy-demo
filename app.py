from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")

# Define input schema
class SensorInput(BaseModel):
    temperature: float
    vibration: float
    pressure: float
    rotational_speed: int
    operating_hours: int

@app.post("/predict")
def predict(data: SensorInput):
    features = np.array([[
        data.temperature,
        data.vibration,
        data.pressure,
        data.rotational_speed,
        data.operating_hours
    ]])

    prediction = model.predict(features)[0]

    return {
        "failure_prediction": int(prediction)
    }
