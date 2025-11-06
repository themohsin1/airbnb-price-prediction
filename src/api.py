from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title="Airbnb Price Prediction API")

pipe = joblib.load("models/airbnb_pipeline.pkl")

@app.get("/")
def home():
    return {"message": "Welcome to the Airbnb Price Prediction API 🏠"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = pipe.predict(df)
    return {"predicted_price": round(float(pred[0]), 2)}
