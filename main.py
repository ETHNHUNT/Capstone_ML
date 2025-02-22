import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained XGBoost model
model = joblib.load("xgb_titanic_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define request schema
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    Fare: float
    SibSp: int
    Parch: int
    Embarked: str

# Preprocess input data
def preprocess_input(data: Passenger):
    df = pd.DataFrame([data.model_dump()])
        
    df = df.assign(Age=df["Age"].fillna(df["Age"].median()))
    df = df.assign(Embarked=df["Embarked"].fillna("S"))

    # Encode categorical variables
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # Ensure correct column order
    features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]
    return df[features].values

@app.post("/predict")
def predict(data: Passenger):
    input_data = preprocess_input(data)
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
