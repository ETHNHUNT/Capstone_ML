import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained XGBoost model
model = joblib.load("xgb_titanic_model.pkl")

# Initialize FastAPI app
app = FastAPI()

