from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
@app.get("/api/")
async def api(a1: float=0, a2: float=0, a3: float=0):
    clf = joblib.load("rf.pkl")
    
    pred = clf.predict([[a1, a2, a3]])
    return {"predict": bool(pred)}
    