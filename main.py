from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import uvicorn

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
@app.get("/api/")
async def api(temp: float=0, dTdt: float=0, dTdx: float=0):
    clf = joblib.load("rf.pkl")
    
    pred = clf.predict([[temp, dTdt, dTdx]])
    return {"predict": bool(pred)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")