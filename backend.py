import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI()

# Load the model
try:
    model = joblib.load('./model/random_forest_protein_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class ProteinFeatures(BaseModel):
    Nneg: int
    Npos: int
    Exc1: float
    Exc2: float
    I_Index: float
    A_Index: float
    GRAVY: float
    Ser: int
    Thr: int
    Tyr: int
    ExpAA: float
    PredHel: int

@app.get("/")
async def root():
    return {"message": "Protein Classification API is running"}

@app.post("/predict")
async def predict(features: ProteinFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        feature_array = np.array([[
            features.Nneg,
            features.Npos,
            features.Exc1,
            features.Exc2,
            features.I_Index,
            features.A_Index,
            features.GRAVY,
            features.Ser,
            features.Thr,
            features.Tyr,
            features.ExpAA,
            features.PredHel
        ]])
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        probabilities = model.predict_proba(feature_array)[0]
        
        # Get class names
        class_names = model.classes_
        
        # Create response with probabilities for each class
        response = {
            "predicted_class": str(prediction),  # Convert to string to ensure serialization
            "probabilities": {
                str(class_name): float(prob)  # Convert class name to string and probability to float
                for class_name, prob in zip(class_names, probabilities)
            }
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 