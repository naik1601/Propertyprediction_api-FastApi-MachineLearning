import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import io

app=FastAPI()
model=joblib.load("housing_model.joblib")
feature_names=joblib.load("feature_names.joblib")

class PropertyFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income of the block")
    HouseAge: float = Field(gt=0, description="Age of the house")
    AveRooms: float = Field(gt=0, description="Average number of rooms")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms")
    Population: float = Field(gt=0, description="Population of the block")
    AveOccup: float = Field(gt=0, description="Average occupancy")
    Latitude: float = Field(gt=0, description="Latitude of the block")
    Longitude: float = Field(gt=0, description="Longitude of the block")

@app.get("/")
def home():
    return {"message": "Welcome to the Property Prediction API",
            "status": "success",
            "endpoint": "send POST request to /predict"
            }
@app.get("/health")
def health_check():
    return {"status": "running",
            "model":"RandomForestRegressor",
            "features": feature_names,}

@app.post("/predict")
def predict_price(features: PropertyFeatures):
    try:
        input_data = pd.DataFrame([
            {
            "MedInc": features.MedInc,
            "HouseAge": features.HouseAge,
            "AveRooms": features.AveRooms,
            "AveBedrms": features.AveBedrms,
            "Population": features.Population,
            "AveOccup": features.AveOccup,
            "Latitude": features.Latitude,
            "Longitude": features.Longitude
            }
            ])
        
      
        prediction = model.predict(input_data)[0]
        predic_usd = prediction * 100000
        
        return {"predicted_price":  f"${predic_usd:,.0f} hundread thousand",
                "predicted_price_short":  f"${predic_usd:,.2f} hundread thousand"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predictfile")
async def predict_price_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")
    
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    required_columns = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"
    ]
    
    if not all(col in df.columns for col in required_columns):
        raise HTTPException(status_code=400, detail="Invalid CSV format. Please ensure all required columns are present.")

    try:
        
        predictions=model.predict(df)
        predic_usd = predictions * 100000
        output=df.to_csv(index=False)
        return StreamingResponse(io.BytesIO(output.encode()), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

   
    return {"predicted_prices": predic_usd.tolist()}