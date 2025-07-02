import os
import pandas as pd
import joblib
import mlflow
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from .pydantic_models import PredictionInput, PredictionOutput

load_dotenv()

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    print("Application starting up...")
    
    # Load the processing pipeline
    pipeline_path = os.path.join('src', 'processing_pipeline.joblib')
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Processing pipeline not found at {pipeline_path}. Run data_processing.py.")
    ml_models['pipeline'] = joblib.load(pipeline_path)
    print("Processing pipeline loaded successfully.")

    # Load the registered model from MLflow
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable not set.")
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    model_name = "CreditRiskModel-RandomForest"
    model_stage = "Production"  # Or "Staging", depending on your workflow
    model_uri = f"models:/{model_name}/{model_stage}"
    
    try:
        ml_models['model'] = mlflow.pyfunc.load_model(model_uri)
        print(f"Model '{model_name}' version '{model_stage}' loaded successfully from MLflow.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from MLflow: {e}")

    yield
    
    # --- Shutdown Logic ---
    print("Application shutting down...")
    ml_models.clear()

# Create the FastAPI app with the lifespan manager
app = FastAPI(
    title="Credit Risk Prediction API",
    description="An API to predict credit risk probability for new customers.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", tags=["Status"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "API is running"}

@app.post("/predict", tags=["Prediction"], response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Predicts the credit risk probability for a single customer.
    """
    if 'pipeline' not in ml_models or 'model' not in ml_models:
        raise HTTPException(status_code=503, detail="Model or pipeline not loaded. Please wait.")

    try:
        # 1. Convert Pydantic input to a Pandas DataFrame
        input_df = pd.DataFrame([input_data.model_dump()])
        
        # 2. Apply the processing pipeline
        processed_input = ml_models['pipeline'].transform(input_df)
        
        # 3. Make a prediction using the loaded model
        risk_probability = ml_models['model'].predict(processed_input)[0]

        # Ensure the probability is a standard float
        risk_probability_float = float(risk_probability)

        return PredictionOutput(risk_probability=risk_probability_float)

    except Exception as e:
        # For any errors during prediction, return an internal server error.
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")