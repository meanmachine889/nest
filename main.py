from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import datetime
from fastapi.middleware.cors import CORSMiddleware

# Load the model, encoder, vectorizer, and feature columns
model = joblib.load('model1.pkl')
encoder = joblib.load('encoder1.pkl')
vectorizer = joblib.load('vectorizer1.pkl')
feature_columns = joblib.load('feature_columns1.pkl')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Input data schema
class ClinicalTrialInput(BaseModel):
    nct_id: Optional[str] = None
    Study_Title: str
    Conditions: str
    Interventions: str
    Phases: str
    Enrollment: int
    Study_Type: str
    Study_Design: str
    Start_Date: datetime.date
    Primary_Completion_Date: datetime.date
    Locations: str

def preprocess_data(input_data: pd.DataFrame, encoder, vectorizer, feature_columns):
    # Ensure consistent column names
    input_data.columns = input_data.columns.str.replace('_', ' ')
    
    # Define categorical columns
    categorical_columns = ['Phases', 'Study Type']
    
    # Encode categorical features
    encoded_cats = encoder.transform(input_data[categorical_columns])
    encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Process text features
    text_features = []
    for col in ['Study Title', 'Conditions', 'Interventions', 'Study Design', 'Locations']:
        tfidf_matrix = vectorizer.transform(input_data[col].fillna(''))
        text_features.append(pd.DataFrame(tfidf_matrix.toarray(), columns=[f"{col}tfidf{i}" for i in range(tfidf_matrix.shape[1])]))
    text_features_df = pd.concat(text_features, axis=1)
    
    # Combine all features
    processed_data = pd.concat([input_data[['Enrollment']], encoded_cats_df, text_features_df], axis=1)
    for col in feature_columns:
        if col not in processed_data.columns:
            processed_data[col] = 0
    processed_data = processed_data[feature_columns]
    return processed_data



@app.post("/predict")
async def predict(input_data: ClinicalTrialInput):
    # Convert input to DataFrame
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])
    
    # Preprocess input
    processed_input_data = preprocess_data(input_df, encoder, vectorizer, feature_columns)
    
    # Make prediction
    prediction = model.predict(processed_input_data)
    predicted_days = prediction[0]
    predicted_date = pd.to_datetime(input_dict["Start_Date"]) + pd.to_timedelta(predicted_days, unit='D')
    
    return {"predicted_completion_date": str(predicted_date.date())}
