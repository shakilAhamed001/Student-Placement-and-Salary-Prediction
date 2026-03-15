from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- Load Models ---
placement_model = joblib.load('logistic_regression_placement_model.joblib')
salary_model = joblib.load('linear_regression_model.joblib')

# --- FastAPI App Instance ---
app = FastAPI(title="Student Placement and Salary Prediction API")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define Pydantic Models for Input Data ---

# Features used for placement prediction
class PlacementInput(BaseModel):
    cgpa: float
    coding_skills: float
    dsa_score: float
    aptitude_score: float
    communication_skills: float
    ml_knowledge: float
    system_design: float
    internships: int
    projects_count: int
    certifications: int
    hackathons: int
    backlogs: int

# Features used for salary prediction
# Note: Use Literal for categorical features to ensure valid inputs
class SalaryInput(BaseModel):
    cgpa: float
    coding_skills: float
    dsa_score: float
    aptitude_score: float
    communication_skills: float
    ml_knowledge: float
    system_design: float
    internships: int
    projects_count: int
    certifications: int
    hackathons: int
    backlogs: int
    open_source_contributions: int
    extracurriculars: int
    # Use the unique values derived from the dataset
    # Ensure these match the exact unique values from df['branch'].unique().tolist()
    branch: Literal['ECE', 'Chemical', 'EE', 'CE', 'CSE', 'IT', 'ME']
    # Ensure these match the exact unique values from df['college_tier'].unique().tolist()
    college_tier: Literal['Tier-3', 'Tier-2', 'Tier-1']

# --- Define Pydantic Models for Output Data ---
class PlacementOutput(BaseModel):
    predicted_placement_status: int # 0 for Not Placed, 1 for Placed

class SalaryOutput(BaseModel):
    predicted_salary_lpa: float


# --- Prediction Endpoints ---

@app.post("/predict_placement", response_model=PlacementOutput, summary="Predict Student Placement Status")
async def predict_placement(data: PlacementInput):
    """
    Predicts whether a student will be placed (1) or not (0) based on their academic and skill metrics.
    """
    input_df = pd.DataFrame([data.dict()])
    prediction = placement_model.predict(input_df)[0]
    return {"predicted_placement_status": int(prediction)}

@app.post("/predict_salary", response_model=SalaryOutput, summary="Predict Salary Package for Placed Students")
async def predict_salary(data: SalaryInput):
    """
    Predicts the expected salary package (in LPA) for a student, assuming they are placed.
    This model should be used for placed students only.
    """
    input_dict = data.dict()

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])

    non_cat_features = [
        'cgpa', 'coding_skills', 'dsa_score', 'aptitude_score',
        'communication_skills', 'ml_knowledge', 'system_design',
        'internships', 'projects_count', 'certifications',
        'hackathons', 'backlogs', 'open_source_contributions',
        'extracurriculars'
    ]

    processed_df = input_df[non_cat_features].copy()

    for branch_val in ['CSE', 'Chemical', 'ECE', 'EE', 'IT', 'ME']:
        processed_df[f'branch_{branch_val}'] = (input_df['branch'] == branch_val).astype(int)

    for tier_val in ['Tier-2', 'Tier-3']:
        processed_df[f'college_tier_{tier_val}'] = (input_df['college_tier'] == tier_val).astype(int)

    # Reorder to match exact training column order
    processed_df = processed_df[list(salary_model.feature_names_in_)]

    prediction = salary_model.predict(processed_df)[0]
    return {"predicted_salary_lpa": round(float(prediction), 2)}