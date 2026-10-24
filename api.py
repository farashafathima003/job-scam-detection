from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import predict_job
from fraud_similarity import find_similar_fraudulent_jobs


app = FastAPI(
    title="Job Scam Detection API",
    description="API for detecting potentially fraudulent job advertisements",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class JobRequest(BaseModel):
    job_text: str


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Job Scam Detection API is running!"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# PREDICT JOB
# ============================================================

@app.post("/predict")
def predict(request: JobRequest):

    # ML prediction
    result = predict_job(
        request.job_text
    )

    # Find similar fraudulent jobs
    similar_jobs = find_similar_fraudulent_jobs(
        request.job_text,
        top_k=5
    )

    # Add similar jobs to result
    result["similar_fraudulent_jobs"] = similar_jobs

    return result