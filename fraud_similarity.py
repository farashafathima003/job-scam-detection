import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOAD DATASET
# ============================================================

DATASET_PATH = "dataset/fake_job_postings.csv"

df = pd.read_csv(DATASET_PATH)

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

for column in text_columns:
    df[column] = df[column].fillna("")

df["combined_text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)


# ============================================================
# SELECT FRAUDULENT JOBS
# ============================================================

fraudulent_jobs = df[df["fraudulent"] == 1].copy()

print(
    "Number of fraudulent jobs:",
    len(fraudulent_jobs)
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

print("Creating embeddings for fraudulent jobs...")

fraud_embeddings = embedding_model.encode(
    fraudulent_jobs["combined_text"].tolist(),
    show_progress_bar=True
)

print("Fraudulent job embeddings created!")


# ============================================================
# FIND SIMILAR FRAUDULENT JOBS
# ============================================================

def find_similar_fraudulent_jobs(
    job_text,
    top_k=5
):

    # Create embedding for new job
    job_embedding = embedding_model.encode(
        [job_text]
    )

    # Calculate similarity
    similarities = cosine_similarity(
        job_embedding,
        fraud_embeddings
    )[0]

    # Get highest similarity indexes
    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        job = fraudulent_jobs.iloc[index]

        results.append({
            "similarity": round(
                float(similarities[index]),
                4
            ),

            "title": str(
                job["title"]
            ),

            "location": str(
                job["location"]
            )
        })

    return results