import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/fake_job_postings.csv")

print("Dataset loaded!")
print("Total jobs:", len(df))


# ==========================================
# 2. PREPARE JOB TEXT
# ==========================================

df["text"] = (
    df["title"].fillna("") + " " +
    df["company_profile"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["requirements"].fillna("") + " " +
    df["benefits"].fillna("")
)


# ==========================================
# 3. USE FIRST 100 JOBS
# ==========================================

sample_df = df.head(100).copy()

print("Jobs selected for similarity search:", len(sample_df))


# ==========================================
# 4. LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# 5. CREATE EMBEDDINGS FOR DATASET JOBS
# ==========================================

print("\nCreating embeddings...")

job_embeddings = model.encode(
    sample_df["text"].tolist(),
    show_progress_bar=True
)

print("Embeddings created successfully!")


# ==========================================
# 6. ENTER A NEW JOB
# ==========================================

new_job = input(
    "\nEnter a job advertisement to search:\n"
)


# ==========================================
# 7. CREATE EMBEDDING FOR NEW JOB
# ==========================================

new_job_embedding = model.encode([new_job])


# ==========================================
# 8. CALCULATE SIMILARITY
# ==========================================

similarities = cosine_similarity(
    new_job_embedding,
    job_embeddings
)[0]


# ==========================================
# 9. FIND TOP 5 SIMILAR JOBS
# ==========================================

top_indices = similarities.argsort()[-5:][::-1]


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("       MOST SIMILAR JOB POSTINGS")
print("==========================================")


for rank, index in enumerate(top_indices, start=1):

    job = sample_df.iloc[index]

    similarity_score = similarities[index]

    print("\n------------------------------------------")

    print("Rank:", rank)

    print(
        "Similarity Score:",
        round(similarity_score, 4)
    )

    print(
        "Title:",
        job["title"]
    )

    print(
        "Fraudulent:",
        job["fraudulent"]
    )


print("\n==========================================")