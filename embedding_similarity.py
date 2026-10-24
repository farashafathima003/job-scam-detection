from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# 2. TWO JOB ADVERTISEMENTS
# ==========================================

job_1 = """
Work from home job. Earn Rs. 50,000 per month.
No experience required. Pay Rs. 2,000 registration fee.
Start working immediately.
"""

job_2 = """
Online work from home opportunity.
Earn Rs. 45,000 every month.
No previous experience needed.
Applicants must pay a registration fee before starting.
"""


# ==========================================
# 3. CREATE EMBEDDINGS
# ==========================================

embedding_1 = model.encode([job_1])
embedding_2 = model.encode([job_2])


# ==========================================
# 4. CALCULATE SIMILARITY
# ==========================================

similarity = cosine_similarity(
    embedding_1,
    embedding_2
)


# ==========================================
# 5. DISPLAY RESULT
# ==========================================

print("\n==========================================")
print("       JOB SIMILARITY TEST")
print("==========================================")

print(
    "Similarity Score:",
    round(similarity[0][0], 4)
)

print("\n==========================================")