from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example job advertisement
job_text = """
Work from home job. Earn Rs. 50,000 per month.
No experience required. Pay Rs. 2,000 registration fee.
"""

# Convert text into embedding
embedding = model.encode(job_text)

print("Embedding created successfully!")
print("Embedding size:", len(embedding))

print("\nFirst 10 values:")
print(embedding[:10])