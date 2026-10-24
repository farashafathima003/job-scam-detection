import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/fake_job_postings.csv")

print("Dataset loaded!")
print("Total jobs:", len(df))


# ==========================================
# 2. PREPARE TEXT DATA
# ==========================================

df["text"] = (
    df["title"].fillna("") + " " +
    df["company_profile"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["requirements"].fillna("") + " " +
    df["benefits"].fillna("")
)

print("Text data prepared!")


# ==========================================
# 3. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(df["text"])
y = df["fraudulent"]


# ==========================================
# 4. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Fraud detection model trained!")


# ==========================================
# 6. LOAD EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ==========================================
# 7. SELECT FRAUDULENT JOBS
# ==========================================

fraud_df = df[df["fraudulent"] == 1].copy()

print(
    "Fraudulent jobs available for similarity search:",
    len(fraud_df)
)


# ==========================================
# 8. CREATE FRAUDULENT JOB EMBEDDINGS
# ==========================================

print("\nCreating fraudulent job embeddings...")

fraud_embeddings = embedding_model.encode(
    fraud_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True
)

print("Fraudulent embeddings created!")


# ==========================================
# 9. RISK LEVEL
# ==========================================

def get_risk_level(probability):

    if probability < 30:
        return "Low Risk"

    elif probability < 70:
        return "Medium Risk"

    else:
        return "High Risk"


# ==========================================
# 10. SCAM INDICATORS
# ==========================================

def detect_scam_indicators(job_text):

    indicators = []

    text = job_text.lower()

    if (
        "registration fee" in text
        or "processing fee" in text
        or "application fee" in text
        or ("pay" in text and "fee" in text)
    ):
        indicators.append(
            "Requests upfront payment or registration fee"
        )

    if (
        "security deposit" in text
        or "deposit" in text
    ):
        indicators.append(
            "Requests a security deposit"
        )

    if (
        "no experience" in text
        or "without experience" in text
    ):
        indicators.append(
            "No experience required"
        )

    if (
        "work from home" in text
        or "work-from-home" in text
    ):
        indicators.append(
            "Work-from-home opportunity"
        )

    if (
        "earn" in text
        and (
            "per month" in text
            or "monthly" in text
            or "daily" in text
        )
    ):
        indicators.append(
            "Promises high or attractive earnings"
        )

    if (
        "urgent" in text
        or "immediately" in text
        or "act now" in text
        or "limited vacancies" in text
    ):
        indicators.append(
            "Uses urgent hiring language"
        )

    if (
        "guaranteed job" in text
        or "guaranteed income" in text
        or "guaranteed salary" in text
        or "guaranteed" in text
    ):
        indicators.append(
            "Uses guaranteed-income or guaranteed-job language"
        )

    if (
        "easy money" in text
        or "easy income" in text
        or "quick money" in text
    ):
        indicators.append(
            "Promises easy or quick money"
        )

    if (
        "no interview" in text
        or "without interview" in text
    ):
        indicators.append(
            "Claims that no interview is required"
        )

    if (
        "bank details" in text
        or "bank account" in text
        or "account number" in text
    ):
        indicators.append(
            "Requests sensitive banking information"
        )

    if (
        "whatsapp" in text
        or "telegram" in text
    ):
        indicators.append(
            "Uses messaging platforms for recruitment"
        )

    return indicators


# ==========================================
# 11. EXPLANATION
# ==========================================

def generate_explanation(indicators):

    if not indicators:

        return (
            "No major warning signs were detected "
            "in this job advertisement."
        )

    explanations = []

    for indicator in indicators:

        if "upfront payment" in indicator.lower():

            explanations.append(
                "The job asks applicants to pay money "
                "before starting the job."
            )

        elif "security deposit" in indicator.lower():

            explanations.append(
                "The advertisement requests a security deposit."
            )

        elif "no experience" in indicator.lower():

            explanations.append(
                "The advertisement states that no experience is required."
            )

        elif "work-from-home" in indicator.lower():

            explanations.append(
                "The advertisement offers a work-from-home opportunity."
            )

        elif "attractive earnings" in indicator.lower():

            explanations.append(
                "The advertisement promises attractive earnings."
            )

        elif "urgent" in indicator.lower():

            explanations.append(
                "The advertisement uses urgent hiring language."
            )

        elif "guaranteed" in indicator.lower():

            explanations.append(
                "The advertisement uses guaranteed job or income language."
            )

        elif "easy or quick money" in indicator.lower():

            explanations.append(
                "The advertisement promises easy or quick money."
            )

        elif "no interview" in indicator.lower():

            explanations.append(
                "The advertisement claims that no interview is required."
            )

        elif "banking information" in indicator.lower():

            explanations.append(
                "The advertisement requests sensitive banking information."
            )

        elif "messaging platforms" in indicator.lower():

            explanations.append(
                "The advertisement uses a messaging platform for recruitment."
            )

    return " ".join(explanations)


# ==========================================
# 12. DETECT JOB
# ==========================================

def detect_job(job_text):

    # --------------------------------------
    # Fraud probability
    # --------------------------------------

    job_vector = vectorizer.transform([job_text])

    probability = model.predict_proba(job_vector)[0]

    fraud_probability = probability[1] * 100

    risk_level = get_risk_level(fraud_probability)


    # --------------------------------------
    # Scam indicators
    # --------------------------------------

    indicators = detect_scam_indicators(job_text)

    explanation = generate_explanation(indicators)


    # --------------------------------------
    # Similar fraudulent jobs
    # --------------------------------------

    new_embedding = embedding_model.encode(
        [job_text]
    )

    similarities = cosine_similarity(
        new_embedding,
        fraud_embeddings
    )[0]

    top_indices = similarities.argsort()[-5:][::-1]


    # ======================================
    # DISPLAY RESULT
    # ======================================

    print("\n==========================================")
    print("       JOB SCAM DETECTION RESULT")
    print("==========================================")

    print(
        "\nRisk Level:",
        risk_level
    )

    print(
        "Fraud Probability:",
        round(fraud_probability, 2),
        "%"
    )


    # --------------------------------------
    # Indicators
    # --------------------------------------

    print("\nScam Indicators:")

    if indicators:

        for indicator in indicators:
            print("-", indicator)

    else:

        print("- No obvious indicators detected")


    # --------------------------------------
    # Explanation
    # --------------------------------------

    print("\nExplanation:")

    print(explanation)


    # --------------------------------------
    # Similar Jobs
    # --------------------------------------

    print("\nMost Similar Fraudulent Jobs:")

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        job = fraud_df.iloc[index]

        score = similarities[index]

        print("\n", rank, ".", job["title"])

        print(
            "Similarity:",
            round(score, 4)
        )

        print(
            "Location:",
            job["location"]
        )


    print("\n==========================================")


# ==========================================
# 13. USER INPUT
# ==========================================

print("\n==========================================")
print("       JOB SCAM DETECTION SYSTEM")
print("==========================================")

job_text = input(
    "\nEnter the job advertisement:\n"
)

detect_job(job_text)