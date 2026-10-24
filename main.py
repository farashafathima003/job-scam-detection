import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "dataset/fake_job_postings.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))


# ============================================================
# 2. PREPARE TEXT DATA
# ============================================================

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

X_text = df["combined_text"]
y = df["fraudulent"]

print("\nTraining data prepared successfully!")


# ============================================================
# 3. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(X_text)

print("\nTF-IDF conversion completed!")
print("X shape:", X.shape)
print("Y shape:", y.shape)


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData splitting completed!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 5. TRAIN MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\nModel training completed!")


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 7. RISK LEVEL
# ============================================================

def get_risk_level(probability):

    if probability < 30:
        return "Low Risk"

    elif probability < 70:
        return "Medium Risk"

    else:
        return "High Risk"


# ============================================================
# 8. DETECT SCAM INDICATORS
# ============================================================

def detect_scam_indicators(job_text):

    text = job_text.lower()

    indicators = []

    # Upfront payment
    payment_patterns = [
        r"registration fee",
        r"registration fees",
        r"pay.*fee",
        r"payment.*required",
        r"pay.*to start",
        r"pay.*before",
        r"upfront payment",
        r"joining fee",
        r"processing fee"
    ]

    for pattern in payment_patterns:
        if re.search(pattern, text):
            indicators.append(
                "Requests an upfront payment or registration fee."
            )
            break

    # Security deposit
    security_patterns = [
        r"security deposit",
        r"deposit.*job",
        r"refundable deposit"
    ]

    for pattern in security_patterns:
        if re.search(pattern, text):
            indicators.append(
                "Requests a security deposit."
            )
            break

    # No experience
    if re.search(
        r"no experience|no prior experience|without experience",
        text
    ):
        indicators.append(
            "Claims that no experience is required."
        )

    # Work from home
    if re.search(
        r"work from home|work-from-home|wfh|remote job",
        text
    ):
        indicators.append(
            "Promotes work-from-home or remote work."
        )

    # Attractive earnings
    if re.search(
        r"earn.*per month|earn.*daily|high income|attractive salary|"
        r"earn.*weekly|\₹[\d,]+|rs\.?\s?[\d,]+",
        text
    ):
        indicators.append(
            "Promises unusually attractive earnings."
        )

    # Urgency
    if re.search(
        r"apply now|hurry|limited time|immediately|urgent|"
        r"start immediately|act now",
        text
    ):
        indicators.append(
            "Uses urgency or pressure to apply quickly."
        )

    # Guaranteed income
    if re.search(
        r"guaranteed income|guaranteed salary|guaranteed job|"
        r"100% guaranteed|assured income",
        text
    ):
        indicators.append(
            "Promises guaranteed income or employment."
        )

    # Easy money
    if re.search(
        r"easy money|quick money|get rich|make money fast|"
        r"earn easily",
        text
    ):
        indicators.append(
            "Promises easy or quick money."
        )

    # No interview
    if re.search(
        r"no interview|without interview|interview not required",
        text
    ):
        indicators.append(
            "Claims that no interview is required."
        )

    # Bank details
    if re.search(
        r"bank account|bank details|account number|"
        r"credit card|debit card",
        text
    ):
        indicators.append(
            "Requests sensitive financial information."
        )

    # WhatsApp / Telegram
    if re.search(
        r"whatsapp|telegram",
        text
    ):
        indicators.append(
            "Uses messaging platforms such as WhatsApp or Telegram."
        )

    # Identity documents
    if re.search(
        r"aadhaar|aadhar|passport|pan card|identity proof|"
        r"government id",
        text
    ):
        indicators.append(
            "Requests personal identity information."
        )

    return indicators


# ============================================================
# 9. GENERATE EXPLANATION
# ============================================================

def generate_explanation(
    probability,
    risk_level,
    indicators
):

    if risk_level == "High Risk":

        explanation = (
            "This job advertisement shows several characteristics "
            "commonly associated with fraudulent job postings. "
            "Users should verify the employer and avoid making "
            "payments or sharing sensitive personal information."
        )

    elif risk_level == "Medium Risk":

        explanation = (
            "This job advertisement contains some potentially "
            "suspicious characteristics. Further verification of "
            "the company and job offer is recommended."
        )

    else:

        explanation = (
            "This job advertisement shows relatively fewer "
            "suspicious characteristics based on the trained model. "
            "However, users should still verify the employer "
            "before sharing personal information or accepting an offer."
        )

    return explanation


# ============================================================
# 10. PREDICT JOB
# ============================================================

def predict_job(job_text):

    # Convert input text into TF-IDF
    job_vector = vectorizer.transform([job_text])

    # Fraud probability
    probabilities = model.predict_proba(job_vector)[0]

    fraud_probability = probabilities[1] * 100

    # Risk level
    risk_level = get_risk_level(fraud_probability)

    # Scam indicators
    indicators = detect_scam_indicators(job_text)

    # Explanation
    explanation = generate_explanation(
        fraud_probability,
        risk_level,
        indicators
    )

    # Return API-friendly dictionary
    result = {
        "risk_level": risk_level,
        "fraud_probability": round(fraud_probability, 2),
        "scam_indicators": indicators,
        "explanation": explanation
    }

    return result


# ============================================================
# 11. RUN FROM TERMINAL
# ============================================================
# IMPORTANT:
# This section only runs when you directly execute:
# python main.py
#
# It will NOT run when FastAPI imports this file.

if __name__ == "__main__":

    print("\n==========================================")
    print("       JOB SCAM DETECTION SYSTEM")
    print("==========================================")

    job_text = input(
        "\nEnter the job advertisement:\n"
    )

    result = predict_job(job_text)

    print("\n==========================================")
    print("              RESULT")
    print("==========================================")

    print("\nRisk Level:", result["risk_level"])

    print(
        "Fraud Probability:",
        str(result["fraud_probability"]) + "%"
    )

    print("\nScam Indicators:")

    if result["scam_indicators"]:

        for indicator in result["scam_indicators"]:
            print("-", indicator)

    else:
        print("- No major scam indicators detected.")

    print("\nExplanation:")
    print(result["explanation"])

    print("\n==========================================")