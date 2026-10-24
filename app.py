
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Job Scam Detection",
    page_icon="🔍",
    layout="centered"
)

# ==========================================
# 2. CSS STYLING
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #eef5ff;
    color: #1e293b;
}

.stApp h1, .stApp h2, .stApp h3 {
    color: #17365d !important;
}

.stApp p, .stApp label,
.stApp [data-testid="stMarkdownContainer"] {
    color: #1e293b;
}

.stTextArea textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #94a3b8 !important;
}

.stTextArea textarea::placeholder {
    color: #64748b !important;
}

.stButton > button,
.stDownloadButton > button {
    background-color: #174ea6 !important;
    color: #ffffff !important;
    border: 2px solid #174ea6 !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    padding: 10px 18px !important;
}

.stButton > button p,
.stDownloadButton > button p {
    color: #ffffff !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #0d3678 !important;
    color: #ffffff !important;
}

[data-testid="stMetricValue"] {
    color: #17365d !important;
}

[data-testid="stMetricLabel"] {
    color: #334155 !important;
}

[data-testid="stAlert"] p {
    color: #1e293b !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD DATASET AND TRAIN MODELS
# ==========================================

@st.cache_resource
def load_models():

    df = pd.read_csv("dataset/fake_job_postings.csv")

    df["text"] = (
        df["title"].fillna("") + " " +
        df["company_profile"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["requirements"].fillna("") + " " +
        df["benefits"].fillna("")
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X = vectorizer.fit_transform(df["text"])
    y = df["fraudulent"]

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X, y)

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    fraud_df = df[df["fraudulent"] == 1].copy()

    fraud_embeddings = embedding_model.encode(
        fraud_df["text"].tolist(),
        batch_size=32,
        show_progress_bar=False
    )

    return (
        df,
        vectorizer,
        model,
        embedding_model,
        fraud_df,
        fraud_embeddings
    )

# ==========================================
# 4. RISK LEVEL
# ==========================================

def get_risk_level(probability):

    if probability < 30:
        return "Low Risk"
    elif probability < 70:
        return "Medium Risk"
    else:
        return "High Risk"

# ==========================================
# 5. SCAM INDICATORS
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

    if "security deposit" in text or "deposit" in text:
        indicators.append("Requests a security deposit")

    if "no experience" in text or "without experience" in text:
        indicators.append("No experience required")

    if "work from home" in text or "work-from-home" in text:
        indicators.append("Work-from-home opportunity")

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
        indicators.append("Uses urgent hiring language")

    if "guaranteed" in text:
        indicators.append(
            "Uses guaranteed-income or guaranteed-job language"
        )

    if (
        "easy money" in text
        or "easy income" in text
        or "quick money" in text
    ):
        indicators.append("Promises easy or quick money")

    if "no interview" in text or "without interview" in text:
        indicators.append("Claims that no interview is required")

    if (
        "bank details" in text
        or "bank account" in text
        or "account number" in text
    ):
        indicators.append(
            "Requests sensitive banking information"
        )

    if "whatsapp" in text or "telegram" in text:
        indicators.append(
            "Uses messaging platforms for recruitment"
        )

    return indicators

# ==========================================
# 6. SESSION HISTORY
# ==========================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction_key" not in st.session_state:
    st.session_state.last_prediction_key = None

# ==========================================
# 7. APPLICATION HEADING
# ==========================================

st.title("🔍 Job Scam Detection System")

st.write(
    "Check a job advertisement for possible scam warning "
    "signs using Machine Learning."
)

st.info(
    "Paste the job advertisement below. The result is an estimate, "
    "not a guarantee that a job is genuine or fraudulent."
)

job_text = st.text_area(
    "Enter the job advertisement",
    height=220,
    placeholder=(
        "Example: Urgent hiring! Earn money from home. "
        "No experience required. Pay a registration fee..."
    )
)

# ==========================================
# 8. DETECT JOB BUTTON
# ==========================================

if st.button("🔍 Check Job Advertisement"):

    if not job_text.strip():

        st.warning("Please enter a job advertisement first.")

    else:

        try:

            with st.spinner(
                "Loading models and analyzing the advertisement..."
            ):

                (
                    df,
                    vectorizer,
                    model,
                    embedding_model,
                    fraud_df,
                    fraud_embeddings
                ) = load_models()

                job_vector = vectorizer.transform([job_text])

                probabilities = model.predict_proba(job_vector)[0]

                fraud_probability = float(
                    probabilities[1] * 100
                )

                risk_level = get_risk_level(
                    fraud_probability
                )

                indicators = detect_scam_indicators(
                    job_text
                )

                new_embedding = embedding_model.encode(
                    [job_text]
                )

                similarities = cosine_similarity(
                    new_embedding,
                    fraud_embeddings
                )[0]

                top_indices = similarities.argsort()[-5:][::-1]

            # Save only when a new prediction is made
            prediction_key = job_text.strip()

            if (
                prediction_key
                != st.session_state.last_prediction_key
            ):

                st.session_state.prediction_history.append({
                    "Job Advertisement": job_text.strip(),
                    "Risk Level": risk_level,
                    "Fraud Probability": round(
                        fraud_probability, 2
                    )
                })

                st.session_state.last_prediction_key = (
                    prediction_key
                )

            # ==================================
            # 9. DISPLAY RESULT
            # ==================================

            st.divider()
            st.subheader("📊 Detection Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Risk Level",
                    risk_level
                )

            with col2:
                st.metric(
                    "Estimated Fraud Probability",
                    f"{fraud_probability:.2f}%"
                )

            st.progress(
                min(max(int(fraud_probability), 0), 100)
            )

            if risk_level == "High Risk":

                st.error(
                    "This advertisement has several potential warning signs."
                )

            elif risk_level == "Medium Risk":

                st.warning(
                    "Review this advertisement carefully."
                )

            else:

                st.success(
                    "The model estimates a lower fraud risk."
                )

            # ==================================
            # 10. SCAM INDICATORS
            # ==================================

            st.subheader("⚠️ Scam Indicators")

            if indicators:

                for indicator in indicators:
                    st.write(f"⚠️ {indicator}")

            else:

                st.write(
                    "No listed warning signs were detected."
                )

            # ==================================
            # 11. EXPLANATION
            # ==================================

            st.subheader("💡 Explanation")

            if indicators:

                st.write(
                    "The advertisement contains the warning signs "
                    "listed above. Investigate the employer and "
                    "verify the offer before sharing information "
                    "or making payments."
                )

            else:

                st.write(
                    "The selected warning-sign checks did not find "
                    "a match. This does not prove the job is genuine."
                )

            # ==================================
            # 12. SIMILAR FRAUDULENT JOBS
            # ==================================

            st.subheader(
                "🔎 Similar Fraudulent Job Advertisements"
            )

            for rank, index in enumerate(
                top_indices,
                start=1
            ):

                job = fraud_df.iloc[index]
                score = float(similarities[index])

                with st.container(border=True):

                    st.write(
                        f"**{rank}. {job['title']}**"
                    )

                    st.write(
                        f"Text similarity: {score:.4f}"
                    )

                    st.write(
                        f"Location: {job.get('location', 'Not specified')}"
                    )

            st.caption(
                "Similarity measures how closely the text resembles "
                "known fraudulent advertisements. It is not a "
                "fraud probability."
            )

        except FileNotFoundError:

            st.error(
                "Dataset not found. Check that "
                "'Dataset/fake_job_postings.csv' exists "
                "in your project folder."
            )

        except Exception as error:

            st.error(
                f"Something went wrong: {error}"
            )

# ==========================================
# 13. PREDICTION HISTORY DASHBOARD
# ==========================================

st.divider()
st.subheader("📋 Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    total_jobs = len(history_df)

    high_risk = (
        history_df["Risk Level"] == "High Risk"
    ).sum()

    medium_risk = (
        history_df["Risk Level"] == "Medium Risk"
    ).sum()

    low_risk = (
        history_df["Risk Level"] == "Low Risk"
    ).sum()

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Jobs",
        total_jobs
    )

    col2.metric(
        "High Risk",
        int(high_risk)
    )

    col3.metric(
        "Medium Risk",
        int(medium_risk)
    )

    col4.metric(
        "Low Risk",
        int(low_risk)
    )

    # History table
    st.subheader("Checked Advertisements")

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # Risk distribution chart
    st.subheader("📊 Risk Distribution")

    chart_data = pd.DataFrame({
        "Number of Jobs": [
            int(high_risk),
            int(medium_risk),
            int(low_risk)
        ]
    }, index=[
        "High Risk",
        "Medium Risk",
        "Low Risk"
    ])

    st.bar_chart(chart_data)

    # Download history
    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv_data,
        file_name="job_scam_prediction_history.csv",
        mime="text/csv"
    )

    # Clear history
    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = []
        st.session_state.last_prediction_key = None

        st.rerun()

else:

    st.info(
        "No prediction history yet. Check a job advertisement first."
    )

# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.caption(
    "Job Scam Detection System | Machine Learning Project"
)

st.caption(
    "Predictions are estimates. Always verify employers and job offers independently."
)
