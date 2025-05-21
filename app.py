# app.py

from dotenv import load_dotenv
load_dotenv()                       # load from local .env if present
import os, openai, joblib, numpy as np, streamlit as st

# Load your OpenAI key from ENV (no st.secrets)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("Please set OPENAI_API_KEY in your environment or in a .env file")
    st.stop()

st.set_page_config(page_title="Spam Detector")
st.title("📧 AI-Based Spam Email Detection")

# — Sidebar controls —
pipeline    = st.sidebar.radio("Pipeline", ["TF-IDF Only", "TF-IDF + Embeddings"])
model_name  = st.sidebar.selectbox("Model", ["NaiveBayes","SVM"])
explain_llm = st.sidebar.checkbox("Enable LLM Explanation (gpt-4o-mini)")

# — Load models once —
@st.cache_resource
def load_artifacts():
    # Baseline
    nb   = joblib.load("models/nb.joblib")
    svm  = joblib.load("models/svm.joblib")
    tf   = joblib.load("models/tfidf.joblib")
    # Embedding-augmented
    emb_svm, emb_gnb = joblib.load("models/emb_models.joblib")
    emb_tf           = joblib.load("models/emb_tfidf.joblib")
    emb_sc           = joblib.load("models/emb_scaler.joblib")
    return (nb, svm, tf), ((emb_svm, emb_gnb), emb_tf, emb_sc)

(b_nb, b_svm, b_tf), ((e_svm, e_gnb), e_tf, e_sc) = load_artifacts()

# — Input area —
uploaded = st.file_uploader("Upload a .txt email", type=None)
if uploaded:
    text = uploaded.read().decode("utf-8", errors="ignore")
else:
    text = st.text_area("Or paste your email text here", height=200)

# — Classification & Explanation —
if st.button("Classify"):
    if not text.strip():
        st.error("Please provide email content.")
        st.stop()

    with st.spinner("Classifying…"):
        # Feature extraction
        if pipeline == "TF-IDF Only":
            vect  = b_tf
            model = b_nb if model_name=="NaiveBayes" else b_svm
            X = vect.transform([text])
        else:
            emb = openai.embeddings.create(
                model="text-embedding-3-small",
                input=[text[:5000]]         # truncate to fit context
            ).data[0].embedding
            emb_arr = e_sc.transform(np.array(emb)[None,:])
            X_tfidf = e_tf.transform([text]).toarray()
            X = np.hstack([X_tfidf, emb_arr])
            model = e_gnb if model_name=="NaiveBayes" else e_svm

        # Prediction + confidence
        pred = model.predict(X)[0]
        if hasattr(model, "predict_proba"):
            conf = np.max(model.predict_proba(X)) * 100
        else:
            score = model.decision_function(X)[0]
            conf  = 100/(1+np.exp(-score))

    st.success(f"**{pred.upper()}** — Confidence: {conf:.1f}%")

    # Feature explanation (top words)
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        feats = (b_tf if pipeline=="TF-IDF Only" else e_tf).get_feature_names_out()
        top_idxs = np.argsort(coefs)[-10:][::-1]
        st.write("Top spam words:", ", ".join(feats[i] for i in top_idxs))

    # LLM‐powered rationale
    if explain_llm:
        with st.spinner("Asking the LLM for an explanation…"):
            prompt = (
                f"Email:\n{text}\n\n"
                f"Prediction: {pred.upper()} (confidence {conf:.1f}%)\n"
                "In 2–3 sentences, explain why this email is classified as "
                f"{pred.upper()}."
            )
            # New v1.x Chat Completion method:
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user", "content": prompt}],
            )
            explanation = resp.choices[0].message.content.strip()
        st.markdown("> **LLM Explanation:**  " + explanation)
