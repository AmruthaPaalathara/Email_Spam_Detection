# src/train.py

from dotenv import load_dotenv
load_dotenv()           # Load OPENAI_API_KEY from .env
import os
import joblib
import openai
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB     # ← use GaussianNB here
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from src.preprocess import load_and_preprocess

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embeddings(texts, batch_size=100, max_len_chars=5000):
    all_embs = []
    for start in range(0, len(texts), batch_size):
        batch = [t[:max_len_chars] for t in texts[start:start+batch_size]]
        resp = openai.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        for entry in resp.data:
            all_embs.append(entry.embedding)
    return np.array(all_embs)

def main():
    # 1. Load TF-IDF features + labels
    X_tfidf, y, tfidf = load_and_preprocess()

    # 2. Build full_text for embeddings
    df = pd.read_csv('data/spamassassin_raw.csv')
    df['full_text'] = (
        df['from'].fillna('') + ' '
      + df['subject'].fillna('') + ' '
      + df['body'].fillna('')
    )
    texts = df['full_text'].tolist()

    # 3. Fetch embeddings
    print(f" Embedding {len(texts)} emails…")
    embs = get_embeddings(texts)

    # 4. Combine features
    scaler = StandardScaler().fit(embs)
    X = np.hstack([X_tfidf.toarray(), scaler.transform(embs)])

    # 5. Train/test split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 6. Train SVM + GaussianNB
    print(" Training models…")
    svm = LinearSVC(max_iter=10000).fit(Xtr, ytr)
    gnb = GaussianNB().fit(Xtr, ytr)

    # 7. Save
    os.makedirs('models', exist_ok=True)
    joblib.dump((svm, gnb),   'models/emb_models.joblib')
    joblib.dump(tfidf,        'models/emb_tfidf.joblib')
    joblib.dump(scaler,       'models/emb_scaler.joblib')

    print(" Embedding-augmented models (SVM + GaussianNB) saved to /models")

if __name__ == "__main__":
    main()
