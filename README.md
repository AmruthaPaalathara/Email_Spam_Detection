# AI-Based Spam Email Detector

## Overview  
This project implements an email spam detector using classical NLP and modern embeddings + LLM techniques on the SpamAssassin Public Corpus. It supports two pipelines:

1. **TF-IDF Only**  
   - Preprocess text → TF-IDF vector (5 000 features)  
   - Train MultinomialNB & LinearSVC  
2. **TF-IDF + Embeddings**  
   - Concatenate TF-IDF + OpenAI `text-embedding-3-small` (batched & truncated)  
   - Train GaussianNB & LinearSVC  

A Streamlit UI enables uploading/pasting emails, selecting pipeline/model, viewing predictions with confidence, top-10 indicative words, and (optional) LLM explanations via `gpt-4o-mini`.

---

## 🔧 Requirements

- Python 3.10+  
- A valid OpenAI API key  
- An Internet connection for embedding & LLM calls  

**Install dependencies**:
```bash
pip install -r requirements.txt
