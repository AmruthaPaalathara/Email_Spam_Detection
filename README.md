# AI-Based Spam Email Detector

## Overview  
This repository contains an end-to-end pipeline for detecting spam emails using:

1. **Classical NLP** (TF-IDF + MultinomialNB & LinearSVC)  
2. **Hybrid ML** (TF-IDF + OpenAI embeddings → GaussianNB & LinearSVC)  
3. **Optional LLM explanations** (`gpt-4o-mini`)  

All code—including data parsing, preprocessing, model training, evaluation and a Streamlit UI—is included.

---

##  Key Features

- **Spam/Ham classification**  
  Upload or paste an email; the app labels it **SPAM** or **HAM**.

- **Confidence score**  
  Displays the model’s predicted probability or decision-function–derived confidence.

- **Top-10 spam-indicative words**  
  Shows the highest-weight features from the classifier (e.g. “free”, “click”, “offer”).

- **(Optional) LLM rationale** via `gpt-4o-mini`  
  When enabled, sends a brief prompt to OpenAI to explain _why_ the email was labeled spam/ham.

---

## ⚙️ Installation & Setup

1. **Clone & enter**  
   ```bash
   git clone https://github.com/your-username/Email_Spam_Detector.git
   cd Email_Spam_Detector


## Accuracy details

NB: acc=0.950, p=1.000, r=0.700, f1=0.824
SVM: acc=0.987, p=0.989, r=0.930, f1=0.959

## Limitations
Dataset bias: Trained on SpamAssassin 2003 corpus; may not generalize to modern spam.

Attachment & HTML: Does not parse HTML rendering or attachments.

Token limits & cost: Embedding & LLM calls incur API cost and latency; batching/truncation mitigates limits but may lose context.

Model simplicity: Uses TF-IDF + classical classifiers; more complex spam patterns may require transformer fine-tuning.

No real-time ingestion: Demo is interactive, not hooked to live mail servers.

Privacy: Emails sent to OpenAI API—avoid sensitive content.