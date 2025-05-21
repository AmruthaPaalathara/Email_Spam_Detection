import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources (only needs to run once)
nltk.download('punkt')       # tokenizer models
nltk.download('punkt_tab')   # sentence tokenizer support
nltk.download('stopwords')   # list of stopwords
nltk.download('wordnet')     # WordNet lemmatizer data

# Initialize NLP tools
stop = set(stopwords.words('english'))  # English stop-word set
stemmer = PorterStemmer()               # Porter stemming algorithm
lemmatizer = WordNetLemmatizer()        # WordNet lemmatization

def clean(text: str) -> str:
    """
    Normalize input text:
    1) Lowercase & tokenize into words
    2) Filter out non-alpha tokens and stop-words
    3) Stem tokens, then lemmatize stems
    4) Return cleaned text as a single space-joined string
    """
    tokens = nltk.word_tokenize(text.lower())             # step 1
    tokens = [t for t in tokens if t.isalpha() and t not in stop]  # step 2
    stems = [stemmer.stem(t) for t in tokens]              # step 3a
    return " ".join(lemmatizer.lemmatize(s) for s in stems)  # step 3b & 4

def load_and_preprocess(path='data/spamassassin_raw.csv'):
    """
    1) Load raw CSV of parsed emails
    2) Concatenate 'from', 'subject', and 'body' into 'full_text'
    3) Clean each full_text via `clean()`
    4) Vectorize cleaned text using TF-IDF (max 5000 features)
    Returns:
      - X: sparse TF-IDF feature matrix
      - y: series of labels ('ham'/'spam')
      - tfidf: fitted TfidfVectorizer instance
    """
    df = pd.read_csv(path)                                # step 1

    # step 2: merge metadata and content
    df['full_text'] = (
        df['from'].fillna('') + ' '
        + df['subject'].fillna('') + ' '
        + df['body'].fillna('')
    )

    # step 3: apply cleaning function
    df['clean'] = df['full_text'].apply(clean)

    # step 4: fit TF-IDF on cleaned text
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean'])

    return X, df['label'], tfidf

if __name__ == "__main__":
    # When run as a script, print the shape and label distribution
    X, y, tf = load_and_preprocess()
    print(f"Shape: {X.shape}, Labels: {y.value_counts().to_dict()}")
