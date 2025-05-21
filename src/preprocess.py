import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download once
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean(text: str) -> str:
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop]
    stems = [stemmer.stem(t) for t in tokens]
    return " ".join(lemmatizer.lemmatize(s) for s in stems)

def load_and_preprocess(path='data/spamassassin_raw.csv'):
    df = pd.read_csv(path)
    # Combine metadata + body
    df['full_text'] = (
        df['from'].fillna('') + ' '
        + df['subject'].fillna('') + ' '
        + df['body'].fillna('')
    )
    # Clean
    df['clean'] = df['full_text'].apply(clean)
    # Vectorize
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean'])
    return X, df['label'], tfidf

if __name__ == "__main__":
    X, y, tf = load_and_preprocess()
    print(f"Shape: {X.shape}, Labels: {y.value_counts().to_dict()}")
