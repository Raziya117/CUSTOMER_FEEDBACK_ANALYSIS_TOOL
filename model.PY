import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords and tokenizer
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text)).lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def load_and_prepare_dataset(csv_file):
    df = pd.read_csv(csv_file,encoding='latin1')
    df = df[['Rate', 'Review', 'Summary']].dropna()

    # Combine Review and Summary
    df['text'] = df['Review'].astype(str) + " " + df['Summary'].astype(str)

    # Create sentiment based on rating
    df['sentiment'] = df['Rate'].apply(lambda x: 'positive' if x >= 4 else ('neutral' if x == 3 else 'negative'))

    # Preprocess
    df['clean_text'] = df['text'].apply(preprocess)
    return df

def train_model(df):
    X = df['clean_text']
    y = df['sentiment']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    clean = preprocess(text)
    vector = vectorizer.transform([clean])
    return model.predict(vector)[0]
