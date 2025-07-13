# sentiment_analysis.py – Step 3

import pandas as pd

# Load dataset
df = pd.read_csv("data/sentiment140.csv")

# Preview the data
print("🔹 First 5 rows:")
print(df.head())

# Check class distribution
print("\n🔍 Sentiment value counts:")
print(df['sentiment'].value_counts())

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean tweet text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Preview cleaned text
print("\n🧹 Cleaned Tweets:")
print(df[['text', 'clean_text']].head())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Vectorize the cleaned text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])

# Labels (0 = negative, 1 = neutral, 2 = positive)
y = df['sentiment']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

import joblib

# Save the trained model
joblib.dump(model, "sentiment_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
