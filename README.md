# 🐦 Twitter Sentiment Analysis

A simple NLP project to classify tweets as **positive**, **neutral**, or **negative** using a cleaned Sentiment140 dataset and a Logistic Regression model.

---

## 📂 Project Structure

Twitter-Sentiment-Analysis/
├── data/
│ └── sentiment140.csv # Dataset (15 rows for testing)
├── sentiment_analysis.py # Python script (data cleaning, training)
├── sentiment_model.pkl # Saved model
├── tfidf_vectorizer.pkl # Saved vectorizer
└── README.md # Project documentation



---

## ⚙️ Technologies Used

- Python
- NLTK
- Scikit-learn
- Pandas
- TfidfVectorizer
- Logistic Regression

---

## 🔍 How it Works

1. Loads and cleans tweet data
2. Removes stopwords, punctuation, etc.
3. Converts text to numerical vectors (TF-IDF)
4. Trains a Logistic Regression model
5. Predicts tweet sentiment

---

## 📈 Accuracy

Since we used only 15 sample tweets (5 per class), accuracy is low (0.00) — this is expected and intended for **learning and demonstration purposes** only.

---

## ▶️ To Run

Make sure your terminal is in the project folder, then run:

```bash
python sentiment_analysis.py


💡 Future Improvements
Use full Sentiment140 dataset from Kaggle (1.6M tweets)

Apply deep learning (e.g., LSTM or BERT)

Build a Streamlit web app for real-time predictions

📬 Author
A. Sanjay Goud
3rd-Year B.Tech | Data Science & AI
This project was done as part of my AI Internship at Codec Technologies