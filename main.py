import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from transformers import pipeline

# Load Dataset
data = pd.read_csv("/Users/dhruvanavinchander1/Desktop/mediabiasdetection/news_bias_dataset.csv")
texts = data['content']
labels = data['bias']

# Preprocessing
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([word for word in tokens if word.isalnum() and word not in stop_words])

data['processed_text'] = data['content'].apply(preprocess)

# Feature Extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['processed_text']).toarray()
y = labels

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# Sentiment Analysis Pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
example_text = "This is a sample news article."
sentiment_result = sentiment_pipeline(example_text)
print(sentiment_result)

# Prediction Function
def predict_bias(text):
    processed = preprocess(text)
    features = tfidf.transform([processed]).toarray()
    bias_label = model.predict(features)
    sentiment = sentiment_pipeline(text)
    return {"bias": bias_label[0], "sentiment": sentiment}

# Example Usage
sample = "Former President Bill Clinton said Wednesday that President-elect Donald Trump had won the 2024 race fair and square, in contrast to what he still feels was an illegitimate result in 2016.This time, Donald Trump won the race, fair and square, Clinton told The View, adding, I think.In an appearance on the ABC talk show, the ex-president was reminded by co-host Joy Behar that he wrote in his memoir he was so outraged over his wife Hillary Clintons loss in 2016 that he couldnt sleep."
result = predict_bias(sample)
print(result)

