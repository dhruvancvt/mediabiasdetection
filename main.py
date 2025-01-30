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
example_text = "its a glorious day"
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
sample = "Overall, when respondents are asked what outlet they turn to most often for news about government and politics, the most frequent mentions are two cable networks: CNN (named by 16%) and Fox News (14%). But wide ideological differences exist both in the sources that top the list for those on the left and right and in the degree to which there is reliance on a single source. Those with consistently conservative political values are oriented around a single outlet—Fox News—to a much greater degree than those in any other ideological group: Nearly half (47%) of those who are consistently conservative name Fox News as their main source for government and political news. Far fewer choose any other single source: Local radio ranks second, named by 11%, with no other individual source named by more than 5% of consistent conservatives. Those with mostly conservative views also gravitate strongly toward Fox News – 31% name it as their main source, several times the share who name the next most popular sources, including CNN (9%), local television (6%) and radio (6%) and Yahoo News (6%)."
result = predict_bias(sample)
print(result)

