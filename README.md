

# Bias Detection and Sentiment Analysis

This project is designed to detect biases and analyze sentiment in text data using a combination of machine learning and transformer-based models. The project utilizes a Random Forest classifier to detect biases in text and a sentiment analysis pipeline to assess the sentiment of the text.

## Requirements

- Python 3.x
- Pandas
- Numpy
- NLTK
- scikit-learn
- transformers

You can install the required dependencies using pip:

```bash
pip install pandas numpy nltk scikit-learn transformers
```

## Files

- `dataset.csv`: A CSV file containing the text data and their associated bias labels.
- `bias_detection.py`: The main Python script implementing bias detection and sentiment analysis.

## Dataset

The dataset should be in CSV format with at least two columns:
- `text`: The text data (e.g., news articles, social media posts).
- `bias`: The corresponding bias label (e.g., `biased` or `unbiased`).

## Preprocessing

1. **Tokenization**: The text is tokenized into individual words.
2. **Stopwords Removal**: Common stopwords (like 'the', 'and', etc.) are removed to reduce noise.
3. **Lowercasing**: All text is converted to lowercase for uniformity.

## Feature Extraction

The text data is converted into a numerical format using TF-IDF (Term Frequency-Inverse Document Frequency) to quantify the importance of each word.

## Model Training

A Random Forest classifier is trained on the processed text data to predict the bias label (e.g., whether a piece of text is biased or unbiased). The model is evaluated using classification metrics such as precision, recall, and F1-score.

## Sentiment Analysis

The project also integrates a sentiment analysis model from the Hugging Face transformers library to analyze the sentiment of the text. This model categorizes text into three sentiment categories: Positive, Neutral, or Negative.

## Functions

### `preprocess(text)`
Preprocesses the input text by tokenizing it, removing stopwords, and performing other text-cleaning tasks.

### `predict_bias(text)`
Predicts the bias label and sentiment of the input text. It uses the trained Random Forest model for bias prediction and the Hugging Face sentiment analysis pipeline for sentiment prediction.

#### Example Usage:

```python
sample = "The government announced new policies to address climate change."
result = predict_bias(sample)
print(result)
```

## Output Example:

```json
{
  "bias": "unbiased",
  "sentiment": [{"label": "POSITIVE", "score": 0.99}]
}
```

## Evaluation

The performance of the bias detection model is evaluated on a test set, and classification metrics such as precision, recall, and F1-score are printed.

## Conclusion

This project combines classical machine learning and state-of-the-art transformer models to analyze text data for biases and sentiment. It provides a way to assess both the ideological slant of text and the emotional tone of the content.

