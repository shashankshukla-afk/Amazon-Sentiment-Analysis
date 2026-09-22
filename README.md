## Overview

An NLP-based sentiment analysis application that classifies Amazon customer reviews into:

- Negative
- Neutral
- Positive

The project uses a fine-tuned DistilBERT model and provides a Streamlit web interface for sentiment prediction.

## Features

- Sentiment prediction
- Confidence score
- Sentiment gauge
- Session-based prediction history
- Simple Streamlit interface

## Model

```text
distilbert/distilbert-base-uncased

The model was fine-tuned for three-class sentiment classification:

0 → Negative
1 → Neutral
2 → Positive

Project Structure

Amazon-Sentiment-Analysis/
│
├── app.py
├── README.md
├── requirements.txt
├── distilbert_sentiment_final/
├── logistic_regression_model.pkl
├── balanced_logistic_regression_model.pkl
└── traditional_ml_results.xlsx

Run Locally

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/Amazon-Sentiment-Analysis.git

2. Open the project folder

cd Amazon-Sentiment-Analysis

3. Install dependencies

pip install -r requirements.txt

4. Run the application

streamlit run app.py

Open the local URL shown in the terminal, usually:

http://localhost:8501

Usage

1. Open the Home page.


2. Enter a customer review.


3. Click Analyze Sentiment.


4. View the predicted sentiment and confidence.


5. Check the sentiment gauge.


6. Use History to view reviews analyzed during the current session.
