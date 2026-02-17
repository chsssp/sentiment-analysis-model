"""
Sentiment Analysis Model Training Script
A beginner-friendly text classification model for sentiment analysis
NOW WITH 3 CLASSES: Positive, Neutral, Negative
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os

def create_sample_data():
    """Create a sample dataset for demonstration with 3 classes"""
    # Positive reviews
    positive_reviews = [
        "This movie was absolutely wonderful! I loved every minute of it.",
        "Amazing performance by the lead actor. Highly recommended!",
        "A masterpiece! One of the best films I've ever seen.",
        "Fantastic storyline with great character development.",
        "Brilliant cinematography and excellent acting.",
        "I enjoyed this movie thoroughly. Great entertainment!",
        "Excellent film with an engaging plot.",
        "Loved it! Will definitely watch again.",
        "Captivating from start to finish!",
        "Superb direction and outstanding performances.",
        "This product exceeded all my expectations!",
        "Best purchase I've made this year. Highly satisfied.",
        "Outstanding quality and fast shipping.",
        "Absolutely perfect! Works exactly as described.",
        "Five stars! Could not be happier with this.",
        "Great value for money. Would buy again.",
        "Impressive build quality and design.",
        "Exceeded my expectations in every way.",
        "Wonderful experience from start to finish.",
        "Highly recommend to anyone looking for quality.",
        "This is exactly what I needed. Perfect!",
        "Incredible product. Worth every penny.",
        "So happy with this purchase!",
        "Amazing! Better than I imagined.",
        "Superb customer service and great product.",
    ]
    
    negative_reviews = [
        "Terrible film, waste of time and money.",
        "Boring and predictable. Would not watch again.",
        "Disappointing and poorly executed.",
        "Awful movie, couldn't even finish it.",
        "Not worth watching, very disappointing.",
        "Poor script and bad acting throughout.",
        "Completely unwatchable garbage.",
        "Terrible waste of two hours.",
        "Dull and uninteresting.",
        "Worst movie I've seen in years.",
        "Very disappointed with this purchase.",
        "Poor quality. Broke after one use.",
        "Not as described. Waste of money.",
        "Horrible product. Do not buy.",
        "Cheaply made and doesn't work properly.",
        "Completely useless. Total waste.",
        "Terrible customer service and poor quality.",
        "Not worth the price. Very disappointed.",
        "Broke immediately. Very poor quality.",
        "Awful experience. Would not recommend.",
        "Defective product. Requesting refund.",
        "Misleading description. Not what I expected.",
        "Horrible! Nothing like the pictures.",
        "Waste of time and money.",
        "Poor craftsmanship and materials.",
    ]
    
    # NEW: Neutral reviews
    neutral_reviews = [
        "It's okay, nothing special but gets the job done.",
        "Average quality, meets basic expectations.",
        "Not bad, not great, just okay.",
        "It works fine, nothing to complain about.",
        "Acceptable product for the price.",
        "Neither impressed nor disappointed.",
        "Standard quality, as expected.",
        "Decent, but there are better options.",
        "It's fine, does what it's supposed to do.",
        "Mediocre performance, could be better.",
        "Nothing extraordinary, just average.",
        "Satisfactory, but not exceptional.",
        "It's alright, no strong feelings either way.",
        "Reasonable product, no major issues.",
        "Pretty standard, nothing stands out.",
        "Moderate quality, acceptable.",
        "It's okay for the price point.",
        "Average experience overall.",
        "Fair product, meets minimum requirements.",
        "Neither good nor bad, just neutral.",
        "Ordinary quality, nothing special.",
        "It works, that's about it.",
        "Adequate for basic needs.",
        "No complaints, but nothing impressive.",
        "Standard product, average performance.",
    ]
    
    data = {
        'text': positive_reviews + negative_reviews + neutral_reviews,
        'sentiment': (
            ['positive'] * len(positive_reviews) + 
            ['negative'] * len(negative_reviews) + 
            ['neutral'] * len(neutral_reviews)
        )
    }
    return pd.DataFrame(data)

def train_sentiment_model():
    """Train the sentiment analysis model"""
    
    print("Starting Sentiment Analysis Model Training (3-Class)")
    print()
    
    # Load or create data
    print("Loading dataset...")
    df = create_sample_data()
    print(f"Dataset size: {len(df)} samples")
    print(f"Class distribution:")
    print(df['sentiment'].value_counts())
    print()
    
    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['sentiment'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['sentiment']  # Ensures balanced split
    )
    
    # Create TF-IDF vectorizer
    print("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train the model
    print("Training Logistic Regression model (multi-class)...")
    model = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and vectorizer
    print("Saving model and vectorizer...")
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/sentiment_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    
    # Save model metadata
    metadata = {
        'model_type': 'Logistic Regression (Multi-class)',
        'num_classes': 3,
        'accuracy': float(accuracy),
        'num_features': vectorizer.max_features,
        'classes': model.classes_.tolist()
    }
    
    with open('model/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✨ Model training complete! Files saved in 'model/' directory\n")
    
    return model, vectorizer, metadata

def test_model(model, vectorizer):
    """Test the model with some examples"""
    print("🧪 Testing model with sample predictions:\n")
    
    test_texts = [
        "This is an amazing product, I love it!",
        "Terrible experience, very disappointed.",
        "Pretty good, would recommend to others.",
        "It's okay, nothing special.",
        "Absolutely fantastic! Best ever!",
        "Horrible, complete waste of money.",
        "Average quality, does the job."
    ]
    
    for text in test_texts:
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]
        probability = model.predict_proba(text_tfidf)[0]
        
        print(f"Text: '{text}'")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {max(probability):.2%}\n")

if __name__ == "__main__":
    # Train the model
    model, vectorizer, metadata = train_sentiment_model()
    
    # Test the model
    test_model(model, vectorizer)
    
