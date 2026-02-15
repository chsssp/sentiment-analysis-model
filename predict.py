"""
Sentiment Analysis Inference Script
Load the trained model and make predictions on new text
"""

import joblib
import json

class SentimentAnalyzer:
    """Simple sentiment analyzer class"""
    
    def __init__(self, model_path='model'):
        """Load the trained model and vectorizer"""
        self.model = joblib.load(f'{model_path}/sentiment_model.pkl')
        self.vectorizer = joblib.load(f'{model_path}/vectorizer.pkl')
        
        with open(f'{model_path}/metadata.json', 'r') as f:
            self.metadata = json.load(f)
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        text_tfidf = self.vectorizer.transform([text])
        prediction = self.model.predict(text_tfidf)[0]
        probabilities = self.model.predict_proba(text_tfidf)[0]
        
        return {
            'text': text,
            'sentiment': prediction,
            'confidence': float(max(probabilities)),
            'probabilities': {
                self.metadata['classes'][i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        return [self.predict(text) for text in texts]

def main():
    """Demo the sentiment analyzer"""
    print("🎭 Sentiment Analysis Demo\n")
    
    # Load the analyzer
    analyzer = SentimentAnalyzer()
    
    # Test examples
    examples = [
        "I absolutely loved this product! Best purchase ever!",
        "Horrible experience, completely unsatisfied.",
        "It's okay, nothing special but gets the job done.",
        "Outstanding quality and excellent customer service!",
        "Waste of money, very disappointing."
    ]
    
    print("Analyzing sample texts:\n")
    for example in examples:
        result = analyzer.predict(example)
        print(f"📝 Text: {result['text']}")
        print(f"😊 Sentiment: {result['sentiment'].upper()}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"   Probabilities: {result['probabilities']}")
        print()

if __name__ == "__main__":
    main()
