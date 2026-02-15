"""
Simple Flask API for Sentiment Analysis
Run with: python api.py
"""

from flask import Flask, request, jsonify
from predict import SentimentAnalyzer

app = Flask(__name__)

# Load the model once when the app starts
print("Loading sentiment analysis model...")
analyzer = SentimentAnalyzer()
print("Model loaded successfully!")

@app.route('/')
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'Sentiment Analysis API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Analyze sentiment of text',
            '/health': 'GET - Check API health'
        },
        'example': {
            'url': '/predict',
            'method': 'POST',
            'body': {'text': 'This product is amazing!'}
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'loaded'})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for given text"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Please provide text in JSON format: {"text": "your text here"}'
            }), 400
        
        text = data['text']
        
        if not text or not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Make prediction
        result = analyzer.predict(text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict sentiment for multiple texts"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Please provide texts in JSON format: {"texts": ["text1", "text2"]}'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({'error': 'texts must be a list'}), 400
        
        if not texts:
            return jsonify({'error': 'texts list cannot be empty'}), 400
        
        # Make predictions
        results = analyzer.predict_batch(texts)
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n Starting Sentiment Analysis API...")
    print(" API will be available at: http://localhost:5000")
    print("\nExample usage:")
    print('  curl -X POST http://localhost:5000/predict \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"text": "This is amazing!"}\'\n')
    
    app.run(debug=True, host='0.0.0.0', port=5000)
