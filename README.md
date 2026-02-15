# Sentiment Analysis Model

A beginner-friendly sentiment analysis model for classifying text as positive or negative sentiment. Built with scikit-learn and ready to deploy!

## Model Overview

- **Task**: Binary Text Classification (Sentiment Analysis)
- **Algorithm**: Logistic Regression with TF-IDF features
- **Framework**: scikit-learn
- **Performance**: ~80-90% accuracy on test data

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/chsssp/sentiment-analysis-model.git
cd sentiment-analysis-model

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
python train_model.py
```

This will:
- Load/create the training dataset
- Train a Logistic Regression model
- Evaluate performance
- Save the model to the `model/` directory

### Using the Model

```python
from predict import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Predict sentiment
result = analyzer.predict("This product is amazing!")
print(result)
# Output: {'text': '...', 'sentiment': 'positive', 'confidence': 0.95, ...}
```

Or run the demo:

```bash
python predict.py
```

## Project Structure

```
sentiment-analysis-model/
├── train_model.py          # Training script
├── predict.py              # Inference script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── MODEL_CARD.md          # Hugging Face model card
└── model/                 # Saved model files (generated)
    ├── sentiment_model.pkl
    ├── vectorizer.pkl
    └── metadata.json
```

## Use Cases

- Social media sentiment monitoring
- Product review analysis
- Customer feedback classification
- Brand reputation tracking
- Survey response analysis

## Customization

### Using Your Own Data

Replace the `create_sample_data()` function in `train_model.py` with your own dataset:

```python
def load_your_data():
    df = pd.read_csv('your_data.csv')
    # df should have 'text' and 'sentiment' columns
    return df
```

### Improving the Model

- Increase training data size
- Try different algorithms (SVM, Random Forest, etc.)
- Tune hyperparameters
- Use more advanced features (n-grams, word embeddings)

## Model Performance

The model achieves good accuracy on sentiment classification tasks. Performance metrics are displayed after training and saved in `model/metadata.json`.

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## License

MIT License - feel free to use this project for learning and development!

## Acknowledgments

Built with:
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

## Contact

For questions or feedback, please open an issue on GitHub.

