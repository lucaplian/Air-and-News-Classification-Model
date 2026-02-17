"""
Main training script for Air Quality and News Classification Models
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from air_quality_classifier import AirQualityClassifier, generate_sample_data as generate_air_data
from news_classifier import NewsClassifier, generate_sample_data as generate_news_data


def train_air_quality_model():
    """Train and evaluate the air quality classification model"""
    print("=" * 80)
    print("AIR QUALITY CLASSIFICATION MODEL")
    print("=" * 80)
    
    # Generate sample data
    print("\n1. Generating sample air quality data...")
    X, y = generate_air_data(1000)
    print(f"   Generated {len(X)} samples with {X.shape[1]} features")
    
    # Create and train model
    print("\n2. Training Air Quality Classification Model...")
    classifier = AirQualityClassifier(n_estimators=100)
    results = classifier.train(X, y, test_size=0.2)
    
    # Print results
    print(f"\n3. Model Performance:")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print("\n   Classification Report:")
    for line in results['classification_report'].split('\n'):
        if line.strip():
            print(f"   {line}")
    
    print("\n   Feature Importance:")
    for feature, importance in sorted(results['feature_importance'].items(), 
                                      key=lambda x: x[1], reverse=True):
        print(f"   {feature:10s}: {importance:.4f}")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    classifier.save_model('models/air_quality_model.pkl')
    print("\n4. Model saved to models/air_quality_model.pkl")
    
    return classifier, results


def train_news_classification_model():
    """Train and evaluate the news classification model"""
    print("\n" + "=" * 80)
    print("NEWS CLASSIFICATION MODEL")
    print("=" * 80)
    
    # Generate sample data
    print("\n1. Generating sample news article data...")
    texts, labels = generate_news_data(500)
    print(f"   Generated {len(texts)} news articles across 5 categories")
    
    # Create and train model
    print("\n2. Training News Classification Model...")
    classifier = NewsClassifier(max_features=5000)
    results = classifier.train(texts, labels, test_size=0.2)
    
    # Print results
    print(f"\n3. Model Performance:")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   Vocabulary Size: {results['vocab_size']}")
    print("\n   Classification Report:")
    for line in results['classification_report'].split('\n'):
        if line.strip():
            print(f"   {line}")
    
    # Test with sample predictions
    print("\n4. Sample Predictions:")
    test_texts = [
        "Technology company announces new artificial intelligence product launch",
        "Championship basketball team wins finals in overtime thriller",
        "Stock market shows gains as economy strengthens and investor confidence grows",
        "New movie premieres to sold out theaters breaking box office records",
        "Presidential election campaign focuses on healthcare and economic policy"
    ]
    
    predictions = classifier.predict(test_texts)
    probabilities = classifier.predict_proba(test_texts)
    
    for i, (text, pred, probs) in enumerate(zip(test_texts, predictions, probabilities)):
        print(f"\n   Example {i+1}:")
        print(f"   Text: '{text[:70]}...'")
        print(f"   Predicted: {classifier.class_names[pred]}")
        print(f"   Confidence: {probs[pred]:.4f}")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    classifier.save_model('models/news_classifier_model.pkl')
    print("\n5. Model saved to models/news_classifier_model.pkl")
    
    return classifier, results


def main():
    """Main function to train both models"""
    print("\n")
    print("*" * 80)
    print("ARTIFICIAL INTELLIGENCE ASSIGNMENT 2")
    print("AIR QUALITY AND NEWS CLASSIFICATION MODELS")
    print("*" * 80)
    
    # Train air quality model
    air_classifier, air_results = train_air_quality_model()
    
    # Train news classification model
    news_classifier, news_results = train_news_classification_model()
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nAir Quality Model Accuracy:  {air_results['accuracy']:.4f}")
    print(f"News Classification Accuracy: {news_results['accuracy']:.4f}")
    print("\nBoth models have been trained and saved successfully!")
    print("\nModel files:")
    print("  - models/air_quality_model.pkl")
    print("  - models/news_classifier_model.pkl")
    print("\n" + "*" * 80)


if __name__ == '__main__':
    main()
