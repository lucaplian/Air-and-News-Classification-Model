"""
Demonstration and Evaluation Script
Shows how to use the trained models for predictions
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from air_quality_classifier import AirQualityClassifier
from news_classifier import NewsClassifier


def demo_air_quality_model():
    """Demonstrate air quality classification"""
    print("=" * 80)
    print("AIR QUALITY CLASSIFICATION DEMO")
    print("=" * 80)
    
    # Load the model
    print("\n1. Loading trained model...")
    classifier = AirQualityClassifier()
    
    try:
        classifier.load_model('models/air_quality_model.pkl')
        print("   Model loaded successfully!")
    except FileNotFoundError:
        print("   Model not found. Please run train_models.py first.")
        return
    
    # Sample air quality measurements
    print("\n2. Sample Air Quality Predictions:")
    print(f"\n   Features: {', '.join(classifier.feature_names)}")
    
    samples = [
        {
            'data': [12, 25, 18, 6, 0.6, 45],
            'description': 'Clean city air'
        },
        {
            'data': [45, 85, 72, 52, 5.2, 125],
            'description': 'Moderately polluted industrial area'
        },
        {
            'data': [120, 200, 160, 130, 18, 210],
            'description': 'Heavily polluted urban zone'
        }
    ]
    
    for i, sample in enumerate(samples, 1):
        X = np.array([sample['data']])
        prediction = classifier.predict(X)[0]
        probabilities = classifier.predict_proba(X)[0]
        
        print(f"\n   Sample {i}: {sample['description']}")
        print(f"   Measurements: {sample['data']}")
        print(f"   Prediction: {classifier.class_names[prediction]}")
        print(f"   Confidence: {probabilities[prediction]:.2%}")
        
        # Show top 3 probabilities
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        print(f"   Top predictions:")
        for idx in top_3_idx:
            print(f"      {classifier.class_names[idx]}: {probabilities[idx]:.2%}")


def demo_news_classification_model():
    """Demonstrate news classification"""
    print("\n" + "=" * 80)
    print("NEWS CLASSIFICATION DEMO")
    print("=" * 80)
    
    # Load the model
    print("\n1. Loading trained model...")
    classifier = NewsClassifier()
    
    try:
        classifier.load_model('models/news_classifier_model.pkl')
        print("   Model loaded successfully!")
    except FileNotFoundError:
        print("   Model not found. Please run train_models.py first.")
        return
    
    # Sample news articles
    print("\n2. Sample News Article Predictions:")
    
    articles = [
        "Apple announces new iPhone with revolutionary camera technology and improved battery life.",
        "Local sports team advances to championship finals after dramatic playoff victory.",
        "Stock market reaches all-time high as economic indicators show strong growth.",
        "New blockbuster movie shatters box office records in opening weekend.",
        "Congress debates new healthcare legislation amid partisan disagreement.",
        "Researchers develop breakthrough AI algorithm for medical diagnosis.",
        "Tennis star wins Grand Slam tournament for record fifth time.",
        "Fashion week showcases sustainable clothing collections from top designers."
    ]
    
    for i, article in enumerate(articles, 1):
        predictions = classifier.predict([article])
        probabilities = classifier.predict_proba([article])[0]
        predicted_class = predictions[0]
        
        print(f"\n   Article {i}:")
        print(f"   '{article}'")
        print(f"   Predicted Category: {classifier.class_names[predicted_class]}")
        print(f"   Confidence: {probabilities[predicted_class]:.2%}")


def main():
    """Main demonstration function"""
    print("\n")
    print("*" * 80)
    print("CLASSIFICATION MODELS DEMONSTRATION")
    print("*" * 80)
    print()
    
    # Demo air quality model
    demo_air_quality_model()
    
    # Demo news classification model
    demo_news_classification_model()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
