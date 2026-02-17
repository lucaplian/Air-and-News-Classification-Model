"""
News Classification Model
Classifies news articles into different categories.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle


class NewsClassifier:
    """
    News Classification Model using Naive Bayes with TF-IDF
    
    Categories:
        - Business
        - Technology
        - Sports
        - Entertainment
        - Politics
    """
    
    def __init__(self, max_features=5000):
        """
        Initialize the News Classifier
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.model = MultinomialNB()
        self.class_names = ['Business', 'Technology', 'Sports', 
                           'Entertainment', 'Politics']
        
    def prepare_data(self, texts, labels=None):
        """
        Prepare text data using TF-IDF vectorization
        
        Args:
            texts (list): List of text documents
            labels (array-like, optional): Target labels
            
        Returns:
            tuple: Vectorized features and labels (if provided)
        """
        X = self.vectorizer.fit_transform(texts)
        if labels is not None:
            return X, labels
        return X
    
    def train(self, texts, labels, test_size=0.2):
        """
        Train the news classification model
        
        Args:
            texts (list): List of text documents
            labels (array-like): Target labels
            test_size (float): Proportion of data for testing
            
        Returns:
            dict: Training results including accuracy and metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )
        
        # Vectorize the text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_vec, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(
            y_test, y_pred,
            target_names=self.class_names[:len(np.unique(labels))],
            zero_division=0
        )
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'vocab_size': len(self.vectorizer.vocabulary_)
        }
        
        return results
    
    def predict(self, texts):
        """
        Predict news category
        
        Args:
            texts (list): List of text documents
            
        Returns:
            array: Predicted classes
        """
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)
    
    def predict_proba(self, texts):
        """
        Predict probability for each class
        
        Args:
            texts (list): List of text documents
            
        Returns:
            array: Predicted probabilities
        """
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)
    
    def save_model(self, filepath):
        """
        Save the trained model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'class_names': self.class_names
            }, f)
    
    def load_model(self, filepath):
        """
        Load a trained model from disk
        
        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.class_names = data['class_names']


def generate_sample_data(n_samples=500):
    """
    Generate synthetic news article data for demonstration
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        tuple: List of texts and labels
    """
    # Sample news articles for each category
    business_texts = [
        "Stock market reaches new highs as investors show confidence",
        "Company reports strong quarterly earnings beating expectations",
        "Merger and acquisition deal announced between major corporations",
        "Federal Reserve announces interest rate decision",
        "Startup raises millions in venture capital funding",
        "Oil prices fluctuate amid global economic concerns",
        "Retail sales show signs of economic recovery",
        "Banking sector faces new regulatory challenges",
        "Real estate market shows strong growth",
        "International trade negotiations continue"
    ]
    
    technology_texts = [
        "New artificial intelligence breakthrough announced by researchers",
        "Latest smartphone features advanced camera technology",
        "Software company releases major update with new features",
        "Cybersecurity threats on the rise warn experts",
        "Cloud computing adoption accelerates in enterprise",
        "Machine learning algorithm achieves record performance",
        "Tech giant unveils next generation processor",
        "Social media platform introduces new privacy features",
        "Quantum computing research makes significant progress",
        "5G network deployment expands to new cities"
    ]
    
    sports_texts = [
        "Championship game goes into overtime thriller",
        "Star athlete signs record breaking contract",
        "Team wins playoff series advancing to finals",
        "Olympic athlete sets new world record",
        "Coach announces retirement after successful career",
        "Draft picks show promise for upcoming season",
        "Injury sidelines key player for remainder of season",
        "Stadium packed with fans for crucial match",
        "Underdog team pulls off surprising victory",
        "Tournament bracket revealed for upcoming competition"
    ]
    
    entertainment_texts = [
        "Movie breaks box office records in opening weekend",
        "Award show celebrates best performances of the year",
        "Popular TV series announces final season",
        "Music festival lineup features top artists",
        "Celebrity couple announces engagement",
        "Streaming service releases highly anticipated series",
        "Director wins acclaim for latest film",
        "Concert tour sells out venues across country",
        "Book adaptation greenlit for production",
        "Fashion week showcases latest designer collections"
    ]
    
    politics_texts = [
        "Election results show tight race between candidates",
        "New legislation passed by congressional vote",
        "Political debate focuses on key policy issues",
        "Government announces new economic stimulus package",
        "International summit addresses climate change",
        "Campaign rallies draw large crowds",
        "Supreme court ruling impacts national policy",
        "Diplomatic relations strengthened between nations",
        "Reform bill introduced in legislature",
        "Political party convention nominates candidate"
    ]
    
    texts = []
    labels = []
    
    samples_per_class = n_samples // 5
    
    # Generate variations of each category
    for i in range(samples_per_class):
        texts.append(business_texts[i % len(business_texts)])
        labels.append(0)
        
        texts.append(technology_texts[i % len(technology_texts)])
        labels.append(1)
        
        texts.append(sports_texts[i % len(sports_texts)])
        labels.append(2)
        
        texts.append(entertainment_texts[i % len(entertainment_texts)])
        labels.append(3)
        
        texts.append(politics_texts[i % len(politics_texts)])
        labels.append(4)
    
    # Shuffle the data
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return texts, np.array(labels)


if __name__ == '__main__':
    # Generate sample data
    print("Generating sample news article data...")
    texts, labels = generate_sample_data(500)
    
    # Create and train model
    print("\nTraining News Classification Model...")
    classifier = NewsClassifier()
    results = classifier.train(texts, labels)
    
    # Print results
    print(f"\nModel Accuracy: {results['accuracy']:.4f}")
    print(f"Vocabulary Size: {results['vocab_size']}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Test with sample predictions
    print("\nSample Predictions:")
    test_texts = [
        "Technology company releases new software product",
        "Basketball team wins championship game",
        "Stock prices rise on positive economic news"
    ]
    
    predictions = classifier.predict(test_texts)
    for text, pred in zip(test_texts, predictions):
        print(f"  '{text}' -> {classifier.class_names[pred]}")
    
    # Save the model
    classifier.save_model('models/news_classifier_model.pkl')
    print("\nModel saved to models/news_classifier_model.pkl")
