"""
Air Quality Classification Model
Classifies air quality based on various pollutant measurements.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle


class AirQualityClassifier:
    """
    Air Quality Classification Model using Random Forest
    
    Features: PM2.5, PM10, NO2, SO2, CO, O3
    Target: Air Quality Index (AQI) Categories
        - Good (0)
        - Moderate (1)
        - Unhealthy for Sensitive Groups (2)
        - Unhealthy (3)
        - Very Unhealthy (4)
        - Hazardous (5)
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the Air Quality Classifier
        
        Args:
            n_estimators (int): Number of trees in the random forest
            random_state (int): Random state for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10
        )
        self.scaler = StandardScaler()
        self.feature_names = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        self.class_names = ['Good', 'Moderate', 'Unhealthy (Sensitive)', 
                           'Unhealthy', 'Very Unhealthy', 'Hazardous']
        
    def prepare_data(self, X, y=None):
        """
        Prepare and scale the input data
        
        Args:
            X (array-like): Feature matrix
            y (array-like, optional): Target labels
            
        Returns:
            tuple: Scaled features and labels (if provided)
        """
        X_scaled = self.scaler.fit_transform(X)
        if y is not None:
            return X_scaled, y
        return X_scaled
    
    def train(self, X, y, test_size=0.2):
        """
        Train the air quality classification model
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target labels
            test_size (float): Proportion of data for testing
            
        Returns:
            dict: Training results including accuracy and metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(
            y_test, y_pred, 
            target_names=self.class_names[:len(np.unique(y))],
            zero_division=0
        )
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'feature_importance': dict(zip(self.feature_names, 
                                          self.model.feature_importances_))
        }
        
        return results
    
    def predict(self, X):
        """
        Predict air quality class
        
        Args:
            X (array-like): Feature matrix
            
        Returns:
            array: Predicted classes
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict probability for each class
        
        Args:
            X (array-like): Feature matrix
            
        Returns:
            array: Predicted probabilities
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, filepath):
        """
        Save the trained model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
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
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.class_names = data['class_names']


def generate_sample_data(n_samples=1000):
    """
    Generate synthetic air quality data for demonstration
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        tuple: Feature matrix X and labels y
    """
    np.random.seed(42)
    
    # Generate features with different distributions for each class
    X = []
    y = []
    
    samples_per_class = n_samples // 6
    
    # Good air quality (class 0)
    X.append(np.random.normal([10, 20, 15, 5, 0.5, 40], [5, 10, 5, 2, 0.2, 10], 
                              (samples_per_class, 6)))
    y.extend([0] * samples_per_class)
    
    # Moderate (class 1)
    X.append(np.random.normal([25, 50, 40, 20, 2, 80], [8, 15, 10, 5, 0.5, 15], 
                              (samples_per_class, 6)))
    y.extend([1] * samples_per_class)
    
    # Unhealthy for Sensitive (class 2)
    X.append(np.random.normal([45, 80, 70, 50, 5, 120], [10, 20, 15, 8, 1, 20], 
                              (samples_per_class, 6)))
    y.extend([2] * samples_per_class)
    
    # Unhealthy (class 3)
    X.append(np.random.normal([70, 120, 100, 80, 10, 160], [15, 25, 20, 10, 2, 25], 
                              (samples_per_class, 6)))
    y.extend([3] * samples_per_class)
    
    # Very Unhealthy (class 4)
    X.append(np.random.normal([100, 180, 150, 120, 15, 200], [20, 30, 25, 15, 3, 30], 
                              (samples_per_class, 6)))
    y.extend([4] * samples_per_class)
    
    # Hazardous (class 5)
    X.append(np.random.normal([150, 250, 200, 180, 25, 250], [25, 40, 30, 20, 5, 40], 
                              (samples_per_class, 6)))
    y.extend([5] * samples_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


if __name__ == '__main__':
    # Generate sample data
    print("Generating sample air quality data...")
    X, y = generate_sample_data(1000)
    
    # Create and train model
    print("\nTraining Air Quality Classification Model...")
    classifier = AirQualityClassifier()
    results = classifier.train(X, y)
    
    # Print results
    print(f"\nModel Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nFeature Importance:")
    for feature, importance in results['feature_importance'].items():
        print(f"  {feature}: {importance:.4f}")
    
    # Save the model
    classifier.save_model('models/air_quality_model.pkl')
    print("\nModel saved to models/air_quality_model.pkl")
