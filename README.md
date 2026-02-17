# Air Quality and News Classification Models

The second assignment for the course of Artificial Intelligence

## Overview

This project implements two machine learning classification models:

1. **Air Quality Classification Model** - Classifies air quality based on pollutant measurements (PM2.5, PM10, NO2, SO2, CO, O3)
2. **News Classification Model** - Classifies news articles into categories (Business, Technology, Sports, Entertainment, Politics)

## Project Structure

```
Air-and-News-Classification-Model/
├── src/
│   ├── air_quality_classifier.py    # Air quality classification model
│   └── news_classifier.py           # News classification model
├── models/                           # Saved trained models (generated)
├── notebooks/                        # Jupyter notebooks for analysis
├── data/                            # Data directory
├── train_models.py                  # Main training script
├── demo.py                          # Demonstration script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lucaplian/Air-and-News-Classification-Model.git
cd Air-and-News-Classification-Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Models

To train both classification models, run:

```bash
python train_models.py
```

This will:
- Generate synthetic training data for both models
- Train the air quality classifier using Random Forest
- Train the news classifier using Naive Bayes with TF-IDF
- Save the trained models to the `models/` directory
- Display performance metrics and evaluation results

### Running the Demo

To see the trained models in action:

```bash
python demo.py
```

This demonstrates:
- Loading saved models
- Making predictions on sample data
- Displaying prediction results with confidence scores

### Using the Models in Your Code

#### Air Quality Classification

```python
from src.air_quality_classifier import AirQualityClassifier
import numpy as np

# Load trained model
classifier = AirQualityClassifier()
classifier.load_model('models/air_quality_model.pkl')

# Make predictions
# Features: [PM2.5, PM10, NO2, SO2, CO, O3]
sample_data = np.array([[45, 85, 72, 52, 5.2, 125]])
prediction = classifier.predict(sample_data)
probabilities = classifier.predict_proba(sample_data)

print(f"Air Quality: {classifier.class_names[prediction[0]]}")
```

#### News Classification

```python
from src.news_classifier import NewsClassifier

# Load trained model
classifier = NewsClassifier()
classifier.load_model('models/news_classifier_model.pkl')

# Make predictions
articles = ["Technology company announces new AI breakthrough"]
prediction = classifier.predict(articles)
probabilities = classifier.predict_proba(articles)

print(f"Category: {classifier.class_names[prediction[0]]}")
```

## Models

### Air Quality Classifier

- **Algorithm**: Random Forest
- **Features**: 
  - PM2.5 (Fine particulate matter)
  - PM10 (Coarse particulate matter)
  - NO2 (Nitrogen dioxide)
  - SO2 (Sulfur dioxide)
  - CO (Carbon monoxide)
  - O3 (Ozone)
  
- **Classes**:
  - Good (0)
  - Moderate (1)
  - Unhealthy for Sensitive Groups (2)
  - Unhealthy (3)
  - Very Unhealthy (4)
  - Hazardous (5)

### News Classifier

- **Algorithm**: Multinomial Naive Bayes with TF-IDF
- **Features**: TF-IDF vectors (up to 5000 features)
- **Classes**:
  - Business
  - Technology
  - Sports
  - Entertainment
  - Politics

## Model Performance

Both models achieve high accuracy on their respective tasks:
- Air Quality Classifier: ~95%+ accuracy
- News Classifier: ~90%+ accuracy

*Note: Performance metrics are based on synthetic data. Real-world performance may vary.*

## Requirements

- Python 3.7+
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter (optional, for notebooks)

## Future Improvements

- Add real-world datasets
- Implement deep learning models (LSTM, BERT for news classification)
- Add cross-validation
- Hyperparameter tuning
- Web interface for predictions
- API endpoints for model serving

## License

This project is created for educational purposes as part of an Artificial Intelligence course assignment.

## Author

Luca Plian
