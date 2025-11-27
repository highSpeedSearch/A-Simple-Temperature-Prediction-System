# A-Simple-Temperature-Prediction-System
A Simple Temperature Prediction System Based on Random Forest
# Random Forest Temperature Prediction System

This is a temperature prediction system based on Random Forest algorithm. It can train a model using historical weather data and predict future temperatures.

## Table of Contents

1. [System Overview](#system-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Data Format](#data-format)
5. [Usage](#usage)
6. [Code Structure](#code-structure)
7. [Output Files](#output-files)
8. [Dependencies](#dependencies)

## System Overview

The Random Forest Temperature Prediction System uses machine learning techniques to predict temperatures based on historical weather data. The system implements a Random Forest Regressor from scikit-learn to model complex relationships between various weather features and actual temperatures.

## Features

- **Random Forest Modeling**: Uses scikit-learn's RandomForestRegressor for temperature prediction
- **Comprehensive Model Evaluation**: Provides multiple metrics including MAPE, MSE, RMSE, MAE, and R²
- **Feature Importance Analysis**: Identifies which features contribute most to temperature predictions
- **Visualization Capabilities**: Generates charts for feature importance and prediction accuracy
- **Decision Tree Visualization**: Creates visual representation of individual decision trees
- **Cross-dataset Prediction**: Can train on one dataset and predict on another
- **Automatic Feature Alignment**: Handles differences in features between training and prediction datasets

## Installation

To run this system, you need Python 3.x and the following packages:

```bash
pip install pandas numpy scikit-learn matplotlib pydot
```

Additionally, for decision tree visualization, you need to install Graphviz software:
- Windows: Download from https://graphviz.org/download/#windows
- macOS: `brew install graphviz`
- Linux (Ubuntu/Debian): `sudo apt-get install graphviz`

## Data Format

The system works with CSV files containing weather data with the following columns:

- `year`: Year of the observation
- `month`: Month of the observation
- `day`: Day of the observation
- `week`: Day of the week
- `temp_2`: Temperature from 2 days ago
- `temp_1`: Temperature from 1 day ago
- `average`: Historical average temperature
- `actual`: Actual temperature (target variable)
- `friend`: Friend's temperature prediction

Additional features may be included, and the system will automatically handle one-hot encoding for categorical variables.

## Usage

Run the system with:

```bash
python random_forest.py
```

The system will:
1. Load and process the training data from `temps_extended.csv`
2. Train a Random Forest model
3. Evaluate the model performance
4. Generate visualization charts
5. Predict temperatures for data in `temps.csv`

## Code Structure

### TempPre Class

The main functionality is encapsulated in the `TempPre` class:

#### Initialization
```python
def __init__(self, path)
```
Initializes the predictor with data from the specified CSV file.

#### Model Training
```python
def random_forest(self, max_depth, n_estimators)
```
Trains the Random Forest model with specified parameters.

#### Model Evaluation
```python
def evaluate_model(self)
```
Evaluates the model using multiple metrics.

#### Feature Importance
```python
def get_feature_importance(self)
```
Calculates and returns feature importances.

#### Visualization
```python
def plot_feature_importance(self, top_n=10)
def plot_predictions(self)
def renderTree(self, whichtree)
```
Generate various charts and visualizations.

#### Prediction
```python
def predict(self, data=None)
def predict_external_data(self, external_path)
```
Make predictions on test data or external datasets.

### Main Function

The `main()` function demonstrates a complete workflow:
1. Train model using `temps_extended.csv`
2. Evaluate model performance
3. Generate visualizations
4. Predict temperatures in `temps.csv`

## Output Files

The system generates several output files:

- `feature_importance.png`: Bar chart showing feature importances
- `predictions.png`: Scatter plot comparing actual vs predicted values
- `tree.png`: Visualization of a single decision tree (requires Graphviz)

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **pydot**: Graph visualization interface
- **Graphviz** (optional): Required for decision tree visualization

## Implementation Details

### Feature Processing

The system automatically handles:
- One-hot encoding of categorical variables using `pd.get_dummies()`
- Feature alignment between training and prediction datasets
- Missing feature handling by setting them to 0

### Model Evaluation Metrics

The system calculates multiple metrics for comprehensive model evaluation:
- **MAPE (Mean Absolute Percentage Error)**: Measures prediction accuracy as a percentage
- **MSE (Mean Squared Error)**: Measures average squared difference between actual and predicted values
- **RMSE (Root Mean Squared Error)**: Square root of MSE in the same units as the target variable
- **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted values
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model

### Visualization

All charts are saved as PNG files to ensure cross-platform compatibility. The system sets appropriate fonts to handle Chinese characters in plots.

## Extending the System

To extend the system for other prediction tasks:
1. Modify the data loading section to accommodate new features
2. Adjust the feature processing pipeline as needed
3. Tune Random Forest parameters for optimal performance
4. Add new evaluation metrics or visualization methods

The modular design makes it easy to customize for different regression problems.
