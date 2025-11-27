import pandas as pd
import numpy as np
import pydot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

# Set font to resolve Chinese display issues in plots
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TempPre:
    def __init__(self, path):
        """
        Initialize the temperature prediction model
        :param path: Path to the CSV data file
        """
        self.feature_importance = None
        self.rf_model = None
        self.path = path
        self.features = pd.read_csv(self.path)
        year = self.features['year']
        month = self.features['month']
        day = self.features['day']
        dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(year, month, day)]
        self.dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
        self.features = pd.get_dummies(self.features)
        self.labels = np.array(self.features['actual'])
        self.features = self.features.drop('actual', axis=1)
        self.features_list = list(self.features.columns)
        self.features = np.array(self.features)
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(
            self.features, self.labels, test_size=0.25, random_state=42)

    def show_data_info(self):
        """
        Display basic dataset information
        """
        print("Dataset Information:")
        print(f"Total samples: {len(self.features)}")
        print(f"Number of features: {len(self.features_list)}")
        print(f"Feature list: {self.features_list}")
        print(f"Label range: [{self.labels.min()}, {self.labels.max()}]")
        print("-" * 50)

    def random_forest(self, max_depth, n_estimators):
        """
        Train Random Forest model
        :param max_depth: Maximum depth of the tree
        :param n_estimators: Number of trees in the forest
        """
        self.rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        self.rf_model.fit(self.train_features, self.train_labels)

    def MAPE(self):
        """
        Calculate Mean Absolute Percentage Error
        :return: MAPE value
        """
        predictions = self.rf_model.predict(self.test_features)
        error = np.abs(predictions - self.test_labels)
        return np.mean(100 * (error / self.test_labels))

    def evaluate_model(self):
        """
        Comprehensive model evaluation
        :return: Dictionary containing various evaluation metrics
        """
        predictions = self.rf_model.predict(self.test_features)

        # Calculate various evaluation metrics
        mape = self.MAPE()
        mse = mean_squared_error(self.test_labels, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.test_labels, predictions)
        r2 = r2_score(self.test_labels, predictions)

        print("Model Evaluation Results:")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Coefficient of Determination (R²): {r2:.2f}")
        print("-" * 50)

        return {"MAPE": mape, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    def render_tree(self, which_tree):
        """
        Visualize a single decision tree
        :param which_tree: Index of the tree to visualize
        :return: Boolean indicating success or failure
        """
        try:
            tree = self.rf_model.estimators_[which_tree]
            export_graphviz(tree, out_file="tree.dot", feature_names=self.features_list, rounded=True, precision=1)
            (graph,) = pydot.graph_from_dot_file('tree.dot')
            graph.write_png('tree.png')
            return True
        except Exception as e:
            print(f"Failed to generate decision tree image: {e}")
            print("Please ensure Graphviz is installed and added to system PATH")
            return False

    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        :return: List of tuples (feature_name, importance)
        """
        importance = list(self.rf_model.feature_importances_)
        feature_importance = [(feature, round(importance, 2)) for feature, importance in
                              zip(self.features_list, importance)]
        feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
        self.feature_importance = feature_importance
        return self.feature_importance

    def plot_feature_importance(self, top_n=10):
        """
        Plot feature importance chart
        :param top_n: Number of top features to display
        """
        if self.feature_importance is None:
            self.get_feature_importance()

        # Get top N important features
        top_features = self.feature_importance[:top_n]
        features, importances = zip(*top_features)

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(features)), importances)
        plt.xticks(range(len(features)), features, rotation=45)
        plt.title(f'Top {top_n} Feature Importances')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print(f"Feature importance chart saved as feature_importance.png")

    def plot_predictions(self):
        """
        Plot predicted vs actual values
        """
        predictions = self.rf_model.predict(self.test_features)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.test_labels, predictions, alpha=0.5)
        plt.plot([self.test_labels.min(), self.test_labels.max()],
                 [self.test_labels.min(), self.test_labels.max()], 'r--', lw=2)
        plt.xlabel('Actual Temperature')
        plt.ylabel('Predicted Temperature')
        plt.title('Predicted vs Actual Values')
        plt.tight_layout()
        plt.savefig('predictions.png')
        plt.close()
        print("Prediction comparison chart saved as predictions.png")

    def predict(self, data=None):
        """
        Make predictions using the trained model
        :param data: Data to predict on. If None, uses test data
        :return: Predictions
        """
        if self.rf_model is None:
            raise ValueError("Model not trained yet. Please call random_forest method first")

        if data is not None:
            return self.rf_model.predict(data)
        else:
            return self.rf_model.predict(self.test_features)

    def predict_external_data(self, external_path):
        """
        Predict external data using the trained model
        :param external_path: Path to the external CSV file
        :return: Predictions
        """
        if self.rf_model is None:
            raise ValueError("Model not trained yet. Please call random_forest method first")

        # Read external data
        external_data = pd.read_csv(external_path)

        # Process external data to ensure column consistency
        external_data_processed = pd.get_dummies(external_data)

        # Get actual temperatures as labels (if they exist)
        if 'actual' in external_data_processed.columns:
            external_labels = np.array(external_data_processed['actual'])
            external_data_processed = external_data_processed.drop('actual', axis=1)
        else:
            external_labels = None

        # Ensure external data has the same feature columns
        # For columns present in training data but not in external data, add and set to 0
        for col in self.features_list:
            if col not in external_data_processed.columns:
                external_data_processed[col] = 0

        # For columns present in external data but not in training data, remove them
        external_data_processed = external_data_processed.reindex(columns=self.features_list, fill_value=0)

        # Make predictions
        external_predictions = self.rf_model.predict(external_data_processed)

        # If actual labels exist, calculate errors
        if external_labels is not None:
            mape = np.mean(np.abs((external_labels - external_predictions) / external_labels)) * 100
            mse = mean_squared_error(external_labels, external_predictions)
            print(f"\nExternal Data Prediction Results:")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            print(f"Mean Squared Error (MSE): {mse:.2f}")

        # Display prediction results
        print(f"\nPrediction Results Preview (First 10 entries):")
        for i in range(min(10, len(external_predictions))):
            actual_str = f" (Actual: {external_labels[i]:.1f})" if external_labels is not None else ""
            print(f"Entry {i+1} predicted temperature: {external_predictions[i]:.1f}°C{actual_str}")

        return external_predictions


def main():
    """
    Main function to demonstrate the temperature prediction system
    """
    # Train model using temps_extended.csv
    print("Training model using temps_extended.csv...")
    path = "temps_extended.csv"
    # Create temperature predictor object
    temp_predictor = TempPre(path)

    # Display data information
    temp_predictor.show_data_info()

    # Train Random Forest model
    temp_predictor.random_forest(max_depth=15, n_estimators=100)

    # Evaluate model comprehensively
    temp_predictor.evaluate_model()

    # Display feature importance
    feature_importance = temp_predictor.get_feature_importance()
    print("\nFeature Importance Ranking:")
    for feature, importance in feature_importance[:10]:  # Show top 10
        print(f'{feature}: {importance}')
    print("-" * 50)

    # Plot feature importance
    temp_predictor.plot_feature_importance()

    # Plot prediction results
    temp_predictor.plot_predictions()

    # Visualize a single decision tree
    success = temp_predictor.render_tree(0)
    if success:
        print("\nDecision tree saved as tree.png")
    else:
        print("\nSkipping decision tree visualization")

    # Predict data in temps.csv using the trained model
    print("\nPredicting data in temps.csv using the trained model...")
    predictions = temp_predictor.predict_external_data("temps.csv")


if __name__ == "__main__":
    main()