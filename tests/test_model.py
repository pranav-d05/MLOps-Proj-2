# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "pranavdhebe93"
        repo_name = "MLOps-Proj-2"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "Senti-Classifier"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load the vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        cls.X_holdout = load_npz('data/processed/X_test_tfidf.npz')
        cls.y_holdout = np.load('data/processed/y_test.npy')

        cls.X_holdout_df = pd.DataFrame(
            cls.X_holdout.toarray(),
            columns=[str(i) for i in range(cls.X_holdout.shape[1])]
        )

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])

        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])]
        )

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)

        # Verify the input shape
        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out())
        )

        # Verify the output shape
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        # Predict using the new model
        y_pred_new = self.new_model.predict(self.X_holdout_df)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(self.y_holdout, y_pred_new)
        precision_new = precision_score(self.y_holdout, y_pred_new)
        recall_new = recall_score(self.y_holdout, y_pred_new)
        f1_new = f1_score(self.y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy)
        self.assertGreaterEqual(precision_new, expected_precision)
        self.assertGreaterEqual(recall_new, expected_recall)
        self.assertGreaterEqual(f1_new, expected_f1)


if __name__ == "__main__":
    unittest.main()
