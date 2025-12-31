import numpy as np
import pickle
from scipy.sparse import load_npz
from sklearn.svm import LinearSVC
from src.logger import logging
import yaml
import os


def load_data(file_path: str):
    """Load sparse TF-IDF data."""
    try:
        # file_path is used as base path
        X_train = load_npz(os.path.join(file_path, "X_train_tfidf.npz"))
        y_train = np.load(os.path.join(file_path, "y_train.npy"))
        logging.info("Training data loaded from %s", file_path)
        return X_train, y_train
    except Exception as e:
        logging.error("Unexpected error occurred while loading the data: %s", e)
        raise


def train_model(X_train, y_train) -> LinearSVC:
    """Train the SVC model."""
    try:
        clf = LinearSVC(
            C=1.0,
            loss="hinge",
            max_iter=2000
        )
        clf.fit(X_train, y_train)
        logging.info("Model training completed")
        return clf
    except Exception as e:
        logging.error("Error during model training: %s", e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logging.info("Model saved to %s", file_path)
    except Exception as e:
        logging.error("Error occurred while saving the model: %s", e)
        raise


def main():
    try:
        X_train, y_train = load_data("./data/processed")

        clf = train_model(X_train, y_train)

        save_model(clf, "models/model.pkl")
    except Exception as e:
        logging.error("Failed to complete the model building process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
