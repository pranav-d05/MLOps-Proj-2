# feature engineering
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
from scipy.sparse import save_npz
from src.logger import logging
import pickle


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logging.info('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply Count Vectorizer to the data."""
    try:
        logging.info("Applying TFIdf...")
        vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=5,
        max_df=0.9,
        sublinear_tf=True,
        norm="l2"
    )

        X_train = train_data['review'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['review'].values
        y_test = test_data['sentiment'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)


        pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
        logging.info('TfIdf applied and data transformed')

        return X_train_tfidf, X_test_tfidf, y_train, y_test

    except Exception as e:
        logging.error('Error during TFIdf transformation: %s', e)
        raise

def save_data(df, file_path: str) -> None:
    """Save sparse TF-IDF data to disk."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # df here is a sparse matrix (csr_matrix)
        save_npz(file_path, df)

        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        # max_features = 20

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        X_train, X_test, y_train, y_test = apply_tfidf(
            train_data, test_data, max_features
        )


        save_data(X_train, "./data/processed/X_train_tfidf.npz")
        save_data(X_test, "./data/processed/X_test_tfidf.npz")

        np.save("./data/processed/y_train.npy", y_train)
        np.save("./data/processed/y_test.npy", y_test)

    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()