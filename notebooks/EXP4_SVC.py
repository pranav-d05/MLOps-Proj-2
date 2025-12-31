import re
import string
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore")

# ==========================
# MLflow & DAGsHub Setup
# ==========================
MLFLOW_TRACKING_URI = "https://dagshub.com/pranavdhebe93/MLOps-Proj-2.mlflow"
dagshub.init(repo_owner="pranavdhebe93", repo_name="MLOps-Proj-2", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("FINAL High Accuracy Sentiment Model")

# ==========================
# Text Preprocessing
# ==========================
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\d+', '', text)

    text = " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    )
    return text.strip()

# ==========================
# Load & Prepare Data
# ==========================
def load_data(filepath):
    df = pd.read_csv(filepath)

    df["review"] = df["review"].astype(str).apply(preprocess_text)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"]
    )

    # 🔥 Best TF-IDF config for sentiment
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        min_df=5,
        max_df=0.9,
        sublinear_tf=True,
        norm="l2"
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    return X_train, X_test, y_train, y_test, vectorizer

# ==========================
# Train High-Accuracy Model
# ==========================
def train_high_accuracy_model(X_train, X_test, y_train, y_test):
    param_grid = {
        "C": [0.1, 1, 5, 10]
    }

    with mlflow.start_run():
        mlflow.log_param("vectorizer", "TF-IDF (1,2-gram)")
        mlflow.log_param("model", "LinearSVC")

        grid_search = GridSearchCV(
            LinearSVC(),
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("best_cv_f1", grid_search.best_score_)

        mlflow.sklearn.log_model(best_model, "model")

        print("\n🏆 FINAL BEST MODEL")
        print("Best Params:", grid_search.best_params_)
        print("Metrics:", metrics)

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vectorizer = load_data("notebooks/IMDB.csv")
    train_high_accuracy_model(X_train, X_test, y_train, y_test)
