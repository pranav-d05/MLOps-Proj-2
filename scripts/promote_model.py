# promote model using aliases (NEW MLflow registry)

import os
import mlflow


def promote_model():
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
        "https://dagshub.com/pranavdhebe93/MLOps-Proj-2.mlflow"
    )

    client = mlflow.MlflowClient()
    model_name = "Senti-Classifier"

    # Get all versions
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    if not versions:
        raise RuntimeError("No model versions found")

    # Pick latest version numerically
    latest_version = max(
        versions, key=lambda v: int(v.version)
    ).version

    # Assign alias "production"
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=latest_version
    )

    print(
        f"Model '{model_name}' version {latest_version} "
        f"assigned alias 'production'"
    )


if __name__ == "__main__":
    promote_model()
