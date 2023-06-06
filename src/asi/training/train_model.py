import mlflow
import mlflow.sklearn
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import csv

def train_model(
    processed_data_path: str, evaluation_metrics_path: str, model_path: str
):
    # Start MLflow run
    mlflow.start_run(run_name="train_model")

    try:
        # Load the processed data
        data = pd.read_csv(processed_data_path)
        X = data.drop(columns=["Churn"])
        y = data["Churn"].values

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=40, stratify=y
        )

        # Train the model
        knn_model = KNeighborsClassifier(n_neighbors=11)
        knn_model.fit(X_train, y_train)

        # Evaluate the model
        accuracy_knn = knn_model.score(X_test, y_test)
        print("KNN accuracy:", accuracy_knn)

        # Log evaluation metrics to MLflow
        now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        mlflow.log_metric("accuracy", accuracy_knn)
        mlflow.log_param("model", "KNeighborsClassifier")
        mlflow.log_param("timestamp", now)

        # Save the trained model
        with open(evaluation_metrics_path, "a") as f:
            row = [now, 1.0, "KNeighborsClassifier", accuracy_knn]
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(row)
        pickle.dump(knn_model, open(model_path, "wb"))

        # Log the model to MLflow
        mlflow.sklearn.log_model(knn_model, "model")

        # End MLflow run
        mlflow.end_run()

    except Exception as e:
        # Log the exception to MLflow and raise it
        mlflow.log_exception(e)
        raise e
