import csv
import pickle
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def train_model(processed_data_path: str, evaluation_metrics_path: str, model_path: str):
    data = pd.read_csv(processed_data_path)
    X = data.drop(columns=["Churn"])
    y = data["Churn"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=40, stratify=y
    )
    knn_model = KNeighborsClassifier(n_neighbors=11)
    knn_model.fit(X_train, y_train)
    predicted_y = knn_model.predict(X_test)
    accuracy_knn = knn_model.score(X_test, y_test)
    print("KNN accuracy:", accuracy_knn)
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    with open(evaluation_metrics_path, 'a') as f:
        row = [now, 1.0, "KNeighborsClassifier", accuracy_knn]
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(row)
    pickle.dump(knn_model, open(model_path, "wb"))
