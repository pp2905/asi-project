from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import csv
import wandb

def train_model(processed_data_path: str, evaluation_metrics_path: str, model_path: str):
    data = pd.read_csv(processed_data_path)

    X = data.drop(columns=["Churn"])
    y = data["Churn"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=40, stratify=y
    )
    # # Definicja siatki parametrów do przeszukania
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 5, 10],
    #     'min_samples_split': [2, 5, 10]
    # }
    # # Inicjalizacja modelu
    # model = RandomForestClassifier()
    #
    # # Użycie GridSearchCV do przeszukania siatki parametrów
    # grid_search = GridSearchCV(model, param_grid, cv=5)
    # grid_search.fit(X_train, y_train)

    # Najlepsze parametry i najlepszy model
    # best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_
    # print(f"Best model {best_model}")

    knn_model = KNeighborsClassifier(n_neighbors=11)
    knn_model.fit(X_train, y_train)
    predicted_y = knn_model.predict(X_test)
    accuracy_knn = knn_model.score(X_test, y_test)
    print("KNN accuracy:", accuracy_knn)
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    wandb.login(key="9d8b84a3274d42706670aab35313c6333e638c5b")
    wandb.init(project='my-awesome-project', entity='asi-project-2023')
    wandb.log({'accuracy': accuracy_knn})
    wandb.finish()

    with open(evaluation_metrics_path, 'a') as f:
        row = [now, 1.0, "KNeighborsClassifier", accuracy_knn]
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(row)

    pickle.dump(knn_model, open(model_path, "wb"))
