import pickle
import warnings

import pandas as pd
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from asi.evaluation.evaluation import evaluate_model, plot_roc_curve
from asi.evaluation.hyperparameter_tuning import tune_hyperparameters
from asi.models.decision_tree import build_decision_tree
from asi.models.random_forest import build_random_forest

warnings.simplefilter(action='ignore', category=FutureWarning)


def train_model(
    processed_data_path: str, evaluation_metrics_path: str, model_path: str
):
    wandb.login(key="9d8b84a3274d42706670aab35313c6333e638c5b")
    wandb.init(project="my-awesome-project", entity="asi-project-2023")

    df = pd.read_csv(processed_data_path)

    # Kodowanie binarne dla kolumny 'Churn'
    label_encoder = LabelEncoder()
    df["Churn"] = label_encoder.fit_transform(df["Churn"])

    # Inne operacje na danych i budowanie modelu
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Budowanie modelu regresji logistycznej
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    # Ocena modelu LogisticRegression
    evaluate_model(logistic_model, X_test, y_test, evaluation_metrics_path)
    plot_roc_curve(logistic_model, X_test, y_test)

    # Dostosowanie hiperparametrów dla RandomForestClassifier
    random_forest_model = build_random_forest(X_train, y_train)
    param_grid = {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10]}
    tune_hyperparameters(random_forest_model, X_train, y_train, param_grid)

    # Budowanie modelu RandomForestClassifier z optymalnymi hiperparametrami
    random_forest_model_optimal = build_random_forest(
        X_train, y_train, n_estimators=200, max_depth=10
    )

    # Ocena modelu RandomForestClassifier
    evaluate_model(random_forest_model_optimal, X_test, y_test, evaluation_metrics_path)
    plot_roc_curve(random_forest_model_optimal, X_test, y_test)

    # Budowanie modelu DecisionTreeClassifier
    decision_tree_model = build_decision_tree(X_train, y_train)

    # Ocena modelu DecisionTreeClassifier
    evaluate_model(decision_tree_model, X_test, y_test, evaluation_metrics_path)
    plot_roc_curve(decision_tree_model, X_test, y_test)

    pickle.dump(logistic_model, open(model_path, "wb"))

    wandb.finish()

