import pickle
import warnings

import pandas as pd
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from asi.evaluation.evaluation import evaluate_model
from asi.evaluation.hyperparameter_tuning import tune_hyperparameters
from asi.models.decision_tree import build_decision_tree
from asi.models.random_forest import build_random_forest

warnings.simplefilter(action="ignore", category=FutureWarning)
wandb.login(key="9d8b84a3274d42706670aab35313c6333e638c5b")


def make_dataset(processed_data_path: str):
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
    return X_train, X_test, y_train, y_test


def run_sweep_random_forest(
    X_train, X_test, y_train, y_test, evaluation_metrics_path: str, model_path
):
    def evaluate_random_forest():
        wandb.init(tags=["RandomForestModel"])
        random_forest_model = build_random_forest(
            X_train,
            y_train,
            n_estimators=wandb.config["n_estimators"],
            max_depth=wandb.config["max_depth"],
        )

        # Model evaluation RandomForestClassifier
        evaluate_model(random_forest_model, X_test, y_test, evaluation_metrics_path)
        pickle.dump(
            random_forest_model, open(model_path["random_forest_model_path"], "wb")
        )
        wandb_config_dict = {
            "n_estimators": wandb.config["n_estimators"],
            "max_depth": wandb.config["max_depth"],
        }
        experiment_artifact = wandb.Artifact(
            "random_forest_model",
            type="model",
            description="Random Forest Model",
            metadata=wandb_config_dict,
        )
        experiment_artifact.add_file(model_path["random_forest_model_path"])
        wandb.log_artifact(experiment_artifact)
        wandb.run.link_artifact(
            experiment_artifact, "asi-project-2023/my-awesome-project/RandomForestModel"
        )

    sweep_config = {
        "method": "grid",
        "metric": {"goal": "maximize", "name": "accuracy"},
        "parameters": {
            "n_estimators": {"values": [100, 200, 300]},
            "max_depth": {"values": [None, 5, 10]},
        },
    }
    sweep_id = wandb.sweep(
        project="my-awesome-project", entity="asi-project-2023", sweep=sweep_config
    )
    wandb.agent(sweep_id, function=evaluate_random_forest, count=100)


def train_model(
    processed_data_path: str, evaluation_metrics_path: str, model_path: str
):
    X_train, X_test, y_train, y_test = make_dataset(processed_data_path)
    run_sweep_random_forest(
        X_train, X_test, y_train, y_test, evaluation_metrics_path, model_path
    )
    wandb.init(
        project="my-awesome-project", entity="asi-project-2023", tags=["LogisticModel"]
    )

    # Building a logistic regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)

    # Assessment of the LogisticRegression model
    evaluate_model(logistic_model, X_test, y_test, evaluation_metrics_path)
    # plot_roc_curve(logistic_model, X_test, y_test)

    wandb.init(
        project="my-awesome-project",
        entity="asi-project-2023",
        tags=["DecisionTreeModel"],
    )

    # Building a DecisionTreeClassifier model
    decision_tree_model = build_decision_tree(X_train, y_train)

    # Assessment of the DecisionTreeClassifier model
    evaluate_model(decision_tree_model, X_test, y_test, evaluation_metrics_path)
    # plot_roc_curve(decision_tree_model, X_test, y_test)

    pickle.dump(logistic_model, open(model_path["logistic_model_path"], "wb"))
    pickle.dump(decision_tree_model, open(model_path["decision_tree_model_path"], "wb"))

    # artifact decision tree
    wandb_config_dict = {
        "n_estimators": wandb.config["n_estimators"],
        "max_depth": wandb.config["max_depth"],
    }
    experiment_artifact = wandb.Artifact(
        "decision_tree_model",
        type="model",
        description="Decision Tree Model",
        metadata=wandb_config_dict,
    )
    experiment_artifact.add_file(model_path["decision_tree_model_path"])
    wandb.log_artifact(experiment_artifact)
    wandb.run.link_artifact(
        experiment_artifact, "asi-project-2023/my-awesome-project/DecisionTreeModel"
    )

    # artifact logistic model
    wandb_config_dict = {
        "n_estimators": wandb.config["n_estimators"],
        "max_depth": wandb.config["max_depth"],
    }
    experiment_artifact = wandb.Artifact(
        "logistic_model",
        type="model",
        description="Logistic Regression Model",
        metadata=wandb_config_dict,
    )
    experiment_artifact.add_file(model_path["logistic_model_path"])
    wandb.log_artifact(experiment_artifact)
    wandb.run.link_artifact(
        experiment_artifact, "asi-project-2023/my-awesome-project/LogisticModel"
    )

    wandb.finish()
