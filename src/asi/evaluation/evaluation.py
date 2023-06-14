import csv
from datetime import datetime

import wandb


def evaluate_model(model, X_test, y_test, evaluation_metrics_path):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    accuracy = model.score(X_test, y_test)
    print("Accuracy:", accuracy)

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    wandb.summary["accuracy"] = accuracy
    print(wandb.run.name)
    run_name = f"Confusion_Matrix_{wandb.run.name}"
    wandb.log(
        {
            run_name: wandb.sklearn.plot_confusion_matrix(
                y_test, y_pred, labels=["No", "Yes"]
            )
        }
    )
    wandb.sklearn.plot_roc(y_test, y_probs, labels=["No", "Yes"])

    with open(evaluation_metrics_path, "a") as f:
        row = [now, 1.0, str(model), accuracy]
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(row)
