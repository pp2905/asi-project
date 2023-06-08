import csv
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import roc_curve, roc_auc_score


def evaluate_model(model, X_test, y_test, evaluation_metrics_path):
    # Ocena modelu na zbiorze testowym
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    accuracy = model.score(X_test, y_test)
    print("Accuracy:", accuracy)

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # wandb.log({"accuracy": accuracy})
    wandb.summary["accuracy"] = accuracy
    print(wandb.run.name)
    run_name = f"Confusion_Matrix_{wandb.run.name}"
    wandb.log({run_name:wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels=["No","Yes"])})
    # y_test_arr = np.array(y_test)
    # y_probs_arr = np.array(y_probs)
    wandb.sklearn.plot_roc(y_test, y_probs, labels=["No","Yes"])

    with open(evaluation_metrics_path, "a") as f:
        row = [now, 1.0, str(model), accuracy]
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(row)


# def plot_roc_curve(model, X_test, y_test):
#     # Tworzenie krzywej ROC
#     y_probs = model.predict_proba(X_test)[:, 1]
#     fpr, tpr, thresholds = roc_curve(y_test, y_probs)
#     auc = roc_auc_score(y_test, y_probs)
#
#     # Rysowanie wykresu
#     plt.figure()
#     plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
#     plt.plot([0, 1], [0, 1], "k--")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("Receiver Operating Characteristic")
#     plt.legend(loc="lower right")
#     plt.savefig(f"data/evaluation/{str(model)}.png")
