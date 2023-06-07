import csv
from datetime import datetime

import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import roc_curve, roc_auc_score


def evaluate_model(model, X_test, y_test, evaluation_metrics_path):
    # Ocena modelu na zbiorze testowym
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print("Accuracy:", accuracy)

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    wandb.log({"accuracy": accuracy})

    with open(evaluation_metrics_path, "a") as f:
        row = [now, 1.0, str(model), accuracy]
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(row)


def plot_roc_curve(model, X_test, y_test):
    # Tworzenie krzywej ROC
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)

    # Rysowanie wykresu
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(f"data/evaluation/{str(model)}.png")
