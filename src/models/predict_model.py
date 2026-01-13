import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate_model():
    """
    evaluate trained catboost model on test set
    """
    # load test data
    X_test = pd.read_csv("../../data/processed/X_test.csv")
    y_test = pd.read_csv("../../data/processed/y_test.csv")

    # load trained model
    model = CatBoostClassifier()
    model.load_model("../../models/model.cbm")

    # load best threshold
    with open("../../models/best_threshold.json", "r") as f:
        threshold_data = json.load(f)
        tuned_thr = threshold_data["best_threshold"]

    def metrics_for_threshold(thr, y_true, y_prob):
        y_pred = (y_prob >= thr).astype(int)
        return {
            "threshold": thr,
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            "bal_acc": balanced_accuracy_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "pr_auc": average_precision_score(y_true, y_prob),
            "confusion": confusion_matrix(y_true, y_pred),
            "preds": y_pred,
        }

    # generate predictions
    y_test_prob = model.predict_proba(X_test)[:, 1]
    tuned_stats = metrics_for_threshold(tuned_thr, y_test, y_test_prob)

    print(f"\nusing tuned threshold: {tuned_stats['threshold']:.3f}")
    print("\ntest metrics (tuned threshold):")
    print(f"macro-F1: {tuned_stats['macro_f1']:.4f}")
    print(f"weighted-F1: {tuned_stats['weighted_f1']:.4f}")
    print(f"balanced accuracy: {tuned_stats['bal_acc']:.4f}")
    print(f"ROC-AUC: {tuned_stats['roc_auc']:.4f}")
    print(f"PR-AUC: {tuned_stats['pr_auc']:.4f}")

    # print confusion matrix
    print("\ntest confusion matrix (tuned):")
    print(tuned_stats["confusion"])

    # print classification report
    print("\ntest classification report (tuned):")
    print(classification_report(y_test, tuned_stats["preds"], digits=3))

    # plot confusion matrix with tuned threshold
    plt.figure(figsize=(8, 6))
    sns.heatmap(tuned_stats["confusion"], annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("../../figures/model/confusion_matrix.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    evaluate_model()
