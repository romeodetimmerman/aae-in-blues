import json
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import shap
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
    X_test = pd.read_csv("../../data/processed/X_test.csv", na_filter=False)
    y_test = pd.read_csv("../../data/processed/y_test.csv")

    # convert categorical features
    cat_features = list(X_test.select_dtypes("object").columns)
    X_test[cat_features] = X_test[cat_features].astype("category")

    # load trained model
    model = CatBoostClassifier()
    model.load_model("../../models/model.cbm")

    # load best threshold
    with open("../../models/best_threshold.json", "r") as f:
        threshold_data = json.load(f)
        best_thr = threshold_data["best_threshold"]

    # generate predictions
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_thr).astype(int)

    print(f"\nusing threshold: {best_thr:.3f}")

    # calculate metrics
    macro_f1 = f1_score(y_test, y_test_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_test_pred, average="weighted")
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_prob)
    pr_auc_macro = average_precision_score(y_test, y_test_prob)

    # print test metrics
    print("\ntest metrics (threshold-tuned):")
    print(f"macro-F1: {macro_f1:.4f}")
    print(f"weighted-F1: {weighted_f1:.4f}")
    print(f"balanced accuracy: {bal_acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc_macro:.4f}")

    # print confusion matrix
    print("\ntest confusion matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)

    # print classification report
    print("\ntest classification report:")
    print(classification_report(y_test, y_test_pred, digits=3))

    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("../../figures/confusion_matrix.png", dpi=600)
    plt.show()

    # calculate and plot shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    
    
    # customize shap bar colors per tds article
    shap.plots.bar(shap_values, show=False)
    
    # define default shap colors and custom colors
    default_pos_color = "#ff0051"
    default_neg_color = "#008bfb"
    positive_color = "#1F77B4"
    negative_color = "#D0E2F2"
    
    # recolor rectangles and texts
    for fc in plt.gcf().get_children():
        # ignore last rectangle (legend background)
        for fcc in fc.get_children()[:-1]:
            if isinstance(fcc, matplotlib.patches.Rectangle):
                face_hex = matplotlib.colors.to_hex(fcc.get_facecolor())
                if face_hex == default_pos_color:
                    fcc.set_facecolor(positive_color)
                elif face_hex == default_neg_color:
                    fcc.set_facecolor(negative_color)
            elif isinstance(fcc, plt.Text):
                text_hex = matplotlib.colors.to_hex(fcc.get_color())
                if text_hex == default_pos_color:
                    fcc.set_color(positive_color)
                elif text_hex == default_neg_color:
                    fcc.set_color(negative_color)
    
    plt.savefig("../../figures/shap_bar_plot.png", dpi=600, bbox_inches="tight")
    plt.show()

    # barplot for 1 feature
    shap.plots.waterfall(shap_values[200], show=False)

    for fc in plt.gcf().get_children():
        for fcc in fc.get_children():
            if (isinstance(fcc, matplotlib.patches.FancyArrow)):
                if (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color):
                    fcc.set_facecolor(positive_color)
                    fcc.set_edgecolor(positive_color)
                elif (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color):
                    fcc.set_color(negative_color)   
            elif (isinstance(fcc, plt.Text)):
                if (matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color):
                    fcc.set_color(positive_color)
                elif (matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color):
                    fcc.set_color(negative_color)
    plt.savefig("../../figures/shap_waterfall_plot.png", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    evaluate_model()
