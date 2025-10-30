import json
import numpy as np
import optuna
import pandas as pd
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score


def macro_f1_at_best_threshold(y_true, y_prob):
    """
    find best classification threshold based on macro f1

    params
    ------
    y_true: array-like
        true labels
    y_prob: array-like
        predicted probabilities

    returns
    -------
    best_f1: float
        best macro f1 score
    best_threshold: float
        threshold that achieves best f1
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, average="macro"))
    best_idx = int(np.argmax(f1s))
    return f1s[best_idx], thresholds[best_idx]


def train_model():
    """
    train catboost model with optuna hyperparameter optimization
    """
    # set seed
    seed = 42

    # load data
    X_train = pd.read_csv("../../data/processed/X_train.csv", na_filter=False)
    X_val = pd.read_csv("../../data/processed/X_val.csv", na_filter=False)
    y_train = pd.read_csv("../../data/processed/y_train.csv")
    y_val = pd.read_csv("../../data/processed/y_val.csv")

    # list categorical features
    cat_features = list(X_train.select_dtypes("object").columns)
    for X in (X_train, X_val):
        X[cat_features] = X[cat_features].astype("category")

    # create training and validation pools
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    # optimization function
    def catboost_objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "random_seed": seed,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "early_stopping_rounds": 200,
            "allow_writing_files": False,
        }

        # test class weighting strategies
        class_weight_strategy = trial.suggest_categorical(
            "auto_class_weights", [None, "Balanced", "SqrtBalanced"]
        )
        if class_weight_strategy is not None:
            params["auto_class_weights"] = class_weight_strategy

        # fit model
        model = CatBoostClassifier(**params, verbose=False)
        model.fit(train_pool, eval_set=val_pool, verbose=False)

        # get probabilities for threshold tuning
        y_val_prob = model.predict_proba(X_val)[:, 1]
        macro_f1, thr = macro_f1_at_best_threshold(y_val, y_val_prob)

        # store actual best iteration from early stopping
        best_iteration = model.get_best_iteration()
        trial.set_user_attr("best_iteration", int(best_iteration))

        # report threshold to recover it later
        trial.set_user_attr("best_threshold", float(thr))

        # log secondary metric
        trial.set_user_attr(
            "val_bal_acc",
            float(balanced_accuracy_score(y_val, (y_val_prob >= thr).astype(int))),
        )
        return macro_f1

    # run study over 100 trials
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(catboost_objective, n_trials=100)

    # get best parameters, iteration count, and threshold
    best_params = study.best_trial.params
    best_iteration = study.best_trial.user_attrs["best_iteration"]
    best_class_weight = best_params.get("auto_class_weights")

    # print best trial
    print("\ncatboost - best trial:")
    print(f"\tbest macro-F1: {study.best_value:.4f}")
    print(f"\tbest iteration: {best_iteration}")
    print("\tbest params:")
    for k, v in best_params.items():
        print(f"\t\t{k}: {v}")

    # get top k trials with same class weighting strategy for robust threshold
    same_weight_trials = [
        t
        for t in study.trials
        if t.params.get("auto_class_weights") == best_class_weight
        and t.state == optuna.trial.TrialState.COMPLETE
    ]
    same_weight_trials.sort(key=lambda t: t.value, reverse=True)
    top_k = min(10, len(same_weight_trials))
    top_thresholds = [
        t.user_attrs["best_threshold"] for t in same_weight_trials[:top_k]
    ]
    best_thr = np.mean(top_thresholds)

    print(
        f"\tthreshold (avg of top {top_k} trials w/ same class weights): {best_thr:.3f}"
    )
    print(f"\tthreshold std: {np.std(top_thresholds):.3f}")

    # train final model on train+val with tuned params and best iteration count
    X_tr_final = pd.concat([X_train, X_val], axis=0)
    y_tr_final = pd.concat([y_train, y_val], axis=0)
    tr_final_pool = Pool(X_tr_final, y_tr_final, cat_features=cat_features)

    # use exact iteration count from early stopping
    final_params = {
        **{k: v for k, v in best_params.items() if k != "iterations"},
        "iterations": best_iteration,
        "random_seed": seed,
        "loss_function": "Logloss",
        "allow_writing_files": False,
        "verbose": False,
    }

    # add class weighting if it was used in best trial
    if best_params.get("auto_class_weights") is not None:
        final_params["auto_class_weights"] = best_params["auto_class_weights"]

    # train for exact number of iterations (no early stopping needed)
    final_model = CatBoostClassifier(**final_params)
    final_model.fit(tr_final_pool, verbose=False)

    # save trained model
    final_model.save_model(fname="../../models/model.json", format="json")
    final_model.save_model(fname="../../models/model.cbm", format="cbm")

    # save averaged threshold from top k trials
    with open("../../models/best_threshold.json", "w") as f:
        json.dump({"best_threshold": best_thr}, f)

    print("\nmodel training complete")
    print("saved model to: ../../models/model.cbm")
    print("saved threshold to: ../../models/best_threshold.json")
    print(f"using threshold averaged from top {top_k} trials with same class weighting")


if __name__ == "__main__":
    train_model()
