from __future__ import annotations

import time

from sklearn.ensemble import RandomForestClassifier

from lab9_config import (
    DATA_DIR,
    EVAL_SUBSET,
    QUANTILE,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_N_ESTIMATORS,
    SEED,
    TRAIN_SUBSET,
    VAL_SUBSET,
)
from lab9_utils import (
    ClassificationMetrics,
    compute_classification_metrics,
    optimize_threshold,
    prepare_classification_data,
    print_model_report,
)


def run_random_forest(print_output: bool = True) -> ClassificationMetrics:
    data = prepare_classification_data(
        data_dir=DATA_DIR,
        train_subset=TRAIN_SUBSET,
        val_subset=VAL_SUBSET,
        eval_subset=EVAL_SUBSET,
        quantile=QUANTILE,
        seed=SEED,
    )

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=None if RF_MAX_DEPTH <= 0 else RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )

    train_start = time.time()
    model.fit(data.x_train_flat, data.y_train_cls)
    train_seconds = time.time() - train_start

    val_scores = model.predict_proba(data.x_val_flat)[:, 1]
    best_threshold, best_val_f1 = optimize_threshold(data.y_val_cls, val_scores)

    infer_start = time.time()
    eval_scores = model.predict_proba(data.x_eval_flat)[:, 1]
    infer_seconds = time.time() - infer_start

    score_metrics = compute_classification_metrics(data.y_eval_cls, eval_scores, threshold=best_threshold)

    metrics = ClassificationMetrics(
        model_name="Random Forest",
        eval_split=data.eval_split_name,
        accuracy=score_metrics["accuracy"],
        f1=score_metrics["f1"],
        precision=score_metrics["precision"],
        recall=score_metrics["recall"],
        balanced_accuracy=score_metrics["balanced_accuracy"],
        positive_rate_train=float(data.y_train_cls.mean()),
        positive_rate_eval=float(data.y_eval_cls.mean()),
        score_threshold=best_threshold,
        train_seconds=float(train_seconds),
        inference_seconds=float(infer_seconds),
        train_subset=int(data.x_train_flat.shape[0]),
        val_subset=int(data.x_val_flat.shape[0]),
        eval_subset=int(data.x_eval_flat.shape[0]),
        quantile=float(QUANTILE),
        seed=int(SEED),
        notes=f"n_estimators={RF_N_ESTIMATORS}; max_depth={RF_MAX_DEPTH}; best_val_f1={best_val_f1:.4f}",
    )

    if print_output:
        print_model_report(
            metrics,
            extra_lines=[
                f"data_dir={DATA_DIR}",
                f"n_estimators={RF_N_ESTIMATORS} | max_depth={RF_MAX_DEPTH} | min_samples_leaf={RF_MIN_SAMPLES_LEAF}",
                f"best val F1={best_val_f1:.4f}",
            ],
        )
    return metrics


def main() -> None:
    run_random_forest(print_output=True)


if __name__ == "__main__":
    main()
