from __future__ import annotations
import time
import xgboost as xgb

from lab9_config import (
    DATA_DIR,
    EVAL_SUBSET,
    QUANTILE,
    SEED,
    TRAIN_SUBSET,
    VAL_SUBSET,
    XGB_COLSAMPLE_BYTREE,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_SUBSAMPLE,
)
from lab9_utils import (
    ClassificationMetrics,
    compute_classification_metrics,
    optimize_threshold,
    prepare_classification_data,
    print_model_report,
)


def run_xgboost(print_output: bool = True) -> ClassificationMetrics:
    data = prepare_classification_data(
        data_dir=DATA_DIR,
        train_subset=TRAIN_SUBSET,
        val_subset=VAL_SUBSET,
        eval_subset=EVAL_SUBSET,
        quantile=QUANTILE,
        seed=SEED,
    )

    pos = int(data.y_train_cls.sum())
    neg = int(len(data.y_train_cls) - pos)
    scale_pos_weight = float(neg / max(pos, 1))

    model = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )

    train_start = time.time()
    model.fit(
        data.x_train_flat,
        data.y_train_cls,
        eval_set=[(data.x_val_flat, data.y_val_cls)],
        verbose=False,
    )
    train_seconds = time.time() - train_start

    val_scores = model.predict_proba(data.x_val_flat)[:, 1]
    best_threshold, best_val_f1 = optimize_threshold(data.y_val_cls, val_scores)

    infer_start = time.time()
    eval_scores = model.predict_proba(data.x_eval_flat)[:, 1]
    infer_seconds = time.time() - infer_start

    score_metrics = compute_classification_metrics(data.y_eval_cls, eval_scores, threshold=best_threshold)

    metrics = ClassificationMetrics(
        model_name="XGBoost",
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
        notes=(
            f"n_estimators={XGB_N_ESTIMATORS}; max_depth={XGB_MAX_DEPTH}; "
            f"scale_pos_weight={scale_pos_weight:.4f}; best_val_f1={best_val_f1:.4f}"
        ),
    )

    if print_output:
        print_model_report(
            metrics,
            extra_lines=[
                f"data_dir={DATA_DIR}",
                (
                    f"n_estimators={XGB_N_ESTIMATORS} | max_depth={XGB_MAX_DEPTH} | "
                    f"learning_rate={XGB_LEARNING_RATE} | subsample={XGB_SUBSAMPLE} | "
                    f"colsample_bytree={XGB_COLSAMPLE_BYTREE}"
                ),
                f"scale_pos_weight={scale_pos_weight:.4f} | best val F1={best_val_f1:.4f}",
            ],
        )
    return metrics


def main() -> None:
    run_xgboost(print_output=True)


if __name__ == "__main__":
    main()
