# Lab 9 code for the ClimSim quickstart subset

## How the model work in the project (Transformer)

This model processes each sample by treating it as a sequence, utilizing a self-attention mechanism to learn the interactions between various positions within the sequence, and subsequently predicts a class label for the entire sample. In this document, the model is referred to as the "ClimateColumnTransformer," and its associated training workflow encompasses data loading, network training, selection of the optimal validation threshold, and the reporting of final classification metrics.

## How the model work in the project (Random Forest)

This model learns an ensemble of decision trees based on features within a planar space, which fundamentally distinguishes it from Transformers. While Transformers learn sequential relationships through self-attention mechanisms, Random Forests learn nonlinear decision rules by aggregating numerous tree-based partitions of the feature space.

## How the model work in the project (XGBoost)

It loads preprocessed classification data, utilizes the `scale_pos_weight` parameter to address class imbalance, and trains a boosted tree classifier. Subsequently, it selects the optimal classification threshold based on the F1 score on the validation set, evaluates the model on an independent test set, and records the time required for both training and inference. XGBoost is a tabular data model based on boosting algorithms, whereas the Transformer is a sequence model that incorporates self-attention mechanisms.


The code assumes the same standard quickstart file names used in the current project:

- `train_input.npy`
- `train_target.npy`
- `val_input.npy`
- `val_target.npy`

## What this code does

The original project is regression-oriented, but Lab 9 asks for **accuracy** and **F1**.
To satisfy that requirement with the same dataset, these scripts create a binary label:

if the example is in the top quantile of training-set precipitation, change to 1. Otherwise, change to 0. 

The label is derived from the last 8 target variables using:

- `PRECSC` (snow rate)
- `PRECC` (rain rate)

and uses the rule:

```python
heavy_precip = (PRECSC + PRECC) >= training_quantile_threshold
```

Default threshold is the 90th percentile (`QUANTILE = 0.90`).

## Files

- `lab9_config.py` – hardcoded settings for paths, subsets, seeds, and model hyperparameters
- `lab9_utils.py` – shared preprocessing, label creation, metrics, and console-print helpers
- `train_transformer_classifier.py` – sequence model using a small Transformer encoder
- `train_rf_classifier.py` – Random Forest baseline
- `train_xgb_classifier.py` – XGBoost baseline


## Dependencies

These scripts use:

- Python 3.10+
- NumPy
- PyTorch
- scikit-learn
- XGBoost

```bash
pip install numpy torch scikit-learn xgboost
```

## How to use

Run the Transformer:

```bash
python train_transformer_classifier.py
```

Run the Random Forest:

```bash
python train_rf_classifier.py
```

Run the XGBoost:
```bash
python train_xgb_classifier.py
```
