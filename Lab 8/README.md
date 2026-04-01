# Lab 8 code materials

## What is included

- `models.py` - small Conv1D model
- `data_utils.py` - data loading, standardization, and input reshaping
- `train_cnn.py` - training/evaluation script
- `requirements.txt` - minimal Python dependencies

## Dateset and Code Using

Use the official public resources instead:

- Official ClimSim repository: `https://github.com/leap-stc/ClimSim`
- Dataset's link: `https://leap-stc.github.io/ClimSim/dataset.html` 
- Quickstart's link (CNN example): `https://github.com/leap-stc/ClimSim/blob/main/climsim_utils/data_utils.py`
- Official ClimSim quickstart / dataset information links are listed there.
- The official version of ClimSim also includes a CNN example. This serves as a reference for the current project.
```text
data/
  train_input.npy
  train_target.npy
  val_input.npy
  val_target.npy
```

## How to use:

```bash
python train_cnn.py --data-dir data --epochs 5 --train-subset 4096 --val-subset 1024
```

## What the script does

- reads 124-dim quickstart-style inputs and 128-dim targets,
- reshapes the flat input into a `(channels=6, levels=60)` tensor,
- trains a compact Conv1D regressor,
- reports validation MAE, RMSE, and R2,


