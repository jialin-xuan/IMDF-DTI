# IMDF-DTI

**An Interpretable Multimodal Dynamic Feature Fusion Framework for Drug-Target
Interaction Prediction.** The implementation combines X-Mask feature weighting,
mask-aware DFGU recalibration, and a RESCAL-Hybrid decoder.

## Setup

Linux and Python 3.10 are recommended.

For other CUDA/PyTorch combinations, follow the official
[PyTorch](https://pytorch.org/get-started/previous-versions/) and
[PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
installation instructions. Record the actual environment with `python -m pip freeze`.

## Data and usage

Place the benchmark text files in `DataSets/`; see [data instructions](DataSets/README.md).
Large datasets, feature caches, checkpoints, and experimental logs are not bundled.

Replace `DrugBank` with `BIOSNAP` or `Davis`. Preparing protein features downloads
ESM-2 weights if unavailable; existing trusted `*-new.pkl` caches can be reused.
`IMDF_TORCH_HUB` optionally sets the pretrained-weight cache directory.
