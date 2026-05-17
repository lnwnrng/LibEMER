# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LibEMER is a benchmark library for EEG-based emotion recognition. It implements multiple deep learning models that classify emotions from EEG signals, with optional multimodal support (peripheral physiological signals: eye tracking, EMG, GSR, respiration, BVP, temperature).

Supported datasets: SEED, SEED-IV, SEED-V, DEAP, DREAMER, HCI.

## Running experiments

Each model has its own `*_train.py` script at the repo root. Run with argparse:

```
python DCCA_train.py -model DCCA -dataset seed_de_lds -dataset_path /path/to/SEED \
  -sessions 1 2 -sample_length 1 -stride 1 -batch_size 100 -epochs 200 -lr 1e-4
```

Use presets to avoid specifying all parameters manually:

```
python DCCA_train.py -model DCCA -setting seed_multimodal_sub_dependent_train_val_test_setting \
  -dataset_path /path/to/SEED -batch_size 32 -seed 2025 -epochs 100 -lr 1e-4 -onehot
```

Available presets are defined as the keys of `preset_setting` in `config/setting.py`. Key presets include combinations of:
- Dataset: `seed`, `seedv`, `deap`
- Modality: single-modal (default) or `multimodal`
- Experiment mode: `sub_dependent`, `sub_independent`
- Split type: `train_val_test`, `kfold`, `front_back`, `leave_one_out`

### Key arguments

| Flag | Description |
|------|-------------|
| `-model` | Model name matching a key in `models/Models.py` `Model` dict |
| `-dataset` | Dataset name (see `available_dataset` in `data_utils/load_data.py`) |
| `-dataset_path` | Path to raw dataset directory |
| `-setting` | Preset config name from `preset_setting` dict |
| `-use_multimodal` | Enable multimodal (EEG + bio signals) |
| `-feature_type` | Feature extraction method: `de`, `de_lds`, `psd`, `psd_lds`, `cwt`, etc. |
| `-experiment_mode` | `sub_dependent`, `sub_independent`, `cross_session` |
| `-split_type` | `kfold`, `leave_one_out`, `front_back`, `train_val_test` |
| `-label_used` | Emotion dimension(s): `valence`, `arousal`, `dominance`, `liking` |
| `-sessions` | Which sessions to use (1-indexed) |
| `-bounds` | Binary classification thresholds for continuous labels: `[low, high]` |
| `-batch_size` | Batch size (default 128) |
| `-epochs` | Training epochs (default 40) |
| `-lr` | Learning rate (default 0.001) |
| `-seed` | Random seed (default 42) |
| `-device` | `cuda` or `cpu` |

## Python environment

Always activate the project virtual environment before running Python commands:

```
source ~/data/hzx/myenv/.venv/bin/activate
```

This environment has all required dependencies (PyTorch, numpy, scipy, sklearn, mne, pywt, tqdm). There are no requirements.txt, setup.py, or tests.

## Architecture

### Data flow

```
args → Setting (config/setting.py)
     → get_data() (data_utils/load_data.py) → preprocess/feature extraction → label processing
     → merge_to_part() (data_utils/split.py) — reshapes from (session,subject,trail,sample) to (partition,sample)
     → get_split_index() — generates train/val/test indices per round
     → index_to_data() — materializes train/val/test arrays from indices
     → normalize() → train loop
```

### Three experiment modes (set via `experiment_mode`)

- **sub_dependent**: Within-subject — split each subject's own trials into train/val/test. `merge_to_part` produces one partition per subject.
- **sub_independent**: Cross-subject — some subjects form training, others form test. `merge_to_part` produces one partition per subject, and `get_split_index` assigns entire subjects to splits.
- **cross_session**: Train on some sessions, test on other sessions.

In multimodal mode, parallel functions (`merge_to_part_multimodal`, `index_to_data_multimodal`) handle both EEG and bio data simultaneously.

### Directory structure

- **`models/`** — PyTorch `nn.Module` definitions, one file per model. `Models.py` is the registry mapping model names to classes.
- **`Trainer/`** — Training/evaluation loops, one file per model. Most define their own `train()` and `evaluate()` because models differ in forward pass signatures and optimizer counts. Only `Trainer/training.py` is a generic trainer used by some simpler models.
- **`config/setting.py`** — `Setting` dataclass holding all preprocessing and split parameters. Preset functions build `Setting` objects for specific experiment protocols.
- **`data_utils/load_data.py`** — Dataset readers that parse raw files into uniform `(session, subject, trail, channel, data)` format. `get_data()` orchestrates loading + preprocessing.
- **`data_utils/preprocess.py`** — Signal processing: bandpass filtering, EOG artifact removal (PCA), feature extraction (DE, PSD, CWT, power spectrum), LDS smoothing, segment sliding windows, label encoding.
- **`data_utils/split.py`** — Partition merging, split index generation (KFold, LeaveOneOut, front_back, train_val_test), index-to-data materialization.
- **`utils/metric.py`** — `Metric` (overall accuracy/F1/kappa) and `SubMetric` (per-subject with mean/std across subjects).
- **`utils/experiment_logger.py`** — Tabular console logging and CSV history for each experiment.
- **`utils/args.py`** — Full argument parser with all CLI flags.
- **`utils/store.py`** — Checkpoint save/load, result log file naming, output directory structure.
- **`utils/utils.py`** — `setup_seed()` for reproducibility, `state_log()`/`result_log()`/`sub_result_log()` for experiment-level logging.
- **`data_utils/constants/`** — Channel name lists, 2D grid locations, adjacency matrices for DEAP and SEED datasets.

### Available models

| Model | Source | Description |
|-------|--------|-------------|
| DCCA | `models/DCCA.py` | Deep CCA for multimodal fusion |
| DCCA_AM | `models/DCCA_AM.py` | DCCA with attention mechanism |
| BimodalLSTM | `models/BimodalLSTM.py` | LSTM-based bimodal fusion |
| CRNN | `models/CRNN.py` | Convolutional + recurrent network |
| BDDAE | `models/BDDAE.py` | Bimodal deep denoising autoencoder |
| MCAF | `models/MCAF.py` | Multi-channel attention fusion |
| CMCM | `models/CMCM.py` | Cross-modal correlation matching |
| CFDA_CSF | `models/CFDA_CSF.py` | Cross-subject feature distribution alignment |
| HetEmotionNet | `models/HetEmotionNet.py` | Heterogeneous graph neural network |
| G2G | `models/G2G.py` | Graph-to-graph learning |
| Het_Model | `models/Het.py` | Heterogeneous model variant |

### Multimodal mode specifics

When `-use_multimodal` is set, `get_data()` returns both EEG and bio data arrays. The bio data comes from peripheral physiological signals (eye tracking, EMG, GSR, respiration, BVP, temperature). Bio signals undergo their own frequency-band feature extraction via `bio_extraction()`. EEG and bio data are segmented separately with independent `sample_length`/`stride` and `bio_length`/`bio_stride` parameters. Models receive both `(eeg, bio)` tensors in their forward pass.
