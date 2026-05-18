# LibEMER

EEG-based emotion recognition benchmark library. Implements 10 deep learning models with multimodal support (EEG + peripheral physiological signals).

**Datasets:** SEED (3-class), SEED-V (5-class), SEED-IV (4-class), DEAP (valence/arousal binary).

## Reproduction Results — Subject-Independent (SI)

All experiments: seed=2026, subject-independent, multimodal, session 1 only. Metrics reported as test-set mean (subject-wise std). Acc: accuracy, F1: macro-F1.

### SEED (3-class)

| Model | Acc | F1 | Acc_std | F1_std |
|-------|-----|------|---------|--------|
| DCCA | 0.5624 | 0.4990 | 0.0949 | 0.1104 |
| DCCA_AM | 0.6865 | 0.6108 | 0.0823 | 0.0731 |
| BimodalLSTM | 0.4166 | 0.2800 | 0.1300 | 0.1125 |
| CRNN | **0.7138** | **0.6420** | 0.0823 | 0.1136 |
| BDDAE | 0.6211 | 0.5838 | 0.0101 | 0.0416 |
| MCAF | 0.3450 | 0.2015 | 0.0277 | 0.0379 |
| CMCM | 0.6909 | 0.6681 | 0.2370 | 0.1876 |
| CFDA_CSF | 0.6300 | 0.5166 | 0.0630 | 0.0400 |
| G2G | 0.4685 | 0.3841 | 0.2024 | 0.2113 |

### SEED-V (5-class)

| Model | Acc | F1 | Acc_std | F1_std |
|-------|-----|------|---------|--------|
| DCCA | 0.1787 | 0.1076 | 0.0764 | 0.0499 |
| DCCA_AM | 0.2305 | 0.1066 | 0.0583 | 0.0547 |
| BimodalLSTM | 0.2679 | 0.1326 | 0.0243 | 0.0366 |
| CRNN | 0.2457 | 0.1609 | 0.1385 | 0.0883 |
| BDDAE | 0.2810 | 0.1699 | 0.0971 | 0.0834 |
| MCAF | 0.2203 | 0.1810 | 0.0642 | 0.0821 |
| CMCM | 0.2906 | 0.2234 | 0.0948 | 0.0660 |
| CFDA_CSF | 0.3994 | 0.3350 | 0.0952 | 0.1131 |
| G2G | **0.5027** | **0.4206** | 0.1280 | 0.0601 |

### SEED-IV (4-class)

| Model | Acc | F1 | Acc_std | F1_std |
|-------|-----|------|---------|--------|
| DCCA | 0.3275 | 0.1775 | 0.1267 | 0.1068 |
| DCCA_AM | 0.3698 | 0.2356 | 0.1021 | 0.0712 |
| BimodalLSTM | 0.4223 | 0.3010 | 0.0875 | 0.0904 |
| CRNN | 0.4226 | 0.3005 | 0.0180 | 0.0187 |
| BDDAE | 0.2989 | 0.1644 | 0.0460 | 0.0503 |
| MCAF | 0.2291 | 0.1717 | 0.1039 | 0.0276 |
| CMCM | **0.4310** | **0.4060** | 0.0827 | 0.0971 |
| CFDA_CSF | 0.3619 | 0.2258 | 0.0535 | 0.0802 |
| G2G | — | — | — | — |

*G2G incompatible with SEED-IV (hardcoded eye feature indices).*

## Running Experiments

See `scripts/reproduction/` for reproduction scripts. Each runs all models on a single dataset:

```bash
bash scripts/reproduction/run_seed.sh      # SEED
bash scripts/reproduction/run_seedv.sh     # SEED-V
bash scripts/reproduction/run_seediv.sh    # SEED-IV
bash scripts/reproduction/run_deap_valence.sh
bash scripts/reproduction/run_deap_arousal.sh
```

Full usage details in [CLAUDE.md](CLAUDE.md).
