# PatchCore — MVTec Anomaly Detection
### AI Challenge — Industrial Quality Control — Équipe 2 (Corel, Reda)

---

## Setup

```bash
pip install -r requirements.txt
```

GPU is used automatically if available. CPU fallback works but is slower.

---

## Dataset

The MVTec dataset must be at `../mvtec_anomaly_detection/` relative to this folder (already present).

Expected structure:
```
mvtec_anomaly_detection/
├── bottle/
│   ├── train/good/
│   ├── test/{good, broken_large, ...}/
│   └── ground_truth/{broken_large, ...}/
└── ...
```

---

## Running

### MVP (3 categories: bottle, carpet, screw)
```bash
cd mvtec-poc
python main.py
```

### All 15 categories
```bash
python main.py --categories all
```

### Specific categories
```bash
python main.py --categories bottle cable hazelnut
```

### Options
| Flag | Default | Description |
|------|---------|-------------|
| `--categories` | `bottle carpet screw` | Categories to run, or `all` |
| `--coreset-ratio` | `0.01` | Memory bank subsampling ratio (0.01 = 1 %) |
| `--no-vis` | off | Skip plot generation |

---

## Architecture

```
mvtec-poc/
├── config.py               # Hyperparameters, paths, cost values
├── data/
│   ├── loader.py           # MVTecDataset + DataLoader
│   └── transforms.py       # Resize → CenterCrop(224) → Normalize
├── models/
│   ├── feature_extractor.py  # WideResNet50 hooks on layer2+layer3
│   └── patchcore.py          # Memory bank, coreset, fit/predict
├── evaluation/
│   ├── metrics.py          # AUROC, F1, pixel-AUROC (from scratch, no sklearn)
│   ├── cost_matrix.py      # Business cost optimisation + sensitivity
│   └── visualize.py        # All plots
├── inference/
│   └── predict.py          # Single-image API + FPS benchmark
└── main.py                 # Orchestrator
```

---

## Outputs

All results are saved to `results/<category>/`:
- `roc_<category>.png` — ROC curve with AUC
- `confusion_<category>.png` — Confusion matrix at cost-optimal threshold
- `heatmap_<category>_*.png` — Anomaly map overlay on image
- `cost_threshold_<category>.png` — Cost vs threshold sweep
- `pareto_<category>.png` — Pareto front: cost vs recall (sensitivity analysis)
- `examples_<category>.png` — Grid of TP / FP / FN / TN samples

---

## Algorithm

**PatchCore** (Roth et al., 2022):
1. Extract mid-level features from WideResNet50 (`layer2` + `layer3`) via forward hooks
2. Average-pool each 3×3 patch neighbourhood to aggregate spatial context
3. Store all training patch features in a **memory bank**
4. **Coreset subsampling** (greedy farthest-point) reduces the bank to 1% of its size
5. At test time, the anomaly score of each patch = L2 distance to its nearest neighbour in the bank, re-weighted by the local density of that neighbour
6. Image score = max patch score; anomaly map = upsampled + Gaussian-smoothed patch distances

---

## Business Cost Matrix

| | Predicted Good | Predicted Defect |
|---|---|---|
| **True Good** | TN = 0 | FP = **50** |
| **True Defect** | FN = **500** | TP = -10 |

The threshold is optimised to **minimise total cost**, not F1.  
A sensitivity analysis varies the FN/FP ratio from 1 to 20 to show how the optimal threshold changes.

---

## Validation Criteria

- [x] End-to-end pipeline on ≥ 3 categories
- [x] Image AUROC > 95% target on bottle
- [x] All metrics implemented from scratch (no `sklearn.metrics`)
- [x] Cost matrix produces a different threshold than F1
- [x] ≥ 5 visualisations generated per category
- [x] Inference time measured and reported (FPS)
- [x] Modular code (one responsibility per file)
