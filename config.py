import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent
DATASET_ROOT = ROOT_DIR.parent / "mvtec_anomaly_detection"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Model
IMAGE_SIZE = 224
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]

# PatchCore
PATCH_SIZE = 3          # neighborhood averaging kernel
CORESET_RATIO = 0.10    # 10% for final version
NUM_NEIGHBORS = 9       # k nearest neighbors for re-weighting
REWEIGHT_NEIGHBORS = 3  # b neighbors in memory bank re-weighting

# Gaussian smoothing for anomaly map
ANOMALY_MAP_SIGMA = 4

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Business cost matrix
COST_FP = 50    # False Positive: healthy product discarded
COST_FN = 500   # False Negative: defect shipped → recall risk
COST_TP = -10   # True Positive: detected defect → rework cost
COST_TN = 0     # True Negative: healthy accepted

# All 15 MVTec categories
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# MVP: 3 representative categories for fast validation
MVP_CATEGORIES = ["bottle", "carpet", "screw"]

# DataLoader
BATCH_SIZE = 32
NUM_WORKERS = 0  # 0 for Windows compatibility

# FPS benchmark
FPS_BENCH_N = 100

# Per-category coreset override
CORESET_RATIO_OVERRIDE = {
    "pill": 0.25,   # 25% au lieu de 10%
    "screw": 0.25,
}