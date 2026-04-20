"""
PatchCore — Industrial Anomaly Detection on MVTec AD
AI Challenge — Équipe 2 (Corel, Reda)
"""

import sys
import argparse
import numpy as np
import torch

from config import MVP_CATEGORIES, CATEGORIES, RESULTS_DIR, CORESET_RATIO
from data.loader import get_dataloader
from models.patchcore import PatchCore
from evaluation.metrics import auroc, pixel_auroc, best_f1, roc_curve
from evaluation.cost_matrix import find_optimal_threshold, sensitivity_analysis
from evaluation.visualize import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_heatmap_overlay,
    plot_cost_vs_threshold,
    plot_pareto_cost_recall,
    plot_example_grid,
)
from inference.predict import benchmark_fps


# ------------------------------------------------------------------ #
#  Per-category pipeline                                               #
# ------------------------------------------------------------------ #

def run_category(category, coreset_ratio=0.01, run_vis=True):
    print(f"\n{'='*60}")
    print(f"  Category: {category.upper()}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # --- Data ---
    train_loader = get_dataloader(category, split="train", shuffle=False)
    test_loader = get_dataloader(category, split="test", shuffle=False)

    # --- Fit ---
    from config import CORESET_RATIO_OVERRIDE
    ratio = CORESET_RATIO_OVERRIDE.get(category, coreset_ratio)
    model = PatchCore(device=device, coreset_ratio=ratio)
    model.fit(train_loader)

    # --- Evaluate ---
    results = model.evaluate(test_loader)

    scores = results["scores"]
    labels = results["labels"]
    anomaly_maps = results["anomaly_maps"]
    gt_masks = results["gt_masks"]

    # --- Metrics ---
    img_auroc = auroc(scores, labels)
    pix_auroc = pixel_auroc(anomaly_maps, gt_masks)
    f1_info = best_f1(scores, labels)
    opt_thresh, opt_cost, cost_curve = find_optimal_threshold(scores, labels)

    # Store threshold for example grid
    results["threshold"] = opt_thresh

    # Confusion matrix at optimal threshold
    preds = (scores >= opt_thresh).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    # FPS
    fps = benchmark_fps(model, test_loader, n=min(50, len(test_loader.dataset)))

    # --- Print ---
    print(f"\n  Image AUROC  : {img_auroc:.4f}")
    print(f"  Pixel AUROC  : {pix_auroc:.4f}")
    print(f"  Best F1      : {f1_info['f1']:.4f}  (thresh={f1_info['threshold']:.4f})")
    print(f"  Cost-opt thresh : {opt_thresh:.4f}  (cost={opt_cost:.0f})")
    print(f"  TP/FP/FN/TN  : {tp}/{fp}/{fn}/{tn}")
    print(f"  FPS          : {fps:.1f}")

    # --- Visualisations ---
    if run_vis:
        cat_dir = RESULTS_DIR / category
        cat_dir.mkdir(exist_ok=True)

        fpr, tpr, _ = roc_curve(scores, labels)
        plot_roc_curve(fpr, tpr, img_auroc, category, out_dir=cat_dir)
        plot_confusion_matrix(tp, fp, fn, tn, category, opt_thresh, out_dir=cat_dir)
        plot_cost_vs_threshold(cost_curve, opt_thresh, category, out_dir=cat_dir)

        sens = sensitivity_analysis(scores, labels)
        plot_pareto_cost_recall(sens, category, out_dir=cat_dir)

        # Heatmap overlays: first defect image
        defect_indices = np.where(labels == 1)[0]
        if len(defect_indices) > 0:
            i = int(defect_indices[0])
            img_tensor = test_loader.dataset[i][0]
            plot_heatmap_overlay(
                img_tensor, anomaly_maps[i], gt_masks[i],
                scores[i], labels[i], category, i, out_dir=cat_dir,
            )

        # Example grid (need image tensors for all test samples)
        all_images = [test_loader.dataset[j][0] for j in range(len(test_loader.dataset))]
        plot_example_grid(results, all_images, category, out_dir=cat_dir)

        print(f"  Plots saved -> {cat_dir}")

    return {
        "category": category,
        "img_auroc": img_auroc,
        "pix_auroc": pix_auroc,
        "f1": f1_info["f1"],
        "cost": opt_cost,
        "threshold": opt_thresh,
        "fps": fps,
    }


# ------------------------------------------------------------------ #
#  Summary table                                                       #
# ------------------------------------------------------------------ #

def print_summary(all_results):
    print(f"\n{'='*80}")
    print(f"{'Category':<14} {'ImgAUROC':>9} {'PixAUROC':>9} {'F1':>7} {'Cost':>8} {'Thresh':>8} {'FPS':>6}")
    print(f"{'-'*80}")
    for r in all_results:
        print(
            f"{r['category']:<14} "
            f"{r['img_auroc']:>9.4f} "
            f"{r['pix_auroc']:>9.4f} "
            f"{r['f1']:>7.4f} "
            f"{r['cost']:>8.0f} "
            f"{r['threshold']:>8.4f} "
            f"{r['fps']:>6.1f}"
        )
    if all_results:
        print(f"{'-'*80}")
        avg_img = np.mean([r["img_auroc"] for r in all_results])
        avg_pix = np.mean([r["pix_auroc"] for r in all_results])
        avg_f1 = np.mean([r["f1"] for r in all_results])
        avg_fps = np.mean([r["fps"] for r in all_results])
        print(
            f"{'MEAN':<14} "
            f"{avg_img:>9.4f} "
            f"{avg_pix:>9.4f} "
            f"{avg_f1:>7.4f} "
            f"{'':>8} "
            f"{'':>8} "
            f"{avg_fps:>6.1f}"
        )
    print(f"{'='*80}\n")


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(description="PatchCore MVTec evaluation")
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Categories to run (default: MVP_CATEGORIES). Use 'all' for all 15.",
    )
    parser.add_argument(
        "--coreset-ratio", type=float, default=CORESET_RATIO,
        help=f"Coreset subsampling ratio (default {CORESET_RATIO}).",
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="Skip generating plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.categories is None:
        cats = MVP_CATEGORIES
    elif args.categories == ["all"]:
        cats = CATEGORIES
    else:
        cats = args.categories

    print(f"\nRunning PatchCore on: {cats}")
    print(f"Coreset ratio: {args.coreset_ratio}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Results dir: {RESULTS_DIR}\n")

    all_results = []
    for cat in cats:
        try:
            r = run_category(cat, coreset_ratio=args.coreset_ratio, run_vis=not args.no_vis)
            all_results.append(r)
        except Exception as e:
            print(f"[ERROR] {cat}: {e}")
            import traceback
            traceback.print_exc()

    print_summary(all_results)


if __name__ == "__main__":
    main()
