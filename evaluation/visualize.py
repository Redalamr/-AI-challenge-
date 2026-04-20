import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

from config import RESULTS_DIR, IMAGENET_MEAN, IMAGENET_STD
from utils import t2np


def _denorm(tensor):
    # remet les pixels entre 0 et 255
    img = t2np(tensor).transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def plot_roc_curve(fpr, tpr, auc_val, category, out_dir=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {category}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir or RESULTS_DIR, f"roc_{category}.png")
    plt.close(fig)


def plot_confusion_matrix(tp, fp, fn, tn, category, threshold, out_dir=None):
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Good", "Pred Defect"])
    ax.set_yticklabels(["True Good", "True Defect"])
    ax.set_title(f"Confusion Matrix — {category}\n(threshold={threshold:.4f})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    _save(fig, out_dir or RESULTS_DIR, f"confusion_{category}.png")
    plt.close(fig)


def plot_heatmap_overlay(image_tensor, anomaly_map, gt_mask, score, label,
                         category, idx, out_dir=None):
    img = _denorm(image_tensor)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img)
    axes[0].set_title(f"Image (label={'defect' if label else 'good'})")
    axes[0].axis("off")

    hm_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    axes[1].imshow(img)
    axes[1].imshow(hm_norm, cmap="jet", alpha=0.4)
    axes[1].set_title(f"Anomaly Map (score={score:.3f})")
    axes[1].axis("off")

    axes[2].imshow(gt_mask, cmap="gray")
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis("off")

    fig.suptitle(f"{category} — sample {idx}")
    fig.tight_layout()
    _save(fig, out_dir or RESULTS_DIR, f"heatmap_{category}_{idx}.png")
    plt.close(fig)


def plot_example_grid(results, images_tensor_list, category, out_dir=None, n_each=3):
    scores = results["scores"]
    labels = results["labels"]
    threshold = results.get("threshold", np.median(scores))
    preds = (scores >= threshold).astype(int)

    tp_idx = np.where((preds == 1) & (labels == 1))[0][:n_each]
    fp_idx = np.where((preds == 1) & (labels == 0))[0][:n_each]
    fn_idx = np.where((preds == 0) & (labels == 1))[0][:n_each]
    tn_idx = np.where((preds == 0) & (labels == 0))[0][:n_each]

    groups = [("TP", tp_idx), ("FP", fp_idx), ("FN", fn_idx), ("TN", tn_idx)]
    fig, axes = plt.subplots(4, n_each, figsize=(n_each * 3, 14))

    for row, (name, idxs) in enumerate(groups):
        for col in range(n_each):
            ax = axes[row, col]
            if col < len(idxs):
                i = idxs[col]
                img = _denorm(images_tensor_list[i])
                ax.imshow(img)
                ax.set_title(
                    f"{name}\ns={scores[i]:.2f}\n{results['defect_types'][i]}",
                    fontsize=7,
                )
            ax.axis("off")

    fig.suptitle(f"Examples — {category}", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir or RESULTS_DIR, f"examples_{category}.png")
    plt.close(fig)


def plot_cost_vs_threshold(cost_curve, optimal_threshold, category, out_dir=None):
    thresholds = sorted(cost_curve.keys())
    costs = [cost_curve[t] for t in thresholds]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, costs, lw=2, color="steelblue")
    ax.axvline(optimal_threshold, color="red", linestyle="--",
               label=f"Optimal threshold = {optimal_threshold:.4f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total Cost")
    ax.set_title(f"Cost vs Threshold — {category}")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir or RESULTS_DIR, f"cost_threshold_{category}.png")
    plt.close(fig)


def plot_pareto_cost_recall(sensitivity_results, category, out_dir=None):
    recalls = [r["recall"] for r in sensitivity_results]
    costs = [r["cost"] for r in sensitivity_results]
    ratios = [r["ratio"] for r in sensitivity_results]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(recalls, costs, c=ratios, cmap="viridis", s=80, zorder=3)
    ax.plot(recalls, costs, lw=1, alpha=0.5)
    for r, rec, cost in zip(ratios, recalls, costs):
        ax.annotate(f"r={r}", (rec, cost), textcoords="offset points",
                    xytext=(4, 4), fontsize=6)
    plt.colorbar(sc, ax=ax, label="FN/FP ratio")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Total Cost")
    ax.set_title(f"Pareto: Cost vs Recall — {category}")
    fig.tight_layout()
    _save(fig, out_dir or RESULTS_DIR, f"pareto_{category}.png")
    plt.close(fig)


def _save(fig, directory, filename):
    path = Path(directory) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=100, bbox_inches="tight")
