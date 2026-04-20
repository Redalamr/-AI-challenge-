import numpy as np
from config import COST_FP, COST_FN, COST_TP, COST_TN


def compute_cost(tp, fp, fn, tn, cost_fp=COST_FP, cost_fn=COST_FN,
                 cost_tp=COST_TP, cost_tn=COST_TN):
    return cost_fp * fp + cost_fn * fn + cost_tp * tp + cost_tn * tn


def find_optimal_threshold(scores, labels, cost_fp=COST_FP, cost_fn=COST_FN,
                           cost_tp=COST_TP, cost_tn=COST_TN):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    thresholds = np.unique(scores)

    cost_curve = {}
    zero_fn_thresholds = []  # étape 1 : garder seulement FN=0

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        cost = compute_cost(tp, fp, fn, tn, cost_fp, cost_fn, cost_tp, cost_tn)
        cost_curve[float(t)] = cost
        if fn == 0:
            zero_fn_thresholds.append((t, fp, cost))

    if zero_fn_thresholds:
        # On trie par FP croissant, PUIS par Seuil décroissant
        # On veut le moins de FP possible, et parmi les ex-aequo, le seuil le plus HAUT
        best = min(zero_fn_thresholds, key=lambda x: (x[1], -x[0]))
        n_good = int((labels == 0).sum())
        # Only use FN=0 solution if it doesn't flag every single good image (TN > 0).
        # When even the best FN=0 threshold still gives TN=0, the cost function does
        # better: accepting 1 FN to clear all FPs saves more than it costs
        # (e.g. 26 FPs × 50 = 1300 > 500 = 1 FN).
        if n_good == 0 or best[1] < n_good:
            return float(best[0]), best[2], cost_curve
        # Fall through to cost minimisation when FN=0 requires TN=0.

    # Minimiser le coût total (FP × COST_FP + FN × COST_FN + TP × COST_TP)
    best_threshold = min(cost_curve, key=cost_curve.get)
    return best_threshold, cost_curve[best_threshold], cost_curve

def sensitivity_analysis(scores, labels, ratios=None):
    """
    Vary COST_FN / COST_FP ratio from 1 to 20.
    For each ratio, find the optimal threshold and compute recall.
    Returns list of dicts: {ratio, threshold, cost, recall, f1}
    """
    if ratios is None:
        ratios = list(range(1, 21))

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    results = []

    for r in ratios:
        cfn = COST_FP * r  # keep FP fixed, scale FN
        threshold, cost, _ = find_optimal_threshold(
            scores, labels, cost_fp=COST_FP, cost_fn=cfn
        )
        preds = (scores >= threshold).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        results.append({
            "ratio": r,
            "threshold": threshold,
            "cost": cost,
            "recall": recall,
            "f1": f1,
        })

    return results
