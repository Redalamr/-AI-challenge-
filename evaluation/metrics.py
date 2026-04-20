import numpy as np


def _confusion(preds, labels):
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    return tp, fp, fn, tn


def _roc_curve(scores, labels):
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    tpr_list, fpr_list = [0.0], [0.0]
    tp = fp = 0

    for lbl in sorted_labels:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos if n_pos > 0 else 0.0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0.0)

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    return np.array(fpr_list), np.array(tpr_list)


def _trapz(x, y):
    return float(np.trapz(y, x))


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    fpr, tpr = _roc_curve(scores, labels)
    return _trapz(fpr, tpr)


def pixel_auroc(anomaly_maps, gt_masks):
    all_scores = np.concatenate([m.ravel() for m in anomaly_maps])
    all_labels = np.concatenate([g.ravel().astype(np.int32) for g in gt_masks])
    return auroc(all_scores, all_labels)


def f1_at_threshold(scores, labels, threshold):
    preds = (np.asarray(scores) >= threshold).astype(int)
    tp, fp, fn, tn = _confusion(preds, np.asarray(labels))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def best_f1(scores, labels):
    # on teste tous les seuils possibles
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    thresholds = np.unique(scores)
    best = {"f1": -1.0, "threshold": thresholds[0], "precision": 0.0, "recall": 0.0}
    for t in thresholds:
        f1, prec, rec = f1_at_threshold(scores, labels, t)
        if f1 > best["f1"]:
            best = {"f1": f1, "threshold": t, "precision": prec, "recall": rec}
    return best


def roc_curve(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    fpr, tpr = _roc_curve(scores, labels)
    thresholds = np.concatenate([[sorted_scores[0] + 1e-6], sorted_scores, [0.0]])
    return fpr, tpr, thresholds
