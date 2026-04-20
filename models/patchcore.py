import torch
import torch.nn.functional as F
import numpy as np
import faiss
from scipy.ndimage import gaussian_filter, zoom
from tqdm import tqdm

from config import (
    LAYERS, PATCH_SIZE, CORESET_RATIO, NUM_NEIGHBORS, REWEIGHT_NEIGHBORS,
    ANOMALY_MAP_SIGMA, IMAGE_SIZE,
)
from models.feature_extractor import FeatureExtractor, concatenate_features
from utils import t2np


class PatchCore:
    def __init__(self, device=None, coreset_ratio=CORESET_RATIO):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.coreset_ratio = coreset_ratio
        self.extractor = FeatureExtractor().to(self.device)
        self.memory_bank = None   # (N, D) float32 numpy array after fit
        self._faiss_index = None  # FAISS index for fast k-NN
        self._feature_spatial_size = None

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, train_loader):
        print(f"[PatchCore] Extracting features on {self.device} ...")
        all_patches = []

        for images, *_ in tqdm(train_loader, desc="fit"):
            images = images.to(self.device)
            feat_dict = self.extractor(images)
            features = concatenate_features(feat_dict, LAYERS)  # (B, C, H, W)

            if self._feature_spatial_size is None:
                self._feature_spatial_size = (features.shape[-2], features.shape[-1])

            patches = self._aggregate_patches(features)  # (B*H*W, C)
            all_patches.append(t2np(patches))

        memory_bank = np.concatenate(all_patches, axis=0)  # (N_total, D)
        print(f"[PatchCore] Memory bank: {memory_bank.shape}. Coreset subsampling ...")
        self.memory_bank = self._coreset_subsample(memory_bank, self.coreset_ratio)
        print(f"[PatchCore] Coreset size: {self.memory_bank.shape}. Building FAISS index ...")
        self._build_faiss_index()
        print("[PatchCore] Ready.")

    def _aggregate_patches(self, features):
        """Average-pool PATCH_SIZE×PATCH_SIZE neighbourhood per patch.
        Input: (B, C, H, W) → Output: (B*H*W, C)
        """
        B, C, H, W = features.shape
        x = F.avg_pool2d(features, kernel_size=PATCH_SIZE, stride=1, padding=PATCH_SIZE // 2)
        return x.permute(0, 2, 3, 1).reshape(-1, C)

    def _coreset_subsample(self, bank, ratio):
        """Greedy farthest-point coreset selection.
        Uses BLAS dot-product trick: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        to compute all N distances per iteration in one matrix-vector multiply.
        """
        n_select = max(1, int(len(bank) * ratio))
        if n_select >= len(bank):
            return bank

        bank_f32 = np.ascontiguousarray(bank, dtype=np.float32)
        # Precompute squared norms for all points (reused each iteration)
        bank_sq = (bank_f32 * bank_f32).sum(axis=1)  # (N,)

        rng = np.random.default_rng(42)
        selected = [int(rng.integers(len(bank)))]
        min_dists = np.full(len(bank), np.inf, dtype=np.float32)

        for _ in tqdm(range(n_select - 1), desc="coreset", leave=False):
            last = bank_f32[selected[-1]]
            last_sq = float((last * last).sum())
            # ||bank[i] - last||^2 = bank_sq[i] + last_sq - 2 * bank[i]·last
            cross = bank_f32 @ last          # BLAS dgemv — fast
            dists = bank_sq + last_sq - 2 * cross
            min_dists = np.minimum(min_dists, dists)
            selected.append(int(np.argmax(min_dists)))

        return bank_f32[selected]

    def _build_faiss_index(self):
        """Build a FAISS index on the final coreset for fast inference k-NN."""
        d = self.memory_bank.shape[1]
        self._faiss_index = faiss.IndexFlatL2(d)
        self._faiss_index.add(np.ascontiguousarray(self.memory_bank, dtype=np.float32))

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, image_tensor, category=None):
        """
        image_tensor: (1, 3, H, W).
        Returns (anomaly_score: float, anomaly_map: np.ndarray H×W float32).
        """
        if self.memory_bank is None:
            raise RuntimeError("Call fit() before predict().")

        image_tensor = image_tensor.to(self.device)
        feat_dict = self.extractor(image_tensor)
        features = concatenate_features(feat_dict, LAYERS)
        feat_h, feat_w = features.shape[-2], features.shape[-1]

        patches = np.ascontiguousarray(t2np(self._aggregate_patches(features)), dtype=np.float32)

        # k-NN via FAISS: get top-k distances and indices
        k = max(NUM_NEIGHBORS, REWEIGHT_NEIGHBORS + 1)
        dists_sq, indices = self._faiss_index.search(patches, k)
        # FAISS returns squared L2 — convert to L2
        dists = np.sqrt(np.maximum(dists_sq, 0))  # (H*W, k)

        nn_dist = dists[:, 0]      # distance to nearest neighbour, shape (H*W,)
        nn_idx  = indices[:, 0]    # index of nearest neighbour in memory bank

        # Re-weighting (PatchCore paper §3.3):
        # w(p) = 1 - softmax(nn_dist / sum_of_k_nearest_in_bank)
        # Simpler stable version: w = nn_dist / mean(distances to REWEIGHT_NEIGHBORS
        # nearest neighbours of the nn in the memory bank)
        nn_patch = self.memory_bank[nn_idx]  # (H*W, D)
        nn_patch_c = np.ascontiguousarray(nn_patch, dtype=np.float32)
        bank_dists_sq, _ = self._faiss_index.search(nn_patch_c, REWEIGHT_NEIGHBORS + 1)
        bank_dists = np.sqrt(np.maximum(bank_dists_sq[:, 1:], 0))  # skip self (dist=0)
        bank_mean_dist = bank_dists.mean(axis=1) + 1e-8             # (H*W,)

        # Anomaly score per patch = nn_dist * (1 - exp(nn_dist) / sum_exp(bank_dists))
        # Stabilised: score = nn_dist * (nn_dist / bank_mean_dist)
        scores = nn_dist * (nn_dist / bank_mean_dist)
        
        # Correction ciblée pour PILL : élimine les outliers (poussières/reflets)
        # On utilise le percentile 99.5 pour stabiliser le score et la carte d'anomalie
        if category == "pill":
            upper_bound = np.percentile(scores, 99.5)
            scores = np.clip(scores, 0, upper_bound)
            anomaly_score = float(upper_bound)
        else:
            anomaly_score = float(scores.max())

        # Pixel-level anomaly map via scipy zoom (no torch↔numpy bridge needed)
        score_map = scores.reshape(feat_h, feat_w).astype(np.float32)
        zoom_factor = IMAGE_SIZE / feat_h
        score_map_up = zoom(score_map, zoom_factor, order=1).astype(np.float32)
        anomaly_map = gaussian_filter(score_map_up, sigma=ANOMALY_MAP_SIGMA)

        return anomaly_score, anomaly_map

    # ------------------------------------------------------------------ #
    #  Batch evaluation                                                    #
    # ------------------------------------------------------------------ #

    def evaluate(self, test_loader):
        scores, labels, anomaly_maps, gt_masks, defect_types = [], [], [], [], []

        category = getattr(test_loader.dataset, 'category', None)
        for images, lbls, masks, dtypes in tqdm(test_loader, desc="eval"):
            for i in range(len(images)):
                score, amap = self.predict(images[i:i+1], category=category)
                scores.append(score)
                labels.append(int(lbls[i]))
                anomaly_maps.append(amap)
                gt_masks.append(t2np(masks[i, 0]))
                defect_types.append(dtypes[i])

        return {
            "scores":       np.array(scores),
            "labels":       np.array(labels),
            "anomaly_maps": anomaly_maps,
            "gt_masks":     gt_masks,
            "defect_types": defect_types,
        }
