import time
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from config import FPS_BENCH_N
from data.transforms import get_transform


def predict_single(model, image_path_or_tensor):
    if isinstance(image_path_or_tensor, (str, Path)):
        transform = get_transform()
        img = Image.open(image_path_or_tensor).convert("RGB")
        tensor = transform(img).unsqueeze(0)
    else:
        tensor = image_path_or_tensor
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

    score, anomaly_map = model.predict(tensor)
    return score, anomaly_map


def benchmark_fps(model, test_loader, n=FPS_BENCH_N):
    times = []
    count = 0

    for images, *_ in test_loader:
        for i in range(len(images)):
            if count >= n:
                break
            img = images[i:i+1]
            t0 = time.perf_counter()
            model.predict(img)
            times.append(time.perf_counter() - t0)
            count += 1
        if count >= n:
            break

    if not times:
        return 0.0

    mean_ms = np.mean(times) * 1000
    fps = 1.0 / np.mean(times)
    print(f"Inference: {fps:.1f} FPS  ({mean_ms:.1f} ms/image)  [{count} images]")
    return fps
