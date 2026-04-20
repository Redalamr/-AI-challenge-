import torch
import torchvision.transforms as T
from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def _pil_to_tensor(pic):
    # ToTensor plante avec numpy 2 donc on fait ca a la main
    mode = pic.mode
    w, h = pic.size
    raw = bytearray(pic.tobytes())
    t = torch.frombuffer(raw, dtype=torch.uint8).clone()
    if mode == "RGB":
        t = t.reshape(h, w, 3).permute(2, 0, 1)
    elif mode == "L":
        t = t.reshape(1, h, w)
    else:
        pic = pic.convert("RGB")
        w, h = pic.size
        raw = bytearray(pic.tobytes())
        t = torch.frombuffer(raw, dtype=torch.uint8).clone()
        t = t.reshape(h, w, 3).permute(2, 0, 1)
    return t.float() / 255.0


def get_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(IMAGE_SIZE),
        T.Lambda(_pil_to_tensor),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_mask_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(IMAGE_SIZE),
        T.Lambda(_pil_to_tensor),
    ])
