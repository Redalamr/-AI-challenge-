import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import BACKBONE, LAYERS


class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name=BACKBONE, layers=LAYERS):
        super().__init__()
        self.layers = layers
        self._features = {}

        backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        # vire la tete de classif
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        self._layer_modules = {
            "layer1": backbone.layer1,
            "layer2": backbone.layer2,
            "layer3": backbone.layer3,
            "layer4": backbone.layer4,
        }

        self._hooks = []
        for layer_name in self.layers:
            module = self._layer_modules[layer_name]
            hook = module.register_forward_hook(self._make_hook(layer_name))
            self._hooks.append(hook)

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _make_hook(self, name):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def forward(self, x):
        self._features.clear()
        with torch.no_grad():
            _ = self.backbone(x)
        return dict(self._features)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def concatenate_features(feature_dict, layers, target_size=None):
    maps = [feature_dict[l] for l in layers]
    if target_size is None:
        # on ramene tout a la meme taille
        target_size = min((m.shape[-2], m.shape[-1]) for m in maps)

    aligned = []
    for m in maps:
        if (m.shape[-2], m.shape[-1]) != target_size:
            m = F.interpolate(m, size=target_size, mode="bilinear", align_corners=False)
        aligned.append(m)
    return torch.cat(aligned, dim=1)
