import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from .base import Encoder

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class ResNetEncoder(Encoder):
    """Frozen ImageNet ResNet as MFEC's φ. embed(obs) -> (B, d) pooled features."""

    def __init__(self, model_name="resnet18", weights_path=None,
                 image_size=224, device=None):
        # weights_path=None -> torchvision downloads IMAGENET1K_V1;
        # pass a path on the offline cluster (same reason DINOv2 takes one).
        if weights_path is None:
            self.model = tvm.get_model(model_name, weights="IMAGENET1K_V1")
        else:
            self.model = tvm.get_model(model_name, weights=None)
            sd = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(sd.get("state_dict", sd), strict=True)

        self.state_dim = int(self.model.fc.in_features)   # 512 r18/r34, 2048 r50
        self.model.fc = nn.Identity()                     # -> (B, d) after avgpool
        self.image_size = image_size

        self.model.eval()                    # CRITICAL — see gotcha below
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std  = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        if device is not None:
            self.model.to(device)

    @torch.no_grad()
    def embed(self, obs):
        dev = obs.device
        if next(self.model.parameters()).device != dev:
            self.model.to(dev)
        x = obs.float().reshape(-1, *obs.shape[-3:])
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, (self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        x = (x - self._mean.to(dev)) / self._std.to(dev)
        return self.model(x).float()

    def state(self):
        return {"model_state_dict": self.model.state_dict()}

    def load_state(self, state):
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
