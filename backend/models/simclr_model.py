import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNetSimCLR(nn.Module):
    """
    Matches the training-time architecture:
    ResNet18 backbone + projection head (out_dim=512)
    """
    def __init__(self, base_model="resnet18", out_dim=512):
        super(ResNetSimCLR, self).__init__()

        # Build base encoder
        if base_model == "resnet18":
            backbone = resnet18(weights=None)
            dim_in = backbone.fc.in_features
            backbone.fc = nn.Identity()  # type: ignore[attr-defined]
            self.encoder = backbone
        else:
            raise ValueError(f"Unsupported base model: {base_model}")

        # Projection head: same as training
        self.projector = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)  # (batch, 512)
        z = self.projector(h)
        return z
