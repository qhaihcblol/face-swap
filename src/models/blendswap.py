from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .generator import ADDGenerator
from .encoder import MultilevelAttributesEncoder
from .iresnet import iresnet100


class _Normalize(nn.Module):
    """[0, 1] → [-1, 1]"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0


class BlendSwap(nn.Module):
    """
    BlendSwap face-swap model (inference).

    Args:
        target_img: [N, 3, 256, 256]  N faces to receive the identity
        source_img: [1, 3, 112, 112]  source face (identity donor)

    Returns:
        [N, 3, 256, 256]  swapped faces

    Note: z_id [1, 512] from source broadcasts to N targets naturally
    inside the ADD layers via expand_as — no explicit repeat needed.
    """

    # Training checkpoints use EMA suffixes; remap to clean inference names.
    _CKPT_REMAP: dict[str, str] = {
        "Z_e.":           "identity_encoder.",
        "G_ema.":         "generator.",
        "E_ema.":         "attribute_encoder.",
        "mask_head_ema.": "mask_head.",
    }

    def __init__(self) -> None:
        super().__init__()
        self.identity_encoder = nn.Sequential(
            _Normalize(),
            iresnet100(pretrained=False, fp16=False),
        )
        self.attribute_encoder = MultilevelAttributesEncoder()
        self.generator = ADDGenerator(512, 3)
        self.mask_head = nn.Conv2d(64, 1, 1)

    def forward(self, target_img: torch.Tensor, source_img: torch.Tensor) -> torch.Tensor:
        z_id = F.normalize(self.identity_encoder(source_img))  # [1, 512]
        features = self.attribute_encoder(target_img)
        output = self.generator(z_id, features)
        mask = self.mask_head(features[-1]).sigmoid()
        return output * mask + target_img * (1 - mask)

    @classmethod
    def from_checkpoint(cls, path) -> BlendSwap:
        """Load a training checkpoint, remapping EMA keys to inference names."""
        model = cls()
        raw = torch.load(path, map_location="cpu")
        state = {}
        for k, v in raw.items():
            for old, new in cls._CKPT_REMAP.items():
                if k.startswith(old):
                    k = new + k[len(old):]
                    break
            state[k] = v
        model.load_state_dict(state, strict=False)
        return model.eval()
