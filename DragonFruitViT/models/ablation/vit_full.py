"""Ablation variant S1: Full ViT baseline (no components removed).

Identical to ``models/vit.py``. Included here as the reference baseline
for the ablation study so all variants share the same training script.
"""

from models.vit import VisionTransformer


def get_vit_full(num_classes: int) -> VisionTransformer:
    """Return the full ViT-B/16 baseline model.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        VisionTransformer: Full ViT model with all components active.
    """
    return VisionTransformer(num_classes=num_classes)
