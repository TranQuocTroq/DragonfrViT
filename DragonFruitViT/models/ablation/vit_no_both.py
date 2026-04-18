"""Ablation variant S5: ViT without positional embedding and class token.

Removes both positional embedding and class token simultaneously,
using Global Average Pooling for aggregation.
"""

import torch
import torch.nn as nn
from models.vit import PatchEmbedding, TransformerBlock


class ViT_NoBoth(nn.Module):
    """ViT-B/16 without positional embedding or class token.

    Args:
        num_classes (int): Number of output classes.
        embed_dim (int): Embedding dimension. Defaults to ``768``.
        depth (int): Number of Transformer blocks. Defaults to ``12``.
        num_heads (int): Number of attention heads. Defaults to ``12``.
    """

    def __init__(self, num_classes: int, embed_dim: int = 768, depth: int = 12, num_heads: int = 12) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
        # No class token, no positional embedding
        self.blocks = nn.ModuleList([TransformerBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.norm   = nn.LayerNorm(embed_dim)
        self.head   = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)  # Global Average Pooling
        return self.head(x)


def get_vit_no_both(num_classes: int) -> ViT_NoBoth:
    """Return ViT-B/16 without positional embedding and class token.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        ViT_NoBoth: Model with both positional embedding and class token removed.
    """
    return ViT_NoBoth(num_classes=num_classes)
