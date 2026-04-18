"""Ablation variant S3: ViT without class token (uses Global Average Pooling).

Removes the learnable class token and replaces it with Global Average Pooling
over all patch tokens to assess the role of the class token in aggregating
global image information.
"""

import torch
import torch.nn as nn
from models.vit import PatchEmbedding, TransformerBlock


class ViT_NoCls(nn.Module):
    """ViT-B/16 without class token; uses Global Average Pooling instead.

    Args:
        num_classes (int): Number of output classes.
        embed_dim (int): Embedding dimension. Defaults to ``768``.
        depth (int): Number of Transformer blocks. Defaults to ``12``.
        num_heads (int): Number of attention heads. Defaults to ``12``.
    """

    def __init__(self, num_classes: int, embed_dim: int = 768, depth: int = 12, num_heads: int = 12) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # Positional embedding sized for patches only (no class token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.norm   = nn.LayerNorm(embed_dim)
        self.head   = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        # Global Average Pooling over all patch tokens
        x = self.norm(x).mean(dim=1)
        return self.head(x)


def get_vit_no_cls(num_classes: int) -> ViT_NoCls:
    """Return ViT-B/16 without class token.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        ViT_NoCls: Model using Global Average Pooling instead of class token.
    """
    return ViT_NoCls(num_classes=num_classes)
