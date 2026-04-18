"""Ablation variant S2: ViT without positional embedding.

Removes the learnable positional embedding to assess how much spatial
position information contributes to classification performance.
"""

import torch
import torch.nn as nn
from models.vit import PatchEmbedding, TransformerBlock


class ViT_NoPos(nn.Module):
    """ViT-B/16 without positional embedding.

    Args:
        num_classes (int): Number of output classes.
        embed_dim (int): Embedding dimension. Defaults to ``768``.
        depth (int): Number of Transformer blocks. Defaults to ``12``.
        num_heads (int): Number of attention heads. Defaults to ``12``.
    """

    def __init__(self, num_classes: int, embed_dim: int = 768, depth: int = 12, num_heads: int = 12) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embedding intentionally omitted
        self.blocks = nn.ModuleList([TransformerBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.norm   = nn.LayerNorm(embed_dim)
        self.head   = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # No positional embedding added
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


def get_vit_no_pos(num_classes: int) -> ViT_NoPos:
    """Return ViT-B/16 without positional embedding.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        ViT_NoPos: Model without positional embedding.
    """
    return ViT_NoPos(num_classes=num_classes)
