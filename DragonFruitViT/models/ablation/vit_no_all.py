"""Ablation variant S6: ViT without positional embedding, class token, or MLP.

Removes all three major architectural components simultaneously, leaving
only multi-head self-attention with layer normalization and Global Average Pooling.
"""

import torch
import torch.nn as nn
from models.vit import PatchEmbedding, Attention


class TransformerBlock_AttentionOnly(nn.Module):
    """Transformer block with attention only — no MLP, no second LayerNorm.

    Args:
        dim (int): Feature dimension. Defaults to ``768``.
        num_heads (int): Number of attention heads. Defaults to ``12``.
    """

    def __init__(self, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm1(x))


class ViT_NoAll(nn.Module):
    """ViT-B/16 without positional embedding, class token, or MLP blocks.

    Represents the minimal Transformer architecture: patches communicate
    via self-attention only, with Global Average Pooling for aggregation.

    Args:
        num_classes (int): Number of output classes.
        embed_dim (int): Embedding dimension. Defaults to ``768``.
        depth (int): Number of Transformer blocks. Defaults to ``12``.
        num_heads (int): Number of attention heads. Defaults to ``12``.
    """

    def __init__(self, num_classes: int, embed_dim: int = 768, depth: int = 12, num_heads: int = 12) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock_AttentionOnly(dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)  # Global Average Pooling
        return self.head(x)


def get_vit_no_all(num_classes: int) -> ViT_NoAll:
    """Return the minimal ViT with all major components removed.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        ViT_NoAll: Attention-only model without pos embed, cls token, or MLP.
    """
    return ViT_NoAll(num_classes=num_classes)
