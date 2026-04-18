"""Ablation variant S4: ViT without MLP block in Transformer encoder.

Removes the feed-forward MLP sub-layer from each Transformer block,
leaving only the multi-head self-attention and layer normalization.
"""

import torch
import torch.nn as nn
from models.vit import PatchEmbedding, Attention


class TransformerBlock_NoMLP(nn.Module):
    """Transformer block with attention only — MLP sub-layer removed.

    Args:
        dim (int): Feature dimension. Defaults to ``768``.
        num_heads (int): Number of attention heads. Defaults to ``12``.
    """

    def __init__(self, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads=num_heads)
        # norm2 and MLP intentionally omitted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x


class ViT_NoMLP(nn.Module):
    """ViT-B/16 with MLP blocks removed from all Transformer encoders.

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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks    = nn.ModuleList([TransformerBlock_NoMLP(dim=embed_dim, num_heads=num_heads) for _ in range(depth)])
        self.norm      = nn.LayerNorm(embed_dim)
        self.head      = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


def get_vit_no_mlp(num_classes: int) -> ViT_NoMLP:
    """Return ViT-B/16 without MLP blocks.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        ViT_NoMLP: Model with MLP sub-layers removed.
    """
    return ViT_NoMLP(num_classes=num_classes)
