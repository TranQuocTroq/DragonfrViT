"""Vision Transformer (ViT-B/16) implementation from scratch.

Implements the standard ViT-Base architecture as described in:
    Dosovitskiy et al., "An Image is Worth 16x16 Words:
    Transformers for Image Recognition at Scale", ICLR 2021.

All components are built using only PyTorch base modules (``nn.Module``)
without relying on any pre-trained weights or external ViT libraries.

Example:
    >>> model = get_vit_model(num_classes=5)
    >>> x = torch.randn(2, 3, 224, 224)
    >>> logits = model(x)   # shape: [2, 5]
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and project to embedding space.

    Uses a ``Conv2d`` layer with ``kernel_size == stride == patch_size`` to
    efficiently extract and linearly project patches in a single operation.

    Args:
        img_size (int): Input image size (assumed square). Defaults to ``224``.
        patch_size (int): Side length of each square patch. Defaults to ``16``.
        in_channels (int): Number of input image channels. Defaults to ``3``.
        embed_dim (int): Patch embedding dimension. Defaults to ``768``.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image patches to embedding vectors.

        Args:
            x (torch.Tensor): Input tensor of shape ``[B, C, H, W]``.

        Returns:
            torch.Tensor: Patch embeddings of shape ``[B, num_patches, embed_dim]``.
        """
        x = self.proj(x)       # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)       # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class Attention(nn.Module):
    """Multi-head self-attention module.

    Implements scaled dot-product attention with multiple heads as described
    in Vaswani et al. (2017). Q, K, V projections are fused into a single
    linear layer for efficiency.

    Args:
        dim (int): Input and output feature dimension. Defaults to ``768``.
        num_heads (int): Number of attention heads. Defaults to ``12``.
        qkv_bias (bool): Whether to add bias to Q/K/V projections.
            Defaults to ``True``.
        dropout (float): Dropout probability on attention weights and
            output projection. Defaults to ``0.0``.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # Scale factor to prevent dot-product magnitude from exploding
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x (torch.Tensor): Input of shape ``[B, N, C]``.

        Returns:
            torch.Tensor: Output of shape ``[B, N, C]``.
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Position-wise feed-forward network used inside each Transformer block.

    Applies two linear transformations with a GELU activation in between.
    The hidden dimension is typically 4× the input dimension.

    Args:
        in_features (int): Input and output feature dimension.
        hidden_features (int): Intermediate (expanded) dimension.
        act_layer (nn.Module): Activation function class. Defaults to ``nn.GELU``.
        drop (float): Dropout probability. Defaults to ``0.0``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two-layer MLP with GELU activation.

        Args:
            x (torch.Tensor): Input of shape ``[B, N, C]``.

        Returns:
            torch.Tensor: Output of shape ``[B, N, C]``.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block with pre-normalization.

    Follows the ViT convention of applying LayerNorm *before* each sub-layer
    (pre-norm) and using residual connections around both the attention and
    MLP sub-layers.

    Args:
        dim (int): Feature dimension. Defaults to ``768``.
        num_heads (int): Number of attention heads. Defaults to ``12``.
        mlp_ratio (float): MLP hidden dimension multiplier. Defaults to ``4.0``.
        qkv_bias (bool): Add bias to Q/K/V projections. Defaults to ``True``.
        drop (float): MLP dropout probability. Defaults to ``0.0``.
        attn_drop (float): Attention dropout probability. Defaults to ``0.0``.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP sub-layers with residual connections.

        Args:
            x (torch.Tensor): Input of shape ``[B, N, C]``.

        Returns:
            torch.Tensor: Output of shape ``[B, N, C]``.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT-B/16) for image classification.

    Implements the full ViT-Base architecture: patch embedding, learnable
    class token, learnable positional embeddings, 12 Transformer encoder
    blocks, and a linear classification head.

    Args:
        img_size (int): Input image size (square). Defaults to ``224``.
        patch_size (int): Patch size. Defaults to ``16``.
        in_channels (int): Number of input channels. Defaults to ``3``.
        num_classes (int): Number of output classes.
        embed_dim (int): Embedding dimension. Defaults to ``768``.
        depth (int): Number of Transformer blocks. Defaults to ``12``.
        num_heads (int): Number of attention heads. Defaults to ``12``.
        mlp_ratio (float): MLP hidden size multiplier. Defaults to ``4.0``.
        qkv_bias (bool): Add bias to Q/K/V. Defaults to ``True``.
        drop_rate (float): General dropout rate. Defaults to ``0.0``.
        attn_drop_rate (float): Attention dropout rate. Defaults to ``0.0``.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 5,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim

        # Patch embedding layer
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches  # 196 for 224×224, patch=16

        # Learnable class token — aggregates global image representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable positional embeddings — +1 for the class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(p=drop_rate)

        # Stack of Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear and layer-norm weights.

        Args:
            module (nn.Module): Module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full ViT pipeline.

        Args:
            x (torch.Tensor): Input images of shape ``[B, C, H, W]``.

        Returns:
            torch.Tensor: Class logits of shape ``[B, num_classes]``.
        """
        B = x.shape[0]

        # 1. Patch embedding
        x = self.patch_embed(x)                          # [B, N, D]

        # 2. Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)    # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)            # [B, N+1, D]

        # 3. Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Transformer encoder
        for block in self.blocks:
            x = block(x)

        # 5. Extract class token and classify
        x = self.norm(x)
        cls_output = x[:, 0]                             # [B, D]
        return self.head(cls_output)                     # [B, num_classes]


def get_vit_model(num_classes: int) -> VisionTransformer:
    """Build a ViT-B/16 model for the given number of classes.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        VisionTransformer: Initialized ViT-B/16 model.
    """
    return VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
    )
