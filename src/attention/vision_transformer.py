import torch
import einops

from attention import SelfAttention


class TransformerBlock(torch.nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self._attention = SelfAttention(embed_dim)
        self._norm1 = torch.nn.LayerNorm(embed_dim)
        self._norm2 = torch.nn.LayerNorm(embed_dim)

        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        attention_output, attention_weights = self._attention(self._norm1(x))

        x = x + attention_output

        mlp_out = self._mlp(self._norm2(x))

        x = x + mlp_out

        if return_attention:
            return x, attention_weights
        else:
            return x


class VisionTransformer(torch.nn.Module):
    def __init__(
        self,
        number_classes: int,
        embed_dim: int = 8,
        patch_size: int = 4,
        number_layers: int = 8,
    ):
        super().__init__()

        self._patch_embedder = torch.nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            stride=patch_size,
            kernel_size=patch_size,
        )

        self.layers = torch.nn.ModuleList()
        for _ in range(number_layers):
            self.layers.append(TransformerBlock(embed_dim))

        self._classifier = torch.nn.Linear(embed_dim, number_classes)

        self._positional_embedding = torch.nn.Parameter(
            torch.randn(1, 28**2 // patch_size**2 + 1, embed_dim)
        )

        self._cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        patches = self._patch_embedder(x)

        patches = einops.rearrange(patches, "b c h w -> b (h w) c")

        # Add the CLS token to each image patch sequence
        patches = torch.cat(
            [self._cls_token.expand(patches.shape[0], -1, -1), patches], dim=1
        )

        # Add global positional embedding
        patches = patches + self._positional_embedding

        if return_attention:
            attention_weights = {}

        for layer_idx, layer in enumerate(self.layers):
            if return_attention:
                patches, attention = layer(patches, return_attention=True)
                attention_weights[f"layer_{layer_idx}"] = attention
            else:
                patches = layer(patches)

        cls_output = patches[:, 0]

        output = self._classifier(cls_output)

        if return_attention:
            return output, attention_weights
        else:
            return output
