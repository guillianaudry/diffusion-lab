"""U-Net compact pour images (MNIST 28×28, CIFAR 32×32).

Structure « DDPM » classique : blocs résiduels conditionnés par l'embedding de temps
(et de classe), sous-échantillonnage ×2 entre niveaux, attention au goulot,
connexions de saut entre descente et remontée.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import TimeClassEmbedding


def _norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = _norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = _norm(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)  # (B, HW, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2).reshape(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 2),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        emb_dim = base_channels * 4
        self.embed = TimeClassEmbedding(emb_dim, num_classes)

        ch = base_channels * channel_mults[0]
        self.stem = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Descente : on mémorise le nombre de canaux de chaque connexion de saut.
        skip_channels = [ch]
        self.down = nn.ModuleList()
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(ch, out_ch, emb_dim, dropout))
                ch = out_ch
                skip_channels.append(ch)
            if level != len(channel_mults) - 1:
                self.down.append(Downsample(ch))
                skip_channels.append(ch)

        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        # Remontée : chaque bloc consomme une connexion de saut.
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up.append(ResBlock(ch + skip_channels.pop(), out_ch, emb_dim, dropout))
                ch = out_ch
            if level != 0:
                self.up.append(Upsample(ch))
        assert not skip_channels

        self.out = nn.Sequential(_norm(ch), nn.SiLU(), nn.Conv2d(ch, in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.embed(t, y)
        h = self.stem(x)
        skips = [h]
        for layer in self.down:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for layer in self.up:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), emb)
            else:
                h = layer(h)
        return self.out(h)
