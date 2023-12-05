"""
Código copiado do whisper - OpenAI
- https://github.com/openai/whisper
A classe `Supressor` foi adaptada para o projeto.
"""


from typing import Iterator, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(
        self,
        x: torch.Tensor,
        xa: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        xa: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class Supressor(nn.Module):
    def __init__(self, n_state: int = 512, n_head: int = 8, n_layer: int = 6, n_ctx: int = 1500):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(80, n_state, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(n_state, n_state, 3, padding=1, stride=2),
            nn.GELU(),
        )
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.encoder: Iterator[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_norm_encoder = nn.LayerNorm(n_state)
        self.decoder: Iterator[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln_norm_decoder = nn.LayerNorm(n_state)
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose1d(n_state, n_state, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(n_state, 80, 3, padding=1, stride=2, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = (x + self.positional_embedding).to(x.dtype)
        for encoder in self.encoder:
            x = encoder(x)
        x = self.ln_norm_encoder(x)
        for decoder in self.decoder:
            x = decoder(x)
        x = self.ln_norm_decoder(x)
        x = x.permute(0, 2, 1)
        x = self.trans_conv(x)
        return x

    def from_pretrained(self):
        pass

    def suppress(self, file: str) -> torch.Tensor:
        from dataset.prepare import prepare
        from datasets import Audio, Dataset

        dataset = Dataset.from_dict({"input": file})
        dataset = dataset.cast_column("input", Audio(16000))
        dataset = dataset.map(prepare)
        dataset.set_format(type="torch", columns=["input"])

        x = next(iter(dataset))["input"]
        x = x.unsqueeze(0)
        x = x.to(self.device)

        y_hat = self(x)

        return y_hat
