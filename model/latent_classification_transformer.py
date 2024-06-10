import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import os

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW
from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from model.x_transformer import AbsolutePositionalEmbedding, Encoder


# Helper Functions

# def groupby_prefix_and_trim(prefix, d):
#     kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
#     kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
#     return kwargs_without_prefix, kwargs

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def exists(val):
    return val is not None

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AttentionPool(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.to_q = nn.Linear(embed_dim, embed_dim, bias = False)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias = False)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias = False)
        self.to_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask = None):
        
        bsz, seq_len, embed_dim = x.size()

        q = x.mean(dim=1).unsqueeze(dim=1)
        query = self.to_q(q)
        key, value = self.to_k(x), self.to_v(x)

        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (query, key, value))
        similarity = einsum('b h i d, b h j d -> b h i j', query, key) * (int(embed_dim) ** -0.5)

        if exists(mask):
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            similarity.masked_fill_(~mask, -torch.finfo(similarity.dtype).max)

        attn = similarity.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, value)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out).squeeze(dim=1)

class LatentClassificationTransformer(nn.Module):
    def __init__(
        self,
        tx_dim,
        tx_depth,
        heads,
        latent_dim = None,
        max_seq_len=64,
        dropout = 0.1,
        scale_shift = False,
        num_classes=0,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.scale_shift = scale_shift
        self.num_classes = num_classes

        self.max_seq_len = max_seq_len

        # time embeddings

        sinu_pos_emb = SinusoidalPosEmb(tx_dim)

        time_emb_dim = tx_dim*4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(tx_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_pos_embed_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, tx_dim)
            )

        self.pos_emb = AbsolutePositionalEmbedding(tx_dim, max_seq_len)

        self.encoder = Encoder(
            dim=tx_dim,
            depth=tx_depth,
            heads=heads,
            attn_dropout=dropout,    # dropout post-attention
            ff_dropout=dropout,       # feedforward dropout
            rel_pos_bias=True,
            ff_glu=True,
            time_emb_dim=tx_dim*4 if self.scale_shift else None,
        )
        
        self.input_proj = nn.Linear(latent_dim, tx_dim)
        self.norm = nn.LayerNorm(tx_dim)
        self.output_proj = nn.Linear(tx_dim, latent_dim)

        self.attention_pool = AttentionPool(latent_dim, heads)
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

        init_zero_(self.output_proj)

    def forward(self, x, mask, time):
        """
        x: input, [batch, length, latent_dim]
        mask: bool tensor where False indicates masked positions, [batch, length] 
        time: timestep, [batch]
        """

        time_emb = self.time_mlp(time)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')
        pos_emb = self.pos_emb(x)

        tx_input = self.input_proj(x) + pos_emb + self.time_pos_embed_mlp(time_emb)

        x = self.encoder(tx_input, mask=mask, time_emb=time_emb)
        x = self.norm(x)

        x = self.output_proj(x)
        x = self.attention_pool(x)

        logits = self.classification_head(x)

        return logits
    
    def compute_metrics(self, logits, labels):
        """
        logits: [batch, num_classes]
        labels: [batch, 1]
        """
        logits = logits
        labels = labels
        
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return acc, loss
