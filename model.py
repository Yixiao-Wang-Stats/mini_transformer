"""Character-level decoder Transformer used by the homework."""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class Config:
    block_size: int = 64
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False

    def __post_init__(self):
        for name in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_head must evenly divide n_embd")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # A buffer moves with the model and is saved in checkpoints, but is not trained.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, L, C = x.size()
        # Step 1 (b.i, TODO): project inputs with self.c_attn and split into Q, K, V.
        Q, K, V = ..., ..., ...

        # Step 2 (b.ii, TODO): reshape and transpose Q, K, V into multiple heads.
        Q = ...
        K = ...
        V = ...
        # Uncomment this shape check after completing Step 2.
        # assert Q.shape == (B, self.n_head, L, C // self.n_head)

        # Step 3 (b.iii, TODO): compute scores and scale by sqrt(head dimension).
        att = ...

        # Step 4 (b.iv, TODO): use self.mask to block attention to future positions.
        att = ...

        # Step 5 (b.v, TODO): apply softmax along the key dimension and combine V.
        att = ...
        # Uncomment the supplied dropout after applying softmax.
        # att = self.attn_dropout(att)
        y = ...
        # assert y.shape == (B, self.n_head, L, C // self.n_head)

        # Supplied output: uncomment to concatenate heads and apply output projection.
        # y = y.transpose(1, 2).contiguous().view(B, L, C)
        # y = self.resid_dropout(self.c_proj(y))

        # This placeholder permits the untrained baseline in part (a).
        # Replace it with `return y` when all five steps are complete.
        # train() detects these placeholders and refuses to train until they are removed.
        return torch.randn_like(x)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin_layer1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.lin_layer2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.lin_layer2(self.gelu(self.lin_layer1(x))))


class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([AttentionBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("idx must have shape (batch, nonempty sequence)")
        _, length = idx.size()
        if length > self.config.block_size:
            raise ValueError(f"Sequence length {length} exceeds block_size {self.config.block_size}")
        positions = torch.arange(length, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(positions))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        if not 0 < block_size <= self.config.block_size:
            raise ValueError("New block_size must be positive and cannot exceed the current value")
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size].clone())
        self.transformer.wpe.num_embeddings = block_size
        for block in self.transformer.h:
            block.attn.mask = block.attn.mask[:, :, :block_size, :block_size].clone()

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        parameters = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in parameters if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in parameters if p.dim() < 2], "weight_decay": 0.0},
        ]
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **({"fused": True} if fused else {}))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("top_k must be a positive integer or None")
        if idx.ndim != 2 or idx.size(1) == 0:
            raise ValueError("Generation requires a nonempty prompt")
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probabilities = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probabilities, num_samples=1)), dim=1)
        return idx
