"""
YOCTO: World's Smallest Language Model Family
======================================

Unified Attention architecture with KV-Cache for fast generation.

Paper: "Attention Fields: Unified Projections for Efficient Language Models"
Author: Viraj Deshwal (viraj@reinforceai.com)
https://www.reinforceai.com/yocto
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import ModelConfig


# ==============================================================================
# ROTARY POSITION EMBEDDING (RoPE)
# ==============================================================================

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding with position offset support for KV-cache."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len
    
    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos/sin for positions [offset, offset + seq_len).
        
        Args:
            x: input tensor (for dtype)
            seq_len: number of positions needed
            offset: starting position (for KV-cache)
        """
        total_len = offset + seq_len
        if total_len > self.max_seq_len:
            self._build_cache(total_len)
        
        return (
            self.cos_cached[offset:offset + seq_len].to(x.dtype),
            self.sin_cached[offset:offset + seq_len].to(x.dtype)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    seeking: torch.Tensor,
    offering: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    seeking_rotated = (seeking * cos) + (rotate_half(seeking) * sin)
    offering_rotated = (offering * cos) + (rotate_half(offering) * sin)
    
    return seeking_rotated, offering_rotated


# ==============================================================================
# KV-CACHE
# ==============================================================================

class KVCache:
    """Simple KV-Cache for autoregressive generation."""
    
    def __init__(self):
        self.cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
    
    def reset(self, num_layers: int):
        """Reset cache for new generation."""
        self.cache = [None] * num_layers
    
    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached K, V for a layer."""
        return self.cache[layer_idx]
    
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Update cache with new K, V."""
        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (k, v)
        else:
            cached_k, cached_v = self.cache[layer_idx]
            self.cache[layer_idx] = (
                torch.cat([cached_k, k], dim=2),
                torch.cat([cached_v, v], dim=2)
            )
    
    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        if self.cache and self.cache[0] is not None:
            return self.cache[0][0].shape[2]
        return 0


# ==============================================================================
# UNIFIED ATTENTION WITH KV-CACHE
# ==============================================================================

class UnifiedAttention(nn.Module):
    """
    Unified [Seeking | Offering | Content] Attention with KV-Cache.
    
    Single projection splits into three bands:
        embed_dim -> [seeking | offering | content]
    
    Achieves 67% parameter reduction vs standard Q/K/V.
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.third_dim = config.third_dim
        self.component_head_dim = config.component_head_dim
        self.dropout_p = config.dropout
        
        self.W_unified = nn.Linear(config.embed_dim, config.third_dim * 3, bias=False)
        self.W_output = nn.Linear(config.third_dim, config.embed_dim, bias=False)
        
        self.rope = RotaryPositionEmbedding(
            dim=config.component_head_dim,
            max_seq_len=config.max_seq_len
        )
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Unified projection -> split into three bands
        unified = self.W_unified(x)
        seeking = unified[..., :self.third_dim]
        offering = unified[..., self.third_dim:2*self.third_dim]
        content = unified[..., 2*self.third_dim:]
        
        # Reshape for multi-head attention
        seeking = seeking.view(batch_size, seq_len, self.num_heads, self.component_head_dim).transpose(1, 2)
        offering = offering.view(batch_size, seq_len, self.num_heads, self.component_head_dim).transpose(1, 2)
        content = content.view(batch_size, seq_len, self.num_heads, self.component_head_dim).transpose(1, 2)
        
        # Get position offset from cache
        offset = kv_cache.seq_len if kv_cache is not None else 0
        
        # Apply RoPE with correct positions
        cos, sin = self.rope(seeking, seq_len, offset=offset)
        seeking, offering = apply_rotary_pos_emb(seeking, offering, cos, sin)
        
        # Handle KV-cache
        if kv_cache is not None:
            cached = kv_cache.get(self.layer_idx)
            if cached is not None:
                cached_k, cached_v = cached
                offering = torch.cat([cached_k, offering], dim=2)
                content = torch.cat([cached_v, content], dim=2)
            
            if use_cache:
                # Cache only the NEW k, v (before concat)
                new_k = unified[..., self.third_dim:2*self.third_dim]
                new_v = unified[..., 2*self.third_dim:]
                new_k = new_k.view(batch_size, seq_len, self.num_heads, self.component_head_dim).transpose(1, 2)
                new_v = new_v.view(batch_size, seq_len, self.num_heads, self.component_head_dim).transpose(1, 2)
                # Apply RoPE to new_k for caching
                new_k, _ = apply_rotary_pos_emb(new_k, new_k, cos, sin)
                kv_cache.update(self.layer_idx, new_k, new_v)
        
        # Attention
        is_causal = (kv_cache is None) or (kv_cache.seq_len == 0)
        
        output = F.scaled_dot_product_attention(
            seeking, offering, content,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.third_dim)
        return self.W_output(output)


# ==============================================================================
# FEED-FORWARD NETWORK
# ==============================================================================

class FeedForward(nn.Module):
    """Feed-Forward Network with GELU activation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ==============================================================================
# TRANSFORMER BLOCK
# ==============================================================================

class TransformerBlock(nn.Module):
    """Pre-norm Transformer Block with KV-Cache support."""
    
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.attn = UnifiedAttention(config, layer_idx=layer_idx)
        self.ffn = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + self.ffn(self.norm2(x))
        return x


# ==============================================================================
# YOCTO MODEL
# ==============================================================================

class Yocto(nn.Module):
    """
    Yocto: World's Smallest Language Model.
    
    Uses Unified Attention with KV-Cache for fast generation.
    484K parameters, 946 KB (fp16).
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        self.output = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional KV-cache.
        
        Args:
            input_ids: [batch, seq_len]
            targets: [batch, seq_len] (optional, for training)
            kv_cache: KVCache object (for generation)
            use_cache: whether to update cache
        """
        x = self.dropout(self.token_embedding(input_ids))
        
        for block in self.blocks:
            x = block(x, kv_cache=kv_cache, use_cache=use_cache)
        
        x = self.norm(x)
        
        if targets is not None:
            logits = self.output(x)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=0
            )
            return logits, loss
        else:
            return self.output(x), None
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Generate with KV-cache for fast autoregressive decoding."""
        self.eval()
        
        # Initialize cache
        kv_cache = KVCache() if use_cache else None
        if kv_cache:
            kv_cache.reset(len(self.blocks))
        
        # Process prompt (prefill)
        logits, _ = self(input_ids, kv_cache=kv_cache, use_cache=use_cache)
        
        for _ in range(max_new_tokens):
            # Get logits for last position
            next_logits = logits[:, -1, :]
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(input_ids.size(0)):
                    for token_id in set(input_ids[i].tolist()):
                        if token_id < next_logits.size(-1):
                            next_logits[i, token_id] /= repetition_penalty
            
            # Temperature
            next_logits = next_logits / temperature
            
            # Top-k
            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, k)
                next_logits = torch.where(
                    next_logits < values[:, -1:],
                    torch.full_like(next_logits, float('-inf')),
                    next_logits
                )
            
            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                next_logits = next_logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            
            # Forward with cache (only process new token)
            logits, _ = self(next_token, kv_cache=kv_cache, use_cache=use_cache)
        
        return input_ids