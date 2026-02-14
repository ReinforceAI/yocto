"""
Yocto Model Configuration
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ModelConfig:
    """
    Model architecture configuration.
    
    Constraints for Unified Attention:
        - embed_dim must be divisible by 3
        - third_dim must be divisible by num_heads
        - component_head_dim must be even (for RoPE)
    """
    
    vocab_size: int = 4000
    embed_dim: int = 72
    num_heads: int = 3
    num_layers: int = 4
    ffn_dim: int = 288
    max_seq_len: int = 512
    dropout: float = 0.1
    
    # Derived values
    head_dim: int = field(init=False)
    third_dim: int = field(init=False)
    component_head_dim: int = field(init=False)
    
    def __post_init__(self):
        # Validate
        assert self.embed_dim % 3 == 0, "embed_dim must be divisible by 3"
        
        # Compute derived values
        self.head_dim = self.embed_dim // self.num_heads
        self.third_dim = self.embed_dim // 3
        self.component_head_dim = self.third_dim // self.num_heads
        
        assert self.component_head_dim % 2 == 0, "component_head_dim must be even"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ffn_dim': self.ffn_dim,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        # Only use keys that ModelConfig accepts
        valid_keys = {'vocab_size', 'embed_dim', 'num_heads', 'num_layers', 
                      'ffn_dim', 'max_seq_len', 'dropout'}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)