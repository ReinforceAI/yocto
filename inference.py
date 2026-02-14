"""
YOCTO Inference
===============

Generate text with Yocto, the world's smallest language model family.

Usage:
    python inference.py --prompt "Once upon a time"
    python inference.py --interactive
"""

import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import argparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer as HFTokenizer

from models.yocto import Yocto, KVCache
from utils.config import ModelConfig


class Tokenizer:
    """Tokenizer wrapper for HuggingFace tokenizers."""
    
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    
    def __init__(self, path: str):
        self.tokenizer = HFTokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()
    
    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text).ids
    
    def decode(self, token_ids: list) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def load_model(checkpoint_path: str, device: str) -> tuple:
    """Load model and config from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = ModelConfig.from_dict(checkpoint['config'])
    model = Yocto(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def generate_stream(
    model: Yocto,
    tokenizer: Tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
):
    """Generate tokens with KV-cache, yielding each token's text."""
    
    model.eval()
    device = input_ids.device
    
    # Initialize cache
    kv_cache = KVCache()
    kv_cache.reset(len(model.blocks))
    
    # Prefill: process entire prompt
    with torch.no_grad():
        logits, _ = model(input_ids, kv_cache=kv_cache, use_cache=True)
    
    # Track all generated tokens for repetition penalty
    all_tokens = input_ids[0].tolist()
    
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :]
        
        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(all_tokens):
                if token_id < next_logits.size(-1):
                    next_logits[0, token_id] /= repetition_penalty
        
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
        
        token_id = next_token.item()
        all_tokens.append(token_id)
        
        # Yield decoded token
        token_text = tokenizer.decode([token_id])
        yield token_text
        
        # Check EOS
        if token_id == tokenizer.EOS_ID:
            break
        
        # Forward with cache (only new token)
        with torch.no_grad():
            logits, _ = model(next_token, kv_cache=kv_cache, use_cache=True)


def interactive_mode(model, tokenizer, device, args):
    """Run interactive generation with streaming output."""
    
    print("\n" + "=" * 50)
    print("YOCTO — Interactive Mode (with KV-Cache)")
    print("=" * 50)
    print("Enter a prompt to generate text. Type 'quit' to exit.\n")
    
    while True:
        try:
            prompt = input(">>> ").strip()
            
            if not prompt:
                continue
            if prompt.lower() == 'quit':
                print("Goodbye!")
                break
            
            print()
            print(prompt, end='', flush=True)
            
            input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
            
            import time
            start_time = time.perf_counter()
            token_count = 0
            
            for token_text in generate_stream(
                model, tokenizer, input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            ):
                print(token_text, end='', flush=True)
                token_count += 1
            
            elapsed = time.perf_counter() - start_time
            tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
            
            print(f"\n\n[{token_count} tokens in {elapsed:.2f}s — {tokens_per_sec:.1f} tokens/sec]\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Yocto Inference")
    
    parser.add_argument('--checkpoint', type=str, default='ckpt/model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='ckpt/tokenizer.json',
                        help='Path to tokenizer')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Prompt for generation')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive mode')
    
    # Generation settings
    parser.add_argument('--max-tokens', type=int, default=150)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)
    
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'])
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    tokenizer = Tokenizer(args.tokenizer)
    print(f"Model loaded: {model.num_params:,} parameters ({device})")
    
    # Run
    if args.interactive or args.prompt is None:
        interactive_mode(model, tokenizer, device, args)
    else:
        print()
        print(args.prompt, end='', flush=True)
        
        input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)
        
        import time
        start_time = time.perf_counter()
        token_count = 0
        
        for token_text in generate_stream(
            model, tokenizer, input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ):
            print(token_text, end='', flush=True)
            token_count += 1
        
        elapsed = time.perf_counter() - start_time
        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
        
        print(f"\n\n[{token_count} tokens in {elapsed:.2f}s — {tokens_per_sec:.1f} tokens/sec]")


if __name__ == "__main__":
    main()