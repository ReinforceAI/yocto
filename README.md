# ATTENTION FIELDS

**Unified Projections for Efficient Language Models**

> *We introduce Yocto, a 484K parameter language model that reduces attention parameters by 67% while achieving better perplexity than models 2-4× larger. The key insight: Q, K, V projections share structure and can be unified into a single projection.*

---

# Abstract

Standard transformer attention uses three separate projections (Q, K, V), each with d² parameters. We show this is redundant.

We introduce **Unified Attention**: a single projection whose output splits into [seeking|offering|content] bands. Through training, these bands learn the functions of Q, K, and V respectively—but with **67% fewer attention parameters**.

Results:
- **484,272 total parameters** (1.85 MB at float32, <1 MB quantized)
- **5.7% of parameters in attention** (vs ~25% in standard transformers)
- **9.58 validation perplexity** on TinyStories (matching models 2-4× larger)
- **Geometric preservation**: Controlled experiments show Berry phase and layer orthogonality within 2% of standard attention

We interpret our findings through wave physics: vectors are waveforms, weight matrices are fields, and projection computes amplitude resonance with phase alignment. This interpretation predicted unification would work—the three projections share structure because they transform the same input and optimize the same objective.

The physics of attention is simpler than standard architectures suggest.

---

# 1. Introduction

Transformer attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

where Q = W_Q·x, K = W_K·x, V = W_V·x are separate linear projections. This requires **3d² attention parameters per layer**.

But why three separate matrices? The original paper [1] offered no theoretical justification. It worked, so the field adopted it. Nine years later, we still use three matrices.

**Our question**: What is the minimal parameterization for attention?

**Our finding**: One matrix suffices. A single projection, split into three bands, achieves the same geometric properties with 67% fewer attention parameters—and *better* perplexity.

### Contributions

1. **Unified Attention**: Single projection → [seeking|offering|content] bands
2. **67% reduction** in attention parameters (from 3d² to d²)
3. **Improved perplexity** (9.58 vs ~10-12 baseline)
4. **Geometric verification** via Berry phase and orthogonality measurements
5. **Wave-field interpretation** explaining why unification works

---

# 2. Theory: Why Unification Should Work

## 2.1 Vectors as Waveforms

Every vector is a waveform with d dimensions. Each dimension carries:
- **Amplitude**: |vᵢ| — activation strength
- **Phase**: sign(vᵢ) — positive or negative direction

The dot product between vectors is **wave interference**:

$$\mathbf{x} \cdot \mathbf{w} = \sum_i |x_i| \cdot |w_i| \cdot \text{sign}(x_i) \cdot \text{sign}(w_i)$$

Same sign → constructive interference (positive contribution)
Opposite sign → destructive interference (negative contribution)

## 2.2 Why Q, K, V Share Structure

The three projections are not independent:

1. **Same input**: All three transform x
2. **Same objective**: All three optimize the same loss
3. **Coupled function**: Q must find what K offers

If Q, K, V share underlying structure, learning them jointly (one matrix) should be more efficient than learning them separately (three matrices). The single matrix acts as an implicit regularizer.

**Prediction**: Unified projection will match or exceed standard attention with fewer parameters.

---

# 3. Architecture

## 3.1 Unified Attention

```
Standard:     Q = W_Q·x,  K = W_K·x,  V = W_V·x     [3d² params]
Unified:      u = W·x,    Q = u[:d/3], K = u[d/3:2d/3], V = u[2d/3:]   [d² params]
```

We apply Rotary Position Embedding (RoPE) to Q and K bands but **not** to V. Position affects *routing* (who attends to whom) but not *content* (what information transfers).

## 3.2 Model Configuration

| Component | Value |
|-----------|-------|
| Embedding dimension | 72 |
| Layers | 4 |
| Attention heads | 3 |
| FFN hidden | 288 |
| Vocabulary | 4,000 |
| Context length | 512 |
| **Total parameters** | **484,272** |

### Parameter Distribution

| Component | Parameters | Share |
|-----------|-----------|-------|
| Embeddings | 288,000 | 59.5% |
| **Attention** | **27,648** | **5.7%** |
| FFN | 166,464 | 34.4% |
| Other | 2,160 | 0.4% |

Standard transformers allocate ~25% to attention. Ours uses **5.7%**.

---

# 4. Experiments

## 4.1 Setup

- **Data**: TinyStories [3]
- **Training**: 82,000 steps, batch size 64, AdamW
- **Learning rate**: 1e-3 → 1e-4 (cosine decay)

## 4.2 Results

### Perplexity Comparison

| Model | Total Params | Attention Share | Val PPL |
|-------|-------------|-----------------|---------|
| **Ours (Unified)** | **484K** | **5.7%** | **9.58** |
| TinyStories-1M | 1M | ~25% | ~10-12 |
| seangoedecke [4] | 1.8M | ~25% | ~9.6 |

With **52% fewer total parameters** than TinyStories-1M, we achieve better perplexity. With **73% fewer parameters** than seangoedecke, we match their perplexity.

### Generation Quality

| Metric | Score |
|--------|-------|
| Quality score | 99.6/100 |
| Vocabulary diversity | 67.0% |
| 2-gram repetition | 4.7% |
| Story elements | 68.8% |

**Example** (prompt: "Once upon a time"):

> Once upon a time there was a little girl named Lily. She loved to run and run, but one day she didn't have any friends to play with.
>
> Lily went home and bought her favorite toy to play. When she woke up, she saw something in the closet. It was a ball of yarn! She played with it all night long.

Named characters, temporal progression, narrative coherence—from 484K parameters.

## 4.3 Geometric Verification

Does unified attention preserve the geometric properties of standard attention? We ran controlled experiments comparing attention mechanisms on identical architectures (6-layer transformers, embed_dim=126, trained on synthetic language tasks).

### Berry Phase

Berry phase measures accumulated rotation through layers—the "geometric memory" of the path through representation space.

| Attention Type | Berry Phase | vs Baseline |
|---------------|-------------|-------------|
| Standard Q/K/V | 135.23° | 100% |
| **Unified [seek\|offer\|content]** | **137.32°** | **101.5%** |

Within 2%: **geometric path preserved**.

### Layer Orthogonality

Average angle between consecutive layer representations:

| Attention Type | Mean Angle |
|---------------|------------|
| Standard Q/K/V | 22.54° |
| **Unified** | **22.89°** |

Within 2%: **rotation structure preserved**.

### Interpretation

These controlled experiments demonstrate that unified attention traces an equivalent geometric path through representation space. The 67% parameter reduction does not distort the fundamental geometry—it removes redundancy while preserving structure.

The Yocto model (4 layers, embed_dim=72) applies this same unified architecture to TinyStories, achieving 9.58 perplexity with 5.7% attention parameters.

---

# 5. Analysis

## 5.1 Why Does It Work?

**Shared structure**: Q, K, V transform the same input for the same objective. Separate matrices learn redundant structure. A unified matrix learns it once.

**Implicit regularization**: Fewer parameters may prevent overfitting. Our *improved* perplexity (9.58 vs ~10-12) supports this.

## 5.2 What Does 5.7% Attention Mean?

Our parameter distribution (59.5% embeddings, 5.7% attention, 34.4% FFN) suggests:

- **Attention is routing**: It decides WHERE information flows, not what it becomes
- **Embeddings carry meaning**: Most capacity represents tokens well
- **FFN is essential**: Nonlinear transformation cannot be reduced

## 5.3 Limitations

- **Domain**: TinyStories is constrained. General language may need more capacity.
- **Scale**: 484K tested. Behavior at 100M+ unknown.
- **Context**: 512 tokens tested. Long-context behavior unexplored.

---

# 6. Related Work

**Multi-Query Attention** [5]: Shares K, V across heads. We go further—unifying Q, K, V into a single projection.

**LoRA** [6]: Reduces parameters post-training via low-rank adaptation. We reduce architecturally, during training.

**Efficient Attention**: Linear attention and sparse attention reduce O(n²) complexity. We reduce parameters while keeping full attention.

**F-Net** [7]: Replaces attention with Fourier transforms. Loses differential weighting—all tokens mixed identically, cannot focus on relevant context.

**Mamba** [8]: Replaces attention with selective state spaces. Loses Q/K asymmetry—"what I seek" merges with "what I offer" in a single state.

**Ours**: Preserves both differential weighting and asymmetry. Removes only the redundant parameterization.

---

# 7. Conclusion

We asked: What is the minimal parameterization for attention?

**Answer**: One projection suffices. The three matrices Q, K, V share structure that a unified projection captures more efficiently.

**Results**:
- 67% fewer attention parameters
- 5.7% of model in attention (vs 25% standard)
- Better perplexity than larger models
- Geometric properties preserved within 2%

**Implication**: Standard attention is over-parameterized. The wave-field interpretation predicted this—and experiments confirm it.

---

# References

[1] Vaswani et al., "Attention Is All You Need," NeurIPS 2017.

[2] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021.

[3] Eldan & Li, "TinyStories: How Small Can Language Models Be?," 2023.

[4] Goedecke, "Training a Language Model on a Laptop," 2024.

[5] Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," 2019.

[6] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.

[7] Lee-Thorp et al., "FNet: Mixing Tokens with Fourier Transforms," NAACL 2022.

[8] Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2023.

---

# Appendix: Implementation

```python
class UnifiedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        self.third = embed_dim // 3
        self.W_unified = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out = nn.Linear(self.third, embed_dim, bias=False)
        self.rope = RotaryPositionEmbedding(self.third // num_heads)
        
    def forward(self, x):
        # Single projection → three bands
        u = self.W_unified(x)
        seeking, offering, content = u.split(self.third, dim=-1)
        
        # RoPE on seeking/offering only (not content)
        cos, sin = self.rope(x)
        seeking, offering = apply_rope(seeking, offering, cos, sin)
        
        # Standard attention computation
        out = F.scaled_dot_product_attention(
            seeking, offering, content, is_causal=True
        )
        return self.W_out(out)
```

---

## YOCTO — *The World's Smallest Language Model*

**484,272 Parameters · 946 KB (fp16) · 67% Less Attention · Open Source**

### Quick Start

```bash
git clone https://github.com/reinforceai/yocto
cd yocto
pip install -r requirements.txt
python inference.py --prompt "Once upon a time"
```

### Citation

If you use this work, please cite:

```bibtex
@misc{deshwal2026yocto,
  title={Attention Fields: Unified Projections for Efficient Language Models},
  author={Deshwal, Viraj},
  year={2026},
  url={https://www.reinforceai.com/yocto},
  howpublished={\url{https://github.com/reinforceai/yocto}}
}
```

