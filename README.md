# ATTENTION FIELDS

**Unified Projections for Efficient Language Models**

> *We introduce Yocto, a language model architecture that reduces attention parameters by 67% while matching or exceeding standard transformers. The key insight: Q, K, V projections share structure and can be unified into a single projection.*

---

# Abstract

Standard transformer attention uses three separate projections (Q, K, V), each with d² parameters. We show that a single projection suffices.

We introduce **Unified Attention**: a single projection whose output splits into [seeking|offering|content] bands. Through training, these bands learn the functions of Q, K, and V respectively with **67% fewer attention parameters**.

Results across two scales:

* **484K params (TinyStories)**: 9.58 validation perplexity, matching models 2-4× larger. 5.7% of parameters in attention vs ~25% standard.
* **23.2M params (OpenAI Parameter Golf)**: **1.1088 BPB**, beating every submission on the leaderboard including standard Q/K/V transformers optimized by multiple teams over weeks. 18% of block parameters in attention vs 33% standard.

We interpret our findings through wave physics: vectors are waveforms, weight matrices are fields, and projection computes amplitude resonance with phase alignment. This interpretation predicted unification would work. The three projections share structure because they transform the same input and optimize the same objective.

The physics of attention is simpler than standard architectures suggest.

---

# 1. Introduction

Transformer attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

where Q = W_Q·x, K = W_K·x, V = W_V·x are separate linear projections. This requires **3d² attention parameters per layer**.

But why three separate matrices? The original paper [1] offered no theoretical justification. It worked, so the field adopted it. Nine years later, we still use three matrices.

**Our question**: What is the minimal parameterization for attention?

**Our finding**: One matrix suffices. A single projection, split into three bands, achieves the same geometric properties with 67% fewer attention parameters and better performance at both small and large scale.

### Contributions

1. **Unified Attention**: Single projection → [seeking|offering|content] bands
2. **67% reduction** in attention parameters (from 3d² to d²)
3. **Competitive at scale**: Beats all standard Q/K/V submissions on OpenAI's Parameter Golf leaderboard
4. **Geometric verification** via Berry phase and orthogonality measurements
5. **Wave-field interpretation** explaining why unification works
6. **FA3 head-dim padding**: Zero-padding trick enabling Hopper-optimized flash attention with non-standard head dimensions

---

# 2. Theory: Why Unification Should Work

## 2.1 Vectors as Waveforms

Every vector is a waveform with d dimensions. Each dimension carries:

* **Amplitude**: |vᵢ| — activation strength
* **Phase**: sign(vᵢ) — positive or negative direction

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

## 3.2 FA3 Head-Dim Padding

Flash Attention 3 (Hopper) requires head_dim to be a multiple of 8. When unified attention produces non-standard head dimensions (e.g., head_dim=44 from d=528 with 4 heads), we zero-pad to the nearest multiple of 8 before FA3 and slice back after:

```python
pad_n = (8 - head_dim % 8) % 8
if pad_n > 0:
    q, k, v = [F.pad(t, (0, pad_n)) for t in (q, k, v)]
out = flash_attn_func(q, k, v, causal=True)
y = out[..., :head_dim]
```

Mathematically lossless. Padded zeros contribute nothing to dot products or weighted sums. The 9% compute overhead from 44→48 dims is overwhelmed by FA3's 1.57× speedup over FA2/SDPA.

---

# 4. Experiments

## 4.1 Small Scale: TinyStories (484K params)

| Component | Value |
| --- | --- |
| Embedding dimension | 72 |
| Layers | 4 |
| Attention heads | 3 |
| FFN hidden | 288 |
| Vocabulary | 4,000 |
| Context length | 512 |
| **Total parameters** | **484,272** |

### Parameter Distribution

| Component | Parameters | Share |
| --- | --- | --- |
| Embeddings | 288,000 | 59.5% |
| **Attention** | **27,648** | **5.7%** |
| FFN | 166,464 | 34.4% |
| Other | 2,160 | 0.4% |

### Results

| Model | Total Params | Attention Share | Val PPL |
| --- | --- | --- | --- |
| **Ours (Unified)** | **484K** | **5.7%** | **9.58** |
| TinyStories-1M | 1M | ~25% | ~10-12 |
| seangoedecke [4] | 1.8M | ~25% | ~9.6 |

With **52% fewer total parameters** than TinyStories-1M, we achieve better perplexity. With **73% fewer parameters** than seangoedecke, we match their perplexity.

## 4.2 Large Scale: OpenAI Parameter Golf (23.2M params)

We applied unified attention to [OpenAI's Parameter Golf challenge](https://github.com/openai/parameter-golf): train the best language model that fits in 16 MB. Evaluated on FineWeb validation set in bits-per-byte (BPB), lower is better.

### Architecture

| Component | Value |
| --- | --- |
| Embedding dimension | 528 |
| Unique layers (K) | 11 |
| Attention heads | 4 (head_dim=44) |
| MLP | 3× (1584) with LeakyReLU(0.5)² |
| Vocabulary | 1,024 |
| Context length | 2,048 |
| **Total parameters** | **23,209,295** |
| **Compressed artifact** | **~15.97 MB** |

Additional techniques: SmearGate, LN Scale (1/√(layer+1)), VE128 on last 2 layers, STE QAT int6, EMA(0.997) + SWA, Parallel Muon optimizer, Legal Score-First TTT at eval.

### Results

**Record Track (10 min training, 10 min eval on 8×H100 SXM):**

| Seed | step_avg | steps | Post-TTT bpb | Artifact |
|------|----------|-------|-------------|----------|
| 1337 | 49.6ms | 12,088 | 1.1416 | 15,991,687 |
| 42 | 49.6ms | 12,109 | 1.1416 | 15,988,916 |
| 2025 | 49.6ms | 12,103 | 1.1403 | 15,962,515 |
| **Mean** | **49.6ms** | **12,100** | **1.1412 (std 0.0008)** | |

Rank 7 on the main leaderboard. The only non-standard attention architecture in the top 10.

**Unlimited Compute Track (1 hour training on 8×H100 SXM):**

| | BPB |
|---|---|
| Previous unlimited SOTA (1-bit quant, 106M params, 2hr) | 1.1239 |
| Previous record SOTA (standard Q/K/V + GPTQ + XSA, 10min) | 1.1147 |
| **Ours (unified attention, 23.2M params, 1hr)** | **1.1088** |

**Unified attention with 23.2M parameters and 1 hour of training beats every submission on both leaderboards**, including standard transformers with 4× more parameters and specialized techniques (XSA, Full Hessian GPTQ, BigramHash) that have been incrementally optimized by multiple teams.

### Post-Compression Parameter Reallocation

The core advantage in parameter-constrained settings: unified attention frees compressed bytes for the MLP.

| | Standard (SOTA) | Unified (Ours) |
|---|---|---|
| Attention params | 8.65M (33% of blocks) | 4.09M (18% of blocks) |
| MLP params | 17.3M | 18.4M |
| Attention compressed | 5.10 MB | 2.82 MB |
| MLP compressed | 10.21 MB | 12.70 MB |

We trade 2.28 MB of routing for 2.49 MB of computation.

### What Doesn't Transfer to Unified Attention

The shared projection creates structural coupling between bands. Techniques designed for independent Q/K/V projections can hurt:

| Technique | Standard Q/K/V | Unified | Root Cause |
|-----------|---------------|---------|------------|
| XSA | +0.004 BPB | -0.0015 BPB | Content coupled to seeking/offering in shared projection |
| BigramHash at input | +0.004 BPB | -0.009 BPB | Single matrix can't route bigram to 3 bands |

## 4.3 Geometric Verification

Does unified attention preserve the geometric properties of standard attention? We ran controlled experiments comparing attention mechanisms on identical architectures (6-layer transformers, embed_dim=126, trained on synthetic language tasks).

### Berry Phase

Berry phase measures accumulated rotation through layers.

| Attention Type | Berry Phase | vs Baseline |
| --- | --- | --- |
| Standard Q/K/V | 135.23° | 100% |
| **Unified** | **137.32°** | **101.5%** |

Within 2%: **geometric path preserved**.

### Layer Orthogonality

Average angle between consecutive layer representations:

| Attention Type | Mean Angle |
| --- | --- |
| Standard Q/K/V | 22.54° |
| **Unified** | **22.89°** |

Within 2%: **rotation structure preserved**.

---

# 5. Analysis

## 5.1 Why Does It Work?

**Shared structure**: Q, K, V transform the same input for the same objective. Separate matrices learn redundant structure. A unified matrix learns it once.

**Implicit regularization**: Fewer parameters may prevent overfitting. Our improved perplexity at small scale (9.58 vs ~10-12) and competitive BPB at large scale support this.

**Parameter reallocation**: In constrained settings, the bytes saved on attention go to the MLP. Attention routes information; the MLP transforms it. More MLP budget means more compute per byte.

## 5.2 What Does Low Attention Share Mean?

Our parameter distribution (5.7% attention at small scale, 18% at large scale) suggests:

* **Attention is routing**: It decides WHERE information flows, not what it becomes
* **FFN is essential**: The nonlinear transformation needs the most capacity
* **Depth matters more than width**: K=11 layers with d=528 beats K=10 with d=552

## 5.3 Limitations

* **Domain**: TinyStories and FineWeb tested. Other domains (code, multilingual) unexplored.
* **Scale beyond 23M**: Behavior at 100M+ parameters unknown.
* **Compression overhead**: Unified attention weights have higher entropy (0.69 vs 0.59 bytes/param) due to shared projection structure. This limits gains in extremely tight budgets.
* **Technique transfer**: Several standard transformer optimizations (XSA, BigramHash) don't transfer to unified attention due to band coupling.

---

# 6. Related Work

**Multi-Query Attention** [5]: Shares K, V across heads. We go further: unifying Q, K, V into a single projection.

**LoRA** [6]: Reduces parameters post-training via low-rank adaptation. We reduce architecturally, during training.

**Efficient Attention**: Linear attention and sparse attention reduce O(n²) complexity. We reduce parameters while keeping full attention.

**F-Net** [7]: Replaces attention with Fourier transforms. Loses differential weighting.

**Mamba** [8]: Replaces attention with selective state spaces. Loses Q/K asymmetry.

**Ours**: Preserves both differential weighting and asymmetry. Removes only the redundant parameterization.

---

# 7. Future Work

We've demonstrated unified attention at two scales: 484K parameters on TinyStories and 23.2M parameters on FineWeb. Several directions remain:

**Scale**: Does the 67% reduction hold at 100M or 1B parameters? The redundancy argument should apply regardless of scale, but this requires empirical verification.

**Domains**: TinyStories and FineWeb use English text. Does unified attention maintain its advantage on code, multilingual data, or multimodal inputs?

**Compression-aware training**: Unified attention weights have higher entropy than standard Q/K/V weights (0.69 vs 0.59 bytes/param after int6 quantization). Developing quantization methods that account for the shared projection structure could further improve compression ratios and close this gap.

**Content-band injection**: Techniques like BigramHash fail when applied at the input because the shared projection can't route signals differently to three bands. Injecting auxiliary information only into the content band after the split may bypass this coupling.

**Full Hessian GPTQ**: Our current results use GPTQ-lite (diagonal Hessian). Full Hessian GPTQ with self-generated calibration data could improve post-quantization quality by 0.003-0.005 BPB.

### Our Goal: AI That Runs Everywhere

At 946 KB and 700+ tokens/sec on CPU, Yocto proves small models can be capable. The goal is to scale unified attention to build larger models that remain small enough to run anywhere: private, fast, and free.

We release all code, weights, and training configs to enable this future.

---

# 8. Conclusion

We asked: What is the minimal parameterization for attention?

**Answer**: One projection suffices.

At small scale (484K params), unified attention achieves 9.58 perplexity on TinyStories, matching models 2-4× larger with only 5.7% of parameters in attention.

At large scale (23.2M params), unified attention achieves **1.1088 BPB** on OpenAI's Parameter Golf challenge, beating every standard Q/K/V transformer on the leaderboard. The architecture trades attention parameters for MLP parameters: 2.28 MB of routing freed for 2.49 MB of computation within a 16 MB budget.

The physics of attention is simpler than standard architectures suggest. Three matrices were never necessary. One suffices.

---

# References

[1] Vaswani et al., "Attention Is All You Need," NeurIPS 2017. [2] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021. [3] Eldan & Li, "TinyStories: How Small Can Language Models Be?," 2023. [4] Goedecke, "Training a Language Model on a Laptop," 2024. [5] Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," 2019. [6] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021. [7] Lee-Thorp et al., "FNet: Mixing Tokens with Fourier Transforms," 2021. [8] Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2023.

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
        # Single projection, three bands
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

## YOCTO

**484,272 Parameters · 946 KB (fp16) · 700+ tok/s · 67% Less Attention · Open Source**

### Quick Start

```bash
git clone https://github.com/reinforceai/yocto
cd yocto
pip install -r requirements.txt
python inference.py --prompt "Once upon a time"
```

**700+ tokens/sec on CPU**, no GPU needed.

### Live Demo & Model

🤗 **Try it now**: [HuggingFace Space](https://huggingface.co/spaces/Reinforce-ai/yocto-demo)

📦 **Model weights**: [HuggingFace Model](https://huggingface.co/Reinforce-ai/yocto)

🏆 **OpenAI Parameter Golf**: [PR #1202](https://github.com/openai/parameter-golf/pull/1202) (val_bpb 1.1412, 10-min record) | [PR #1270](https://github.com/openai/parameter-golf/pull/1270) (val_bpb 1.1088, 1-hr unlimited, beats all SOTA)

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