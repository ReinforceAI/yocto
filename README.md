# Unified Attention — Basis-Shared QKV for Efficient Transformers

> *We introduce Yocto, a transformer architecture that constrains Q, K, and V to share a single linear basis rather than learning three independent projections. This basis-sharing constraint is a distinct inductive bias from standard attention — including its fused-QKV implementations — and it wins head-to-head against well-tuned standard transformers at two scales.*

---

# Abstract

Standard transformer attention learns three independent projections W_Q, W_K, W_V, each mapping the input into its own output space. Fused-QKV implementations in PyTorch, nanoGPT, and FlashAttention concatenate these three projections into one matrix for memory-layout reasons, but the three output subspaces remain independent: each third of the output is still a separately learned linear map.

We introduce **Unified Attention**: a single projection whose output is split into contiguous `[seeking | offering | content]` bands. Unlike fused QKV, the three bands are forced to live in non-overlapping coordinate slices of a *single shared output basis*. This is a different inductive bias — not a re-parameterization — and it changes what the architecture can and cannot learn.

We validate this claim at two scales:

* **484K parameters, TinyStories**: 9.58 validation perplexity, matching models 2–4× larger. Attention accounts for only 5.7% of parameters.
* **23.2M parameters, OpenAI Parameter Golf**: **1.1088 BPB**, beating every submission on the leaderboard — including standard Q/K/V transformers at up to 106M parameters that have been iteratively optimized by multiple teams with specialized techniques (XSA, full-Hessian GPTQ, BigramHash).

The structural signature that distinguishes basis sharing from fused QKV is a *transfer asymmetry*: techniques designed for independent Q/K/V projections (XSA, BigramHash) **help** standard transformers and **hurt** ours. If unified attention were merely fused QKV with narrower heads, these techniques would transfer identically. They don't. This asymmetry is direct evidence that the inductive bias is different.

We arrived at this architecture through a wave-physics reading of attention: Q, K, V are three views of the same resonance pattern, and should share the coordinate system in which that pattern is expressed. The 67% reduction in attention parameters falls out of this constraint — it is a consequence, not the goal.

---

# 1. Introduction

Standard transformer attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

where `Q = W_Q·x`, `K = W_K·x`, `V = W_V·x` are three independent linear projections. Fused-QKV implementations concatenate these into a single `W_in` of shape `(3d, d)` for speed, but mathematically each third is still a separate linear map into its own output subspace.

**The question we asked**: what happens if we force Q, K, V to share not just a weight buffer, but a *basis*?

**What we found**: when a single projection `W` maps `x → u ∈ ℝ^d` and Q, K, V occupy disjoint coordinate bands of `u`, the three roles are forced to negotiate within one shared output geometry. This is a strictly stronger constraint than independent projections. It can't express everything independent projections can. But at the scales we've tested, it doesn't need to — and the constraint acts as a beneficial inductive bias.

The basis-sharing constraint has a physical interpretation. Read through wave physics, Q, K, V are three functional views of the same underlying representation: "what am I seeking," "what am I offering," and "what content am I carrying." These three views are not independent — they describe aspects of one thing — and a well-structured architecture should reflect that coupling. Independent projections are free to place Q, K, V anywhere in ℝ^d; basis sharing forces them to share coordinates. The physics says they should.

### Contributions

1. **Basis-shared QKV as an inductive bias**: a constraint distinct from both unfused and fused QKV, forcing Q, K, V into disjoint slices of a single shared output basis.
2. **Structural evidence the bias is real**: XSA and BigramHash — techniques that help standard Q/K/V — actively hurt basis-shared Q/K/V. This transfer asymmetry cannot be explained by parameter count or memory layout.
3. **Empirical wins at two scales**: 9.58 PPL on TinyStories at 484K params; 1.1088 BPB on Parameter Golf at 23.2M params, beating all standard-transformer submissions up to 106M params.
4. **Geometric verification**: cumulative layer rotation and inter-layer angle measurements confirm the basis-shared architecture traces a path through representation space that is within 2% of the standard transformer — the constraint does not distort the underlying geometry, only its parameterization.
5. **FA3 head-dim padding**: a zero-padding trick enabling Hopper-optimized flash attention with the non-standard head dimensions that fall out of basis-shared architectures.
6. **Wave-field interpretation**: a physics framing that predicted basis sharing would work, and that continues to make falsifiable predictions about which techniques should and should not transfer to the new architecture.

---

# 2. Theory: Basis Sharing as a Physical Constraint

## 2.1 Vectors as Waveforms

Every vector is a waveform with `d` dimensions. Each dimension carries:

* **Amplitude**: `|v_i|` — activation strength
* **Phase**: `sign(v_i)` — positive or negative direction

The dot product between vectors is wave interference:

$$\mathbf{x} \cdot \mathbf{w} = \sum_i |x_i| \cdot |w_i| \cdot \text{sign}(x_i) \cdot \text{sign}(w_i)$$

Same sign contributes constructively, opposite sign destructively. A trained projection is a field of such waveforms arranged to resonate selectively with inputs that share their phase structure.

## 2.2 Why Q, K, V Should Share a Basis

In standard attention, Q, K, V are learned as three independent linear maps. Nothing in the architecture forces them to share geometric structure — they happen to be related only because gradient descent couples them through the loss.

But Q, K, V are not independent objects:

1. **Same input.** All three transform the same `x`.
2. **Same objective.** All three are optimized against the same loss.
3. **Coupled function.** The attention score `Q·Kᵀ` is only meaningful if Q and K are expressed in compatible coordinates; the weighted sum over V is only meaningful if V lives in a space the output projection can read.

Read physically: Q, K, V are three views of the same resonance pattern. "What am I seeking," "what am I offering," and "what content am I carrying" are not three independent waveforms — they are three projections of one waveform onto functionally distinct roles. If this reading is right, the three roles should share the basis in which the waveform is expressed, and only differ in which slice of that basis each role occupies.

**Basis sharing is the architectural realization of this constraint.** A single projection `W: ℝ^d → ℝ^d` produces one output vector `u`. Q, K, V are assigned disjoint contiguous slices of `u`. The three roles are forced to be expressible as non-overlapping coordinates of a *single* learned transformation of the input, rather than as three separate transformations that happen to share a training signal.

## 2.3 How Basis Sharing Differs from Fused QKV

This is the distinction that matters most and is easiest to miss.

**Unfused standard attention** learns three independent matrices `W_Q, W_K, W_V`, each of shape `(d, d/n_heads × n_heads) = (d, d)`. Three separate linear maps, three separate output spaces.

**Fused QKV** (used in `nn.MultiheadAttention`, nanoGPT, FlashAttention, and every production transformer) concatenates these three matrices into a single buffer `W_in` of shape `(3d, d)` for memory-access and kernel-launch efficiency. The computation is a single matmul, but *mathematically* each third of the output is still an independent linear map into its own subspace of ℝ^d. Fused QKV is a memory layout, not an inductive bias. A fused-QKV network is function-equivalent to an unfused one.

**Basis-shared QKV** (ours) learns a single matrix `W` of shape `(d, d)` and assigns disjoint coordinate bands of its output to Q, K, V. Critically, Q, K, V now live in the *same* output space — they are forced to use compatible coordinates because they are coordinates of one vector. This cannot be expressed as three independent `(d, d/3)` maps, because those three maps would project into three unrelated subspaces. Ours projects into one subspace, partitioned three ways.

| | Parameter count | Output space structure | Is it an inductive bias? |
|---|---|---|---|
| Unfused standard | 3d² (or d² with `head_dim = d/3n_heads`) | Three independent subspaces | No — it's the default |
| Fused QKV | Same as unfused | Three independent subspaces (same math, different memory layout) | No |
| **Basis-shared (ours)** | d² | **One shared subspace, partitioned into 3 bands** | **Yes — strictly stronger constraint** |

Basis sharing is a strictly smaller hypothesis space than independent projections of any parameter count. It can't express everything independent Q/K/V can. But at the scales we've tested, the restricted hypothesis space acts as a beneficial prior — and the empirical results show it wins.

## 2.4 Predictions From the Physics Framing

The wave-physics reading that motivated basis sharing makes further predictions we can test:

**Prediction 1**: Techniques that assume Q, K, V can be independently manipulated should fail on basis-shared architectures, because Q, K, V share coordinates and can't be moved independently.

**Prediction 2**: Techniques that respect the shared basis should work as well or better, because basis sharing is a cleaner geometric structure to exploit.

Both predictions are confirmed by the transfer asymmetry reported in Section 4.2: XSA and BigramHash (which treat Q, K, V as independent) hurt our architecture; standard quantization and training-time techniques that respect the overall geometry transfer cleanly.

---

# 3. Architecture

## 3.1 Unified Attention

```
Unfused standard:    Q = W_Q·x,  K = W_K·x,  V = W_V·x     [three independent d²/3 maps]
Fused QKV:           [Q; K; V] = W_in·x                     [one buffer, same math as unfused]
Basis-shared (ours): u = W·x
                     Q = u[:d/3], K = u[d/3:2d/3], V = u[2d/3:]
                     [one shared output space, partitioned into 3 bands]
```

We apply Rotary Position Embedding (RoPE) to the Q and K bands but not to V. Position affects *routing* (who attends to whom) but not *content* (what information transfers). This is consistent with the physics reading: position is a phase rotation applied to the seeking/offering waveforms, not to the content they carry.

## 3.2 FA3 Head-Dim Padding

Flash Attention 3 (Hopper) requires `head_dim` to be a multiple of 8. When basis-shared attention produces non-standard head dimensions (e.g., `head_dim = 44` from `d = 528` with 4 heads), we zero-pad to the nearest multiple of 8 before FA3 and slice back after:

```python
pad_n = (8 - head_dim % 8) % 8
if pad_n > 0:
    q, k, v = [F.pad(t, (0, pad_n)) for t in (q, k, v)]
out = flash_attn_func(q, k, v, causal=True)
y = out[..., :head_dim]
```

Mathematically lossless: padded zeros contribute nothing to dot products or weighted sums. The 9% compute overhead from 44 → 48 dims is overwhelmed by FA3's 1.57× speedup over FA2/SDPA.

---

# 4. Experiments

## 4.1 Small Scale: TinyStories (484K params)

| Component | Value |
|---|---|
| Embedding dimension | 72 |
| Layers | 4 |
| Attention heads | 3 |
| FFN hidden | 288 |
| Vocabulary | 4,000 |
| Context length | 512 |
| **Total parameters** | **484,272** |

### Parameter Distribution

| Component | Parameters | Share |
|---|---|---|
| Embeddings | 288,000 | 59.5% |
| **Attention** | **27,648** | **5.7%** |
| FFN | 166,464 | 34.4% |
| Other | 2,160 | 0.4% |

### Results

| Model | Total Params | Val PPL |
|---|---|---|
| **Ours (basis-shared)** | **484K** | **9.58** |
| TinyStories-1M | 1M | ~10–12 |
| seangoedecke [4] | 1.8M | ~9.6 |

At 52% fewer parameters than TinyStories-1M we reach better perplexity, and at 73% fewer than seangoedecke we match it.

## 4.2 Large Scale: OpenAI Parameter Golf (23.2M params)

We applied basis-shared attention to [OpenAI's Parameter Golf challenge](https://github.com/openai/parameter-golf): train the best language model that fits in 16 MB. Evaluated on FineWeb validation in bits-per-byte (BPB), lower is better.

### Architecture

| Component | Value |
|---|---|
| Embedding dimension | 528 |
| Unique layers (K) | 11 |
| Attention heads | 4 (head_dim = 44) |
| MLP | 3× (1584) with LeakyReLU(0.5)² |
| Vocabulary | 1,024 |
| Context length | 2,048 |
| **Total parameters** | **23,209,295** |
| **Compressed artifact** | **~15.97 MB** |

Additional techniques: SmearGate, LN Scale (1/√(layer+1)), VE128 on last 2 layers, STE QAT int6, EMA(0.997) + SWA, Parallel Muon optimizer, Legal Score-First TTT at eval.

### Results — Record Track (10 min train, 10 min eval, 8×H100 SXM)

| Seed | step_avg | steps | Post-TTT BPB | Artifact |
|---|---|---|---|---|
| 1337 | 49.6 ms | 12,088 | 1.1416 | 15,991,687 |
| 42 | 49.6 ms | 12,109 | 1.1416 | 15,988,916 |
| 2025 | 49.6 ms | 12,103 | 1.1403 | 15,962,515 |
| **Mean** | **49.6 ms** | **12,100** | **1.1412 (std 0.0008)** | |

Rank 7 on the main leaderboard, and the only non-standard-attention architecture in the top 10.

### Results — Unlimited Compute Track (1 hour train, 8×H100 SXM)

| | BPB |
|---|---|
| Previous unlimited SOTA (1-bit quant, 106M params, 2hr) | 1.1239 |
| Previous record SOTA (standard Q/K/V + GPTQ + XSA, 10min) | 1.1147 |
| **Ours (basis-shared, 23.2M params, 1hr)** | **1.1088** |

Basis-shared attention at 23.2M parameters and one hour of training beats every submission on both tracks, including standard-transformer architectures with 4.5× more parameters and specialized techniques that have been iteratively optimized by multiple teams.

### Post-Compression Parameter Reallocation

In a parameter-constrained setting, basis sharing frees compressed bytes for the MLP:

| | Standard (SOTA) | Basis-Shared (Ours) |
|---|---|---|
| Attention params | 8.65M (33% of blocks) | 4.09M (18% of blocks) |
| MLP params | 17.3M | 18.4M |
| Attention compressed | 5.10 MB | 2.82 MB |
| MLP compressed | 10.21 MB | 12.70 MB |

We trade 2.28 MB of routing for 2.49 MB of computation. Routing is what basis-shared attention makes cheaper; computation is what the MLP uses the freed budget for.

### The Transfer Asymmetry: Structural Evidence for Basis Sharing

The cleanest empirical evidence that basis-shared attention is a different inductive bias — not just a re-parameterization of fused QKV — is that techniques designed for independent Q/K/V projections **help** standard transformers and **hurt** ours:

| Technique | Standard Q/K/V | Basis-Shared | Root Cause |
|---|---|---|---|
| XSA | +0.004 BPB | **−0.0015 BPB** | Content band is coupled to seeking/offering in the shared projection; XSA assumes independent manipulation |
| BigramHash at input | +0.004 BPB | **−0.009 BPB** | A single matrix can't route bigram features differently to three bands when the bands share a basis |

If basis-shared attention were mathematically equivalent to fused QKV with narrower heads, both techniques would transfer identically. They don't. The sign flips. This is direct structural evidence that the inductive bias is different — not better or worse in an abstract sense, but *different in the specific way the physics framing predicted*. Techniques that assume Q, K, V can be independently manipulated break on an architecture where Q, K, V share coordinates.

This asymmetry is the single most important empirical result in the paper. It cannot be explained by parameter count, memory layout, or implementation detail. It can only be explained by the architecture being structurally different.

## 4.3 Geometric Verification

Does basis-shared attention preserve the geometric path through representation space that standard attention traces? We ran controlled experiments comparing attention mechanisms on identical architectures (6-layer transformers, `embed_dim = 126`, trained on synthetic language tasks).

### Cumulative Layer Rotation

Cumulative rotation measures the total accumulated angle as a representation flows through the stack — a signature of how much the residual stream is being reshaped from layer to layer.

| Attention Type | Cumulative Rotation | vs Baseline |
|---|---|---|
| Standard Q/K/V | 135.23° | 100% |
| **Basis-shared** | **137.32°** | **101.5%** |

Within 2%: the geometric path is preserved.

### Inter-Layer Angle

Average angle between consecutive layer representations:

| Attention Type | Mean Angle |
|---|---|
| Standard Q/K/V | 22.54° |
| **Basis-shared** | **22.89°** |

Within 2%: the per-layer rotation structure is preserved.

**Interpretation**: basis sharing does not distort the geometry of the residual stream — it only constrains how attention parameterizes its contribution to that geometry. The path is the same. The cost of walking the path is lower.

---

# 5. Analysis

## 5.1 Why Does Basis Sharing Work?

**Shared geometry matches the task.** Q, K, V are functionally coupled (same input, same loss, same attention computation). Independent projections give the optimizer three separate degrees of freedom that it must *learn* to couple through gradient descent. Basis sharing hard-codes the coupling into the architecture, turning a learned regularity into a structural one.

**Restricted hypothesis space acts as a prior.** Basis sharing cannot express everything independent Q/K/V can. This is not a weakness at the scales we've tested — it's the mechanism by which the architecture is beneficial. The restrictions are exactly the degrees of freedom independent projections would use to learn redundant or counterproductive structure.

**Parameter reallocation follows for free.** In parameter-constrained settings, the attention budget saved by basis sharing is spent on the MLP, where it does more work per byte. Attention routes information; the MLP transforms it. A model that can afford more transformation per byte of routing is a better language model at small scale.

## 5.2 What the Low Attention Share Tells Us

Our parameter distribution (5.7% attention at 484K scale, 18% at 23.2M scale) suggests a general principle:

* **Attention is routing.** It decides *where* information flows, not *what* it becomes. Routing doesn't need a large fraction of the parameter budget.
* **FFN is essential.** The nonlinear transformation needs the most capacity. This is where the freed budget should go.
* **Depth beats width.** `K = 11` layers at `d = 528` beats `K = 10` at `d = 552` at matched parameter count. More routing decisions beat wider routing.

These are observations from two data points and should be treated as hypotheses for larger-scale validation.

## 5.3 Limitations

* **Domain.** TinyStories and FineWeb. Code, multilingual, and multimodal unexplored.
* **Scale.** 484K and 23.2M tested. Behavior at 100M+ unknown — the restricted hypothesis space may stop being beneficial once the model has enough capacity that independent projections no longer overfit.
* **Compression overhead.** Basis-shared weights have higher entropy after int6 quantization (0.69 vs 0.59 bytes/param) because the shared projection concentrates structure. This limits gains in extremely tight budgets.
* **Technique transfer.** XSA and BigramHash (and potentially other Q/K/V-specific optimizations) don't transfer. Basis sharing is not a drop-in replacement — adopting it means rebuilding the technique stack around the new inductive bias.

---

# 6. Related Work

**Fused QKV** (nanoGPT, PyTorch `MultiheadAttention`, FlashAttention). A memory-layout optimization that concatenates three independent linear projections into a single buffer. Mathematically identical to unfused standard attention. **Our distinction**: we share the *basis*, not just the buffer. Fused QKV has three independent output subspaces; we have one output subspace partitioned three ways. This is why techniques designed for independent Q/K/V break on our architecture but not on fused QKV.

**Multi-Query Attention** [5]. Shares K and V across heads while keeping Q per-head. We go further: we share the underlying basis across Q, K, and V themselves.

**LoRA** [6]. Reduces parameters post-training via low-rank adaptation. We reduce architecturally, during training, via a constraint on the output space.

**F-Net** [7]. Replaces attention with Fourier transforms. Loses differential weighting — all tokens mix identically, which is why it underperforms. We preserve differential weighting and only constrain the output geometry.

**Mamba** [8]. Replaces attention with selective state spaces. Merges "what I seek" and "what I offer" into a single state, losing Q/K asymmetry. We preserve Q/K asymmetry while constraining their shared coordinate system.

**Our place in this list**: we are the only approach that preserves both differential weighting and Q/K asymmetry while changing the architecture's inductive bias in a way that is empirically distinguishable from fused QKV through transfer asymmetry.

---

# 7. Future Work

**Scale.** Does basis sharing hold at 100M and 1B parameters? The restricted-hypothesis-space argument predicts the benefit shrinks as the model gets large enough that independent projections stop overfitting. Locating the crossover is a concrete empirical question.

**Domains.** TinyStories and FineWeb are English text. Code, multilingual, and multimodal are open.

**Compression-aware training.** Basis-shared weights have higher entropy after quantization. A quantization scheme that accounts for the shared projection structure could close the compression gap and improve tight-budget performance.

**Content-band injection.** BigramHash fails at the input because the shared projection can't route bigram features differently to three bands. Injecting auxiliary information only into the content band *after* the split should bypass the coupling and recover the technique's benefit. We have not yet tested this.

**Full-Hessian GPTQ.** Current results use GPTQ-lite (diagonal Hessian). Full Hessian with self-generated calibration should improve post-quantization quality by 0.003–0.005 BPB.

**Further predictions from the physics framing.** The wave-physics reading predicts that *any* technique that treats Q, K, V as independent should fail on basis-shared attention, while any technique that respects the shared basis should transfer cleanly. This is a falsifiable prediction about future work, and we invite others to test it.

### Goal: AI That Runs Everywhere

At 946 KB and 700+ tok/s on CPU, Yocto demonstrates that small models can be capable. The goal is to scale basis-shared attention into larger models that remain small enough to run anywhere — private, fast, free. All code, weights, and training configs are released.

---

# 8. Conclusion

We asked: what happens if Q, K, V are forced to share not just a weight buffer, but a basis?

**Answer**: you get a strictly stronger inductive bias than fused QKV. At small and mid scale, that bias wins.

At 484K parameters on TinyStories, basis-shared attention reaches 9.58 perplexity with 5.7% of parameters in attention — matching models 2–4× larger. At 23.2M parameters on OpenAI's Parameter Golf, it reaches **1.1088 BPB**, beating every standard-transformer submission on the leaderboard including architectures with 4.5× more parameters.

The transfer asymmetry — XSA and BigramHash helping standard Q/K/V and hurting ours — is direct structural evidence that basis-shared attention is a different inductive bias, not a re-parameterization. You cannot get sign-flipped transfer from a memory-layout change.

The physics is simple. Q, K, V are three views of the same resonance pattern. Independent projections let them drift into unrelated subspaces. Basis sharing forces them to share coordinates, which is what the physics said they should do all along. Three matrices were never necessary. One, partitioned, suffices.

---

# References

[1] Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
[2] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021.
[3] Eldan & Li, "TinyStories: How Small Can Language Models Be?," 2023.
[4] Goedecke, "Training a Language Model on a Laptop," 2024.
[5] Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," 2019.
[6] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," 2021.
[7] Lee-Thorp et al., "FNet: Mixing Tokens with Fourier Transforms," 2021.
[8] Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2023.

---

# Appendix: Implementation

```python
class UnifiedAttention(nn.Module):
    """Basis-shared QKV attention.

    Unlike fused QKV (which concatenates three independent projections
    into one buffer), this layer learns a single projection W whose
    output is partitioned into Q, K, V bands. Q, K, V share a basis:
    they are coordinates of one vector, not three.
    """

    def __init__(self, embed_dim, num_heads):
        self.third = embed_dim // 3
        self.W_unified = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_out = nn.Linear(self.third, embed_dim, bias=False)
        self.rope = RotaryPositionEmbedding(self.third // num_heads)

    def forward(self, x):
        # Single projection into a shared output basis.
        u = self.W_unified(x)

        # Partition the shared basis into Q, K, V bands.
        seeking, offering, content = u.split(self.third, dim=-1)

        # RoPE on Q and K bands only. Position is a phase rotation
        # on seeking/offering, not on content.
        cos, sin = self.rope(x)
        seeking, offering = apply_rope(seeking, offering, cos, sin)

        # Standard attention over the partitioned bands.
        out = F.scaled_dot_product_attention(
            seeking, offering, content, is_causal=True
        )
        return self.W_out(out)
```

---

## YOCTO

**484,272 Parameters · 946 KB (fp16) · 700+ tok/s · Basis-Shared QKV · Open Source**

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

🏆 **OpenAI Parameter Golf**: [PR #1202](https://github.com/openai/parameter-golf/pull/1202) (val_bpb 1.1412, 10-min record) · [PR #1270](https://github.com/openai/parameter-golf/pull/1270) (val_bpb 1.1088, 1-hr unlimited, beats all SOTA)

### Citation

```bibtex
@misc{deshwal2026yocto,
  title={Unified Attention: Basis-Shared QKV for Efficient Transformers},
  author={Deshwal, Viraj},
  year={2026},
  url={https://www.reinforceai.com/yocto},
  howpublished={\url{https://github.com/reinforceai/yocto}}
}
```