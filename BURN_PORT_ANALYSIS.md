# FLUX.2 Klein to Rust/Burn Porting Analysis

**Date:** January 18, 2026
**Target:** FLUX.2 [klein] 9B Model
**Framework:** Burn 0.20.0 (Latest Release)

---

## Executive Summary

**Porting Feasibility: HIGH ✅**
**Difficulty Rating: 4/10** (Moderate - mainly manual einops translation)
**Estimated Effort: 1-2 weeks** (for experienced Rust + ML developer)

FLUX.2 Klein is highly suitable for porting to Rust/Burn. The recent Burn 0.20.0 release (January 2025) includes critical features like Flash Attention and GroupNorm that make this project significantly easier than anticipated.

---

## Technology Stack Analysis

### Current FLUX.2 Dependencies

| Component | PyTorch Implementation | Purpose |
|-----------|----------------------|---------|
| Core Model | `torch.nn.Module` | Transformer architecture |
| Attention | `torch.nn.functional.scaled_dot_product_attention` | Multi-head attention |
| Normalization | LayerNorm, GroupNorm, BatchNorm | Feature normalization |
| Autoencoder | Conv2d-based VAE | Image encoding/decoding |
| Tensor Ops | `einops.rearrange` | Tensor reshaping |
| Image Processing | `torchvision.transforms` | Preprocessing |

### Burn 0.20.0 Capabilities

| Feature | Status | Notes |
|---------|--------|-------|
| **GroupNorm** | ✅ Native | `burn::nn::modules::norm::GroupNorm` |
| **Flash Attention** | ✅ Built-in | CubeK-optimized, auto-tuned |
| **MultiHeadAttention** | ✅ Native | Standard transformer attention |
| **Conv2d/Linear** | ✅ Native | Full CNN support |
| **LayerNorm/BatchNorm** | ✅ Native | All normalization layers |
| **Tensor Operations** | ✅ Manual | reshape, permute, transpose, cat, etc. |
| **einops** | ⚠️ Partial | Rust library exists but only for `tch`, not Burn |
| **CUDA/ROCm/Metal** | ✅ Native | Multi-backend support |
| **safetensors** | ✅ Excellent | Native Rust support |

---

## Detailed Findings

### 1. GroupNorm Support ✅

**Status:** Fully supported in `burn::nn::modules::norm`

```rust
// Configuration
let config = GroupNormConfig::new(num_groups, num_channels)
    .with_epsilon(1e-6);
let group_norm = config.init(&device);

// Usage
let output = group_norm.forward(input); // [B, C, *] -> [B, C, *]
```

**Features:**
- Learnable affine parameters (gamma/beta)
- Configurable epsilon for numerical stability
- Supports any number of spatial dimensions

**References:**
- [GroupNorm API Documentation](https://docs.rs/burn/latest/burn/nn/modules/norm/struct.GroupNorm.html)

### 2. Attention Mechanisms ✅

**Status:** Fully supported with state-of-the-art optimizations

#### Standard Multi-Head Attention

```rust
let config = MultiHeadAttentionConfig::new(d_model, n_heads)
    .with_dropout(0.1);
let mha = config.init(&device);
```

**Features:**
- Q, K, V linear projections
- Output projection
- Dropout support
- Based on "Attention Is All You Need"

**References:**
- [MultiHeadAttention API](https://burn.dev/docs/burn/nn/attention/struct.MultiHeadAttention.html)

#### Flash Attention (Burn 0.20.0) 🚀

**Major advancement:** Burn 0.20.0 includes Flash Attention implementation through CubeK architecture.

**Key Benefits:**
- **Automatic kernel optimization** for CPU and GPU
- **Tensor Core support** on NVIDIA hardware (Blackwell, Ada, Hopper)
- **Dynamic tiling** - selects optimal dimensions per backend
- **TMA (Tensor Memory Accelerator)** support
- **Fused operations** for 3x speedups

**Performance:**
- Functionally correct kernels already generated
- Low-level optimizations ongoing for full tensor core saturation
- Unified CPU/GPU execution path

**References:**
- [Burn 0.20.0 Release Notes](https://burn.dev/blog/release-0.20.0/)
- [CubeCL GitHub](https://github.com/tracel-ai/cubecl)

### 3. Tensor Operations ⚠️

**Status:** All primitives available, manual translation required

#### Available Operations

From [`burn::tensor::Tensor`](https://burn.dev/docs/burn/tensor/struct.Tensor.html):

**Shape Manipulation:**
- `reshape(shape)` - with `-1` inference support
- `permute(axes)` - arbitrary dimension reordering
- `transpose()` / `swap_dims()` - dimension swapping
- `flatten(start, end)` - collapse dimensions
- `squeeze()` / `unsqueeze()` - add/remove size-1 dims
- `movedim(source, dest)` - move dimensions

**Concatenation & Splitting:**
- `cat(tensors, dim)` - concatenate along dimension
- `stack(tensors, dim)` - stack along new dimension
- `chunk(chunks, dim)` - split into N chunks
- `split(sizes, dim)` - split with custom sizes

**Advanced Indexing:**
- `slice(ranges)` - flexible slicing with negative indices
- `select(dim, indices)` - gather along dimension
- `gather(dim, indices)` - advanced gathering
- `scatter(dim, indices, src)` - scatter values

#### Einops Translation Required

**Challenge:** FLUX.2 uses `einops.rearrange()` extensively (~30-40 instances)

**Example Translation:**

```python
# PyTorch with einops
qkv = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
```

```rust
// Burn equivalent
let [batch, seq_len, hidden] = qkv.dims3();
let head_dim = hidden / (3 * num_heads);
let qkv = qkv
    .reshape([batch, seq_len, 3, num_heads, head_dim])
    .permute([2, 0, 3, 1, 4]);
```

**Existing Rust einops:**
- [einops Rust crate](https://docs.rs/einops/latest/einops/) exists
- **Limitation:** Only supports `tch` (PyTorch bindings), not Burn tensors
- Provides `Rearrange`, `Reduce`, `Repeat` operations
- Could potentially be adapted for Burn (future work)

### 4. CubeCL & CubeK Performance

**Burn 0.20.0 Performance Highlights:**

#### CPU Benchmarks
- **max_pool2d** (2, 32, 512, 512): 4.66ms vs LibTorch 16.96ms (**4x faster**)
- **reduce_argmin** (axis 0): 6.89ms vs LibTorch 230.4ms (**33x faster**)
- **Fused operations**: 4.85ms vs 14.9ms unfused (**3x speedup**)

#### GPU Benchmarks
- **matmul** (1, 4096, 4096): 639µs vs LibTorch 627µs (**on par**)
- Tensor Core utilization on modern NVIDIA GPUs
- Supports NVIDIA Blackwell architecture

**Architecture Features:**
- **Kernel fusion** - automatic operation combining
- **JIT compilation** - runtime specialization
- **Multi-backend** - CUDA, ROCm, Metal, Vulkan, WebGPU, CPU
- **Lazy execution** - CPU backend now matches GPU behavior

**Reference:**
- [Burn 0.20 Performance Analysis](https://burn.dev/blog/release-0.20.0/)

---

## Code Structure Analysis

### Core Files to Port

| File | Lines | Complexity | Dependencies | Porting Effort |
|------|-------|-----------|--------------|----------------|
| `model.py` | 485 | High | einops, torch.nn | 3-4 days |
| `autoencoder.py` | 337 | Medium | einops, torch.nn | 2-3 days |
| `sampling.py` | 395 | Medium | einops, torchvision | 2-3 days |
| `util.py` | ~200 | Low | PIL, torchvision | 1 day |
| **Total (core)** | **~1400** | - | - | **8-11 days** |

### Files to Skip (Out of Scope)

| File | Reason |
|------|--------|
| `text_encoder.py` | LLM inference (use existing solution) |
| `openrouter_api_client.py` | API client (skip or rewrite) |
| `watermark.py` | Optional feature |
| `system_messages.py` | Prompt templates (trivial) |

---

## High-Level Porting Plan

### Phase 1: Project Setup (1 day)

**Goal:** Create Rust/Burn project structure with proper dependencies

#### Tasks:
1. **Initialize Cargo project**
   ```bash
   cargo new flux2-burn --lib
   cd flux2-burn
   ```

2. **Configure `Cargo.toml` dependencies**
   ```toml
   [dependencies]
   burn = "0.20"
   burn-ndarray = "0.20"  # CPU backend
   burn-cuda = "0.20"     # NVIDIA GPU backend
   safetensors = "0.4"
   image = "0.25"
   half = "2.3"           # for bfloat16
   ```

3. **Set up project structure**
   ```
   src/
   ├── lib.rs              # Main library entry
   ├── model/
   │   ├── mod.rs
   │   ├── flux2.rs        # Main transformer model
   │   ├── layers.rs       # Custom layers (RMSNorm, etc.)
   │   └── attention.rs    # Attention blocks
   ├── autoencoder/
   │   ├── mod.rs
   │   ├── encoder.rs
   │   └── decoder.rs
   ├── sampling/
   │   ├── mod.rs
   │   └── denoising.rs
   └── utils/
       ├── mod.rs
       ├── image.rs        # Image preprocessing
       └── tensor_ops.rs   # einops replacements
   ```

4. **Create test infrastructure**
   - Unit tests for each module
   - Integration tests for end-to-end inference
   - Numerical comparison tests vs PyTorch

### Phase 2: Core Model Architecture (3-4 days)

**Goal:** Implement transformer backbone from `model.py`

#### 2.1: Basic Building Blocks (1 day)

**Implement custom layers:**

- [ ] `RMSNorm` (model.py:436-445)
- [ ] `QKNorm` (model.py:448-457)
- [ ] `SiLUActivation` (model.py:182-189)
- [ ] `Modulation` (model.py:192-204)
- [ ] `MLPEmbedder` (model.py:385-393)

**Utility functions:**
- [ ] `timestep_embedding()` (model.py:412-433)
- [ ] `rope()` - RoPE embeddings (model.py:469-476)
- [ ] `apply_rope()` (model.py:479-484)

**Testing:**
- Compare output with PyTorch for each layer
- Verify numerical precision (fp32, bf16)

#### 2.2: Attention Mechanism (1 day)

**Implement attention with einops translation:**

- [ ] `attention()` function (model.py:460-466)
  - Port RoPE application
  - Use Burn's built-in scaled_dot_product_attention or Flash Attention
  - Translate `rearrange(x, "B H L D -> B L (H D)")`

- [ ] `SelfAttention` module (model.py:167-179)
  - Q, K, V projections
  - QKNorm integration
  - Output projection

**Key einops patterns to handle:**
```python
# Pattern 1: Split and rearrange
rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=num_heads)

# Pattern 2: Merge heads
rearrange(x, "B H L D -> B L (H D)")
```

**Testing:**
- Attention output matches PyTorch
- Gradients match (if training)

#### 2.3: Transformer Blocks (1-2 days)

**Implement stream blocks:**

- [ ] `SingleStreamBlock` (model.py:229-282)
  - Pre-normalization with LayerNorm
  - Linear projections
  - MLP with SiLU gated activation
  - Modulation integration

- [ ] `DoubleStreamBlock` (model.py:285-382)
  - Dual-stream architecture (image + text)
  - Cross-attention between streams
  - Separate MLPs for each stream

**Critical einops translations:**
- `rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)`
- Stream concatenation and splitting

**Testing:**
- Block-level output parity
- Memory usage profiling

#### 2.4: Main Model (1 day)

**Implement `Flux2` main class:**

- [ ] `EmbedND` position embedder (model.py:396-409)
- [ ] Input/output projections
- [ ] Stack double blocks (depth=8 for Klein 9B)
- [ ] Stack single blocks (depth_single_blocks=24 for Klein 9B)
- [ ] `LastLayer` output projection (model.py:207-226)

**Model configurations:**
- [ ] `Klein9BParams` dataclass
- [ ] `Klein4BParams` dataclass (future)

**Testing:**
- Full forward pass with random inputs
- Compare with PyTorch model outputs
- Benchmark inference speed

### Phase 3: Autoencoder (2-3 days)

**Goal:** Port VAE encoder/decoder from `autoencoder.py`

#### 3.1: Basic CNN Blocks (1 day)

**Implement standard blocks:**

- [ ] `ResnetBlock` (autoencoder.py:54-81)
  - GroupNorm (use Burn's native implementation)
  - Conv2d with proper padding
  - Skip connections

- [ ] `AttnBlock` (autoencoder.py:24-51)
  - Self-attention on spatial dimensions
  - Scaled dot-product attention

- [ ] `Downsample` (autoencoder.py:84-94)
  - Conv2d with stride=2
  - Asymmetric padding workaround

- [ ] `Upsample` (autoencoder.py:97-105)
  - Nearest neighbor interpolation
  - Conv2d refinement

**Testing:**
- Per-block numerical parity
- Memory layout verification

#### 3.2: Encoder (0.5 days)

**Implement `Encoder` class:**

- [ ] Input convolution
- [ ] Downsampling blocks with ResNet + optional Attention
- [ ] Middle block with attention
- [ ] Output normalization and projection
- [ ] Quant conv for latent space

**einops patterns:**
```python
# Patchify operation
rearrange(mean, "... c (i pi) (j pj) -> ... (c pi pj) i j", pi=2, pj=2)
```

#### 3.3: Decoder (0.5 days)

**Implement `Decoder` class:**

- [ ] Post-quant conv
- [ ] Middle blocks
- [ ] Upsampling blocks
- [ ] Output projection

**einops patterns:**
```python
# Unpatchify operation
rearrange(z, "... (c pi pj) i j -> ... c (i pi) (j pj)", pi=2, pj=2)
```

#### 3.4: Full AutoEncoder (1 day)

**Integrate encoder + decoder:**

- [ ] `AutoEncoder` class
- [ ] BatchNorm normalization layer
- [ ] `encode()` method
- [ ] `decode()` method
- [ ] `normalize()` / `inv_normalize()` methods

**Testing:**
- Encode-decode round trip
- Latent space statistics
- Visual reconstruction quality

### Phase 4: Sampling & Inference (2-3 days)

**Goal:** Port denoising loop and utilities from `sampling.py`

#### 4.1: Image Preprocessing (1 day)

**Replace torchvision with Rust `image` crate:**

- [ ] `to_rgb()` - color space conversion
- [ ] `center_crop_to_multiple_of_x()` - crop to multiples of 16
- [ ] `cap_pixels()` - resize to max pixel count
- [ ] `cap_min_pixels()` - enforce min dimensions and aspect ratio
- [ ] `default_images_prep()` - normalize to [-1, 1]

**Testing:**
- Pixel-perfect match with PyTorch preprocessing
- Handle various image formats (PNG, JPEG, WebP)

#### 4.2: Position Encoding (0.5 days)

**Implement coordinate systems:**

- [ ] `prc_img()` - image token position IDs (sampling.py:141-151)
- [ ] `prc_txt()` - text token position IDs (sampling.py:93-103)
- [ ] `compress_time()` - temporal coordinate compression (sampling.py:12-21)
- [ ] `scatter_ids()` - position-based token scattering (sampling.py:24-49)

**einops translation:**
```python
rearrange(out, "(t h w) c -> 1 c t h w", t=t, h=h, w=w)
rearrange(x, "c h w -> (h w) c")
```

#### 4.3: Denoising Loop (1 day)

**Implement flow matching:**

- [ ] `get_schedule()` - timestep scheduling (sampling.py:244-248)
- [ ] `compute_empirical_mu()` - schedule parameters (sampling.py:251-266)
- [ ] `generalized_time_snr_shift()` - SNR shift (sampling.py:240-241)
- [ ] `denoise()` - main denoising loop (sampling.py:269-307)
- [ ] `denoise_cfg()` - classifier-free guidance variant (sampling.py:316-362)

**Testing:**
- Single denoising step output
- Full generation matches PyTorch

#### 4.4: Reference Image Encoding (0.5 days)

**Multi-image conditioning:**

- [ ] `encode_image_refs()` - encode multiple reference images (sampling.py:52-90)
- [ ] Handle single vs multi-image scenarios
- [ ] Time offset computation for multiple refs

### Phase 5: Model Loading & Safetensors (1 day)

**Goal:** Load PyTorch weights into Burn model

#### 5.1: Weight Loading (0.5 days)

**Implement safetensors loader:**

- [ ] Parse safetensors metadata
- [ ] Map PyTorch parameter names to Burn structure
- [ ] Handle shape mismatches (transpose Conv2d weights if needed)
- [ ] Convert dtypes (fp32, bf16, fp16)

**Example mapping:**
```rust
// PyTorch: model.img_in.weight [hidden_size, in_channels]
// Burn:    img_in.weight [in_channels, hidden_size]
```

#### 5.2: Model Initialization (0.5 days)

**Create model from weights:**

- [ ] Load Klein 9B configuration
- [ ] Initialize model structure
- [ ] Load weights from safetensors
- [ ] Verify all parameters loaded
- [ ] Set model to eval mode

**Testing:**
- Weight checksums match
- No missing/extra parameters
- Inference runs without errors

### Phase 6: Testing & Validation (2-3 days)

**Goal:** Ensure numerical parity and correctness

#### 6.1: Unit Tests (1 day)

**Per-component testing:**

- [ ] RoPE embeddings match PyTorch
- [ ] Attention output matches
- [ ] Layer normalization (LayerNorm, GroupNorm, RMSNorm)
- [ ] Activation functions (SiLU, GELU)
- [ ] Autoencoder encode/decode
- [ ] Image preprocessing pipeline

**Tolerance levels:**
- fp32: `rtol=1e-5, atol=1e-7`
- bf16: `rtol=1e-2, atol=1e-4`

#### 6.2: Integration Tests (1 day)

**End-to-end validation:**

- [ ] Generate image with fixed seed in both PyTorch and Burn
- [ ] Compare pixel-by-pixel (allow for minor numerical differences)
- [ ] Test various prompts and image sizes
- [ ] Test single-reference editing
- [ ] Test multi-reference editing

**Success criteria:**
- Generated images visually identical
- Numerical outputs within tolerance
- No crashes or panics

#### 6.3: Performance Benchmarking (1 day)

**Measure inference speed:**

- [ ] Single image generation (512x512, 1024x1024)
- [ ] Batch generation (if supported)
- [ ] Memory usage (VRAM)
- [ ] CPU vs GPU backend comparison
- [ ] Compare against PyTorch baseline

**Target metrics (Klein 9B):**
- **Latency:** <1s on modern GPU (RTX 4090, H100)
- **Memory:** ~9-12GB VRAM
- **Throughput:** Competitive with PyTorch

### Phase 7: CLI & User Interface (1-2 days)

**Goal:** Create usable command-line interface

#### 7.1: CLI Application (1 day)

**Implement using `clap` crate:**

```rust
flux2-burn generate \
  --prompt "a cat wearing a hat" \
  --model ./models/klein-9b.safetensors \
  --output output.png \
  --steps 4 \
  --size 1024x1024
```

**Commands:**
- [ ] `generate` - text-to-image
- [ ] `edit` - image editing with prompt
- [ ] `multi-edit` - multi-reference editing

**Options:**
- [ ] Model path
- [ ] Output path
- [ ] Number of steps (default: 4 for distilled)
- [ ] Guidance scale
- [ ] Image dimensions
- [ ] Seed for reproducibility
- [ ] Backend selection (cuda, cpu, etc.)

#### 7.2: Interactive Mode (optional, 0.5 days)

**REPL-style interface:**

- [ ] Load model once, keep in memory
- [ ] Interactive prompt input
- [ ] Quick iterations
- [ ] Preview mode

#### 7.3: Documentation (0.5 days)

**User documentation:**

- [ ] README with quick start
- [ ] Installation instructions
- [ ] Usage examples
- [ ] Performance tips
- [ ] Troubleshooting guide

### Phase 8: Optimization & Polish (1-2 days)

**Goal:** Performance tuning and production readiness

#### 8.1: Performance Optimization (1 day)

**Identify bottlenecks:**

- [ ] Profile with `cargo flamegraph`
- [ ] Check for unnecessary allocations
- [ ] Optimize einops translations (minimize copies)
- [ ] Leverage Burn's kernel fusion
- [ ] Test Flash Attention vs standard attention

**Memory optimization:**
- [ ] Use in-place operations where possible
- [ ] Optimize intermediate tensor lifecycle
- [ ] Consider quantization (int8, fp16)

#### 8.2: Error Handling (0.5 days)

**Robust error handling:**

- [ ] Graceful CUDA OOM handling
- [ ] Invalid input validation
- [ ] Model loading errors
- [ ] Helpful error messages

#### 8.3: Code Quality (0.5 days)

**Final polish:**

- [ ] Run `cargo clippy` - fix all warnings
- [ ] Format with `cargo fmt`
- [ ] Add documentation comments
- [ ] Write examples in `examples/`
- [ ] CI/CD setup (GitHub Actions)

---

## Risk Assessment & Mitigation

### High Risk Areas

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Einops translation errors** | High | Medium | Extensive unit testing, visual comparison |
| **Numerical divergence** | High | Medium | Strict tolerance testing, double-precision debug mode |
| **Missing Burn features** | High | Low | Already verified all features exist |
| **Performance regression** | Medium | Medium | Benchmark continuously, leverage Flash Attention |
| **CUDA compatibility** | Medium | Low | Test on multiple GPU architectures |

### Medium Risk Areas

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **safetensors compatibility** | Medium | Low | Test with multiple model checkpoints |
| **Image preprocessing differences** | Medium | Medium | Pixel-perfect validation |
| **Memory leaks** | Medium | Low | Use Rust's ownership system, profile memory |
| **Cross-platform issues** | Low | Medium | CI testing on Linux, macOS, Windows |

---

## Success Criteria

### Must Have (MVP)
- ✅ Load Klein 9B model weights
- ✅ Generate images from text prompts
- ✅ Output matches PyTorch within 1% pixel error
- ✅ Run on CUDA GPUs
- ✅ Inference time ≤ 1.5x PyTorch baseline

### Should Have
- ✅ Image editing (single reference)
- ✅ CLI with common options
- ✅ CPU backend support
- ✅ Comprehensive test suite
- ✅ Documentation

### Nice to Have
- ⭐ Multi-reference editing
- ⭐ Quantization (int8, fp16)
- ⭐ Batch inference
- ⭐ Python bindings (PyO3)
- ⭐ WebGPU backend for web deployment

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. Project Setup | 1 day | None |
| 2. Core Model | 3-4 days | Phase 1 |
| 3. Autoencoder | 2-3 days | Phase 1 |
| 4. Sampling | 2-3 days | Phases 2, 3 |
| 5. Model Loading | 1 day | Phases 2, 3 |
| 6. Testing | 2-3 days | Phases 2-5 |
| 7. CLI | 1-2 days | Phase 6 |
| 8. Optimization | 1-2 days | Phase 7 |
| **Total** | **13-20 days** | - |

**Realistic estimate for experienced developer:** 2-3 weeks
**With learning curve (new to Burn):** 3-4 weeks

---

## Resources & References

### Official Documentation
- [Burn Book](https://burn.dev/book/)
- [Burn API Docs](https://burn.dev/docs/burn/)
- [CubeCL GitHub](https://github.com/tracel-ai/cubecl)
- [Burn Release Notes 0.20.0](https://burn.dev/blog/release-0.20.0/)

### Key API References
- [GroupNorm](https://docs.rs/burn/latest/burn/nn/modules/norm/struct.GroupNorm.html)
- [MultiHeadAttention](https://burn.dev/docs/burn/nn/attention/struct.MultiHeadAttention.html)
- [Tensor Operations](https://burn.dev/docs/burn/tensor/struct.Tensor.html)

### Community Resources
- [Burn GitHub Discussions](https://github.com/tracel-ai/burn/discussions)
- [Burn Discord](https://discord.gg/uPEBbYYDB6)
- [Rust ML Community](https://www.arewelearningyet.com/)

### Related Projects
- [Stable Diffusion Burn](https://github.com/Gadersd/stable-diffusion-burn) - Reference implementation
- [Stable Diffusion XL Burn](https://github.com/Gadersd/stable-diffusion-xl-burn) - Advanced example

---

## Conclusion

Porting FLUX.2 Klein to Rust/Burn is **highly feasible** with the recent 0.20.0 release. The main challenge is translating einops patterns, which is mechanical but time-consuming. The built-in Flash Attention and GroupNorm support significantly reduces complexity.

**Key advantages of Burn port:**
1. **Performance:** Flash Attention + CubeCL optimizations
2. **Safety:** Rust's memory safety eliminates segfaults
3. **Portability:** Single codebase for CPU, CUDA, ROCm, Metal, WebGPU
4. **Deployment:** Easier to embed in production systems
5. **Type safety:** Compile-time shape checking (to some extent)

**Recommended approach:**
- Start with Phase 1-2 to validate feasibility
- Implement Phase 3-4 in parallel if resources allow
- Prioritize testing throughout (Phase 6)
- Polish CLI and docs last (Phases 7-8)

This port would be an excellent showcase for Burn's capabilities and could serve as a reference for future vision model ports.
