# Quintessentia 2.0.0 validation — 2026-11-24

This release promotes **pure Q1 binary-halving cubes (V4)** to the default CUDA execution mode (`--weight-format auto` -> `q1_cubes`, `k=5`, `root_bits=8`), broadens universal model and quantized-source support to **37 / 47 verified local checkpoints across 26 `model_type` families**, and ships one-button launchers (`./quint`, `quint.cmd`) with anti-loop default generation profiles. Model support remains checkpoint-specific; explicit `q1`/`q2`/`q4`/`q6`/`q8`/`bf16`, `--feedback-mode off`, and `--device cpu` paths remain intact.

## 1. Regression & CI coverage

All tests were executed in WSL 2 Linux (`Python 3.12`, `PyTorch 2.6 + CUDA 12.6`, `RTX 3060 Ti 8 GB`):

- **CPU-only CI contract (`CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1`):**
  - `python -m compileall -q quintessentia_cli src tests quintessentia.py` — exit `0`.
  - `python quintessentia.py --version` — `Quintessentia 2.0.0`.
  - Full suite (421 tests collected, matching GitHub Actions `ubuntu-latest` without optional `fla` Triton): **233 passed, 188 skipped (CUDA-only), 0 failed**.
  - Two Linux-specific issues uncovered and fixed during pre-release verification:
    1. Replaced anonymous `mmap.mmap(-1, ...)` page-aligned buffer allocation in Linux `O_DIRECT` staging (`quintessentia_cli/model_source.py`) with a pure 4096-byte-aligned `bytearray` slice (`zero mmap`).
    2. Switched CPU single-token MoE dispatch in `_fast_moe_dispatch` to uniform `torch.bmm` per expert so masked zero-weight routes remain bit-identical (`rtol=0, atol=0`) across batch sizes on Linux AVX2 BLAS.
- **Core V4, Q1, CLI, LAN, Quantized Source, Download/Prepare & VRAM Plan suites (GPU enabled):** **99 passed, 0 failed**.
- **Full 45-module GPU sweep (421 tests collected):** **401 passed, 10 skipped**, with 10 failures strictly isolated to two environment/legacy items outside the Q1 V4 runtime: (a) 2 CPU-mode `test_model_source` cases invoking optional `fla` Triton in the local WSL venv (both pass without `fla`), (b) 1 `O_DIRECT` zero-`mmap` test and 1 `test_route_exclusion` CPU ULP case both fixed and re-verified green above, and (c) 6 legacy `test_thought_warmup` synthetic `olmoe` entropy-gate cases sensitive to Linux CUDA 12.6 vs Windows CUDA 12.4 RNG/reduction ordering (`granitemoe` passed 6/6).

## 2. 47-Checkpoint Zoo Census (Pure Q1 Cubes V4, `k=5`)

Hardware: **NVIDIA GeForce RTX 3060 Ti 8 GB**, WSL 2 native `ext4` storage, `--weight-format auto` (`q1_cubes`, `k=5`, `root_bits=8`, `vram_mode=normal`), two sequential chat turns per model (`max_new_tokens=48`, `seed=42`, `enable_thinking=False`).

- **Census outcome:** **37 passed, 10 failed** (up from 31/46 in 1.2.0), covering **26 unique `model_type` families** (`llama`, `qwen2`, `qwen3`, `qwen3_5_text`, `qwen3_5_moe_text`, `gemma2`, `gemma3_text`, `recurrent_gemma`, `phi3`, `cohere2`, `olmo2`, `nemotron`, `granite`, `granitemoe`, `granitemoehybrid`, `smollm3`, `ernie4_5`, `lfm2`, `gpt2`, `gpt_neox`, `bloom`, `mamba`, `mamba2`, `zamba2`, `falcon_h1`, `gpt_oss`).
- Machine-readable results: [model_support_2_0_0.json](model_support_2_0_0.json). Full table: [MODEL_SUPPORT.md](../MODEL_SUPPORT.md).

### Verified scaling curve (124M → 35B MoE on 8 GB VRAM)

| Scale | Checkpoint | Disk | Placement | Decode (T2) | Peak VRAM | Notes |
|---|---|---:|---|---:|---:|---|
| 124M | `37_hf_gpt2-124m` (`Conv1D`) | 0.51 GiB | 12/12 res | **74.31 tok/s** | 162 MiB | Native `Conv1D` Q1 (+30% vs BF16 `57.1 tok/s`) |
| 1B | `01_gguf_Llama-3.2-1B-Instruct-Q4_K_M` | 0.75 GiB | 16/16 res | **39.79 tok/s** | 888 MiB | Clean English & Python |
| 1.2B Hybrid | `31_hf_Zamba2-1.2B-instruct-v2` | 2.27 GiB | 38/38 res | **14.22 tok/s** | 1,185 MiB | Shared transformer via `_tied_weights_keys` |
| 2B Griffin | `33_hf_recurrentgemma-2b-it-8bit` | 3.15 GiB | 26/26 res | **15.89 tok/s** | 1,867 MiB | Direct bitsandbytes 8-bit `SCB` + `cache` fix |
| 4B | `11_gguf_Nemotron-Mini-4B-IQ3_M` | 2.03 GiB | 32/32 res | **14.07 tok/s** | 2,594 MiB | Clean English & Python |
| 7B | `08_gguf_command-r7b-12-2024-IQ3_XS` | 3.47 GiB | 32/32 res | **15.68 tok/s** | 5,997 MiB | 7B fully resident in 6.0 GiB |
| 9B BF16 | `Qwythos-9B-v2-MTP-BF16` (`k=5`) | 17.14 GiB | 32/32 res | **19.91 tok/s** | 6,044 MiB | Full 9B BF16 resident on 8 GB GPU |
| 9B BF16 | `Qwythos-9B-v2-MTP-BF16` (`k=6`, streamed) | 17.14 GiB | 27/32 + 5 V4 | **5.99 tok/s** | ~6.4 GiB | Adaptive V4 streaming, 0 OOM |
| 20B MoE | `45_hf_gpt-oss-20b-MXFP4` | 12.84 GiB | 24/24 trunk + MoE | **0.85 tok/s** | 3,793 MiB | Packed HF MXFP4 experts on demand |
| 35B MoE | `Qwen3.6-35B-A3B` (256 experts) | 23.87 GiB | 40/40 trunk + Q1 MoE | **1.91 tok/s** | 5,567 MiB | Warm prefill `3.95 s`, 0 OOM |

### Layer-15 Q1 reconstruction & kernel micro-benchmark (`Qwythos-9B` BF16)

- Progressive pure-Q1 reconstruction cosine similarity: `k=4`: `0.9961`, `k=5`: `0.9991`, `k=6`: `0.9998`, `k=8`: `0.99998`, `k=16`: `1.000000`.
- Single-pass CUDA Ballot carving (`Kernel #7`): `5.1–5.7 ms` per layer (vs `1,228–1,316 ms` Python loop).
- Bank-conflict-free `POPC10 r2` GEMV + cached native C launch (`_fast_k6`): `3.9×–7.3×` faster than prior Q1 GEMV, raising end-to-end 9B BF16 decode to `19.91 tok/s` (`k=5`) and 1B decode to `39.79 tok/s`.

## 3. Live CLI smoke verification (`quint` default `--weight-format auto`)

- `./quint doctor --model <9B-BF16>`: reported `Backend: unified-q1-cubes / q1_cubes (k=5, root_bits=8)`, `V4 Q1 Cubes: ON (reason=auto_q1_cubes_v4)`, `all layers resident`.
- `./quint ask` (zero format flags): verified clean end-to-end generation on both `Llama-3.2-1B` (`887.2 MiB` peak VRAM) and `Qwythos-9B-BF16` (`6,050.5 MiB` peak VRAM, natural `EOS=True`).

## 4. Remaining boundaries

- The 10 unpassed checkpoints in the 47-model zoo fall into four clear external/unmapped buckets: unmapped GGUF architectures (`gemma3n`, `bitnet`, `minicpm3`), custom `bitnet.cpp` `ggml_type=36` (`I2_S`), multimodal-only VLM config without `AutoModelForCausalLM` (`Qwen2-VL-2B`), and external `trust_remote_code` / legacy pickle `.bin` repositories (`rwkv7`, `xlstm`, `RWKV4-bin`, `OpenELM`, `internlm2_5`).
- Source quantization quality still bounds output quality (`IQ1_S` and `IQ3_XXS` source weights degrade text regardless of Q1 bit-plane depth).
- Multimodal checkpoints execute text decoding only; vision and audio towers are skipped.
