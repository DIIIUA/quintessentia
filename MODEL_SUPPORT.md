# Model support in Quintessentia 2.0.0

**Support is checkpoint-specific.** The fresh **2026-11-24** local census tested all **47** checkpoints in the test zoo on the **V4 pure Q1 cubes** runtime (`--weight-format auto` -> `q1_cubes`, `k=5`, `root_bits=8`) on an **NVIDIA GeForce RTX 3060 Ti 8 GB** (WSL 2 native `ext4`): **37 passed**, **10 failed** (up from 31/46 in 1.2.0). Machine-readable per-checkpoint outcomes are recorded in [benchmarks/model_support_2_0_0.json](benchmarks/model_support_2_0_0.json); full release validation is in [benchmarks/RELEASE_2_0_0.md](benchmarks/RELEASE_2_0_0.md).

Each passing entry completed engine initialization, two sequential chat turns (`max_new_tokens=48`, `seed=42`: GPU cache explanation + Python binary search), and finite output generation with zero OOM. Base (non-instruct) models and ultra-aggressive source quantizations (`IQ1_S`, `IQ3_XXS`) are noted honestly in the remarks column.

| # | Checkpoint | Result | Architecture (`model_type`) | Disk | Placement | Decode (T2) | Peak VRAM | Notes / failure detail |
| :-: | :--- | :---: | :--- | ---: | :---: | ---: | ---: | :--- |
| 1 | `01_gguf_Llama-3.2-1B-Instruct-Q4_K_M` | passed | `llama` | 0.75G | 16/16 res | 39.79 tok/s | 888M | Clean English & Python |
| 2 | `02_gguf_SmolLM3-3B-IQ3_XXS` | passed | `smollm3` | 1.18G | 36/36 res | 14.08 tok/s | 2,548M | Degraded source `IQ3_XXS` text quality |
| 3 | `03_gguf_Phi-4-mini-instruct-IQ3_XXS` | passed | `phi3` | 1.56G | 32/32 res | 16.73 tok/s | 3,039M | Clean English & Python |
| 4 | `04_gguf_granite-3.3-2b-instruct-Q4_K_M` | passed | `granite` | 1.44G | 40/40 res | 11.75 tok/s | 2,183M | Coherent output |
| 5 | `05_gguf_gemma-3-1b-it-Q4_K_M` | passed | `gemma3_text` | 0.75G | 26/26 res | 10.26 tok/s | 700M | Clean English & Python |
| 6 | `06_gguf_gemma-2-2b-it-Q4_K_M` | passed | `gemma2` | 1.59G | 26/26 res | 13.32 tok/s | 1,802M | Clean English & Python |
| 7 | `07_gguf_OLMo-2-0425-1B-Q4_K_M` | passed | `olmo2` | 0.87G | 16/16 res | 24.82 tok/s | 899M | Clean English & Python |
| 8 | `08_gguf_command-r7b-12-2024-IQ3_XS` | passed | `cohere2` | 3.47G | 32/32 res | 15.68 tok/s | 5,997M | Clean Python (7B fully resident) |
| 9 | `09_gguf_gemma-3n-E2B-it-Q2_K_L` | failed | — | 2.07G | — | — | — | load: `UnsupportedModelError`: GGUF architecture `'gemma3n'` is not yet mapped |
| 10 | `10_gguf_MiniCPM3-4B-MLA-Q4_K_M` | failed | — | 2.30G | — | — | — | load: `UnsupportedModelError`: GGUF architecture `'minicpm3'` is not in core `transformers` |
| 11 | `11_gguf_Nemotron-Mini-4B-IQ3_M` | passed | `nemotron` | 2.03G | 32/32 res | 14.07 tok/s | 2,594M | Clean English & Python |
| 12 | `12_gguf_Qwen3.5-0.8B-Q4_K_M` | passed | `qwen3_5_text` | 0.50G | 24/24 res | 20.90 tok/s | 573M | Clean English & Python |
| 13 | `13_gguf_Falcon-H1-0.5B-Q4_K_M` | passed | `falcon_h1` | 0.29G | 36/36 res | 9.59 tok/s | 510M | Parallel Mamba2 + Attention hybrid (newly unlocked) |
| 14 | `14_gguf_granite-4.0-tiny-hybrid-IQ3_XXS` | passed | `granitemoehybrid` | 2.55G | 40/40 res | 3.28 tok/s | 2,885M | 62-expert Mamba2 + MoE hybrid (newly unlocked) |
| 15 | `15_gguf_LFM2-350M-Q4_K_M` | passed | `lfm2` | 0.21G | 16/16 res | 29.21 tok/s | 360M | Clean English & Python |
| 16 | `16_gguf_Quant_SmolLM2-135M-Q2_K` | passed | `llama` | 0.08G | 30/30 res | 18.18 tok/s | 161M | Coherent Python for 135M `Q2_K` |
| 17 | `17_gguf_Quant_Qwen2.5-0.5B-Q3_K_M` | passed | `qwen2` | 0.40G | 24/24 res | 19.29 tok/s | 417M | Clean English & Python |
| 18 | `18_gguf_Quant_Qwen2.5-0.5B-Q4_0` | passed | `qwen2` | 0.40G | 24/24 res | 25.65 tok/s | 377M | Clean English & Python |
| 19 | `19_gguf_Quant_Llama-3.2-1B-Q4_1` | passed | `llama` | 0.77G | 16/16 res | 32.44 tok/s | 1,044M | Clean English & Python |
| 20 | `20_gguf_Quant_SmolLM2-135M-IQ4_XS` | passed | `llama` | 0.08G | 30/30 res | 15.07 tok/s | 161M | Clean Python |
| 21 | `21_gguf_Quant_Llama-3.2-1B-IQ4_NL` | passed | `llama` | 0.72G | 16/16 res | 26.85 tok/s | 1,044M | Clean English & Python |
| 22 | `22_gguf_Quant_Llama-3.2-1B-IQ2_XXS` | passed | `llama` | 0.43G | 16/16 res | 31.21 tok/s | 1,044M | Clean Python |
| 23 | `23_gguf_Quant_Llama-3.2-1B-IQ1_S` | passed | `llama` | 0.39G | 16/16 res | 26.82 tok/s | 1,044M | Degraded source 1-bit `IQ1_S` weights |
| 24 | `24_gguf_BitNet-b1.58-2B-TQ2_0` | failed | — | 1.12G | — | — | — | load: `UnsupportedModelError`: GGUF architecture `'bitnet'` is not yet mapped |
| 25 | `25_gguf_BitNet-b1.58-2B-i2_s` | failed | — | 1.11G | — | — | — | load: `UnsupportedModelError`: custom `bitnet.cpp` ggml type `36` (`I2_S`) |
| 26 | `26_hf_Qwen3-0.6B-FP8` | passed | `qwen3` | 1.00G | 28/28 res | 12.70 tok/s | 469M | Block FP8 -> pure Q1 cubes |
| 27 | `27_hf_ERNIE-4.5-0.3B-PT` | passed | `ernie4_5` | 0.68G | 18/18 res | 20.37 tok/s | 427M | Clean English & Python |
| 28 | `28_hf_granite-3.1-1b-a400m-MoE` | passed | `granitemoe` | 2.49G | 24/24 res | 10.50 tok/s | 900M | Clean English & Python (Q1 MoE) |
| 29 | `29_hf_mamba-130m` | passed | `mamba` | 0.48G | 24/24 res | 22.09 tok/s | 163M | Base SSM 130M (native Q1 cubes) |
| 30 | `30_hf_mamba2-370m` | passed | `mamba2` | 0.69G | 48/48 res | 5.27 tok/s | 487M | Base SSM 370M (native Q1 cubes) |
| 31 | `31_hf_Zamba2-1.2B-instruct-v2` | passed | `zamba2` | 2.27G | 38/38 res | 14.22 tok/s | 1,185M | Shared transformer via `_tied_weights_keys` (newly unlocked) |
| 32 | `32_hf_rwkv7-0.1B-g1` | failed | — | 0.36G | — | — | — | load: requires `trust_remote_code=True` (external repository code) |
| 33 | `33_hf_recurrentgemma-2b-it-8bit` | passed | `recurrent_gemma` | 3.15G | 26/26 res | 15.89 tok/s | 1,867M | Direct bitsandbytes 8-bit `SCB` + `cache` fix (newly unlocked) |
| 34 | `34_hf_xlstm-german-wikipedia` | failed | — | 0.42G | — | — | — | load: `UnsupportedModelError`: non-standard config schema / external `xlstm` code |
| 35 | `35_hf_RWKV4-169m-bin` | failed | — | 0.63G | — | — | — | load: `UnsupportedModelError`: legacy pickle `.bin` only (no safetensors/GGUF) |
| 36 | `36_hf_OpenELM-270M-Instruct` | failed | — | 0.51G | — | — | — | load: `UnsupportedModelError`: non-standard config schema / `trust_remote_code` |
| 37 | `37_hf_gpt2-124m` | passed | `gpt2` | 0.51G | 12/12 res | 74.31 tok/s | 162M | Base model; native `Conv1D` -> Q1 cubes (+30% vs BF16) |
| 38 | `38_hf_bloom-560m` | passed | `bloom` | 1.06G | 24/24 res | 27.06 tok/s | 416M | Base model; repetitive token degradation under chat prompt |
| 39 | `39_hf_pythia-410m` | passed | `gpt_neox` | 0.85G | 24/24 res | 29.95 tok/s | 345M | Base model (coherent continuation) |
| 40 | `40_hf_SmolVLM2-256M` | passed | `llama` | 0.96G | 30/30 res | 15.34 tok/s | 149M | Text decoder branch -> Q1 cubes |
| 41 | `41_hf_Qwen2-VL-2B-Instruct-AWQ` | failed | — | 2.75G | — | — | — | load: `UnsupportedModelError`: `Qwen2VLConfig` has no `AutoModelForCausalLM` class |
| 42 | `42_hf_Qwen2.5-0.5B-AWQ` | passed | `qwen2` | 0.69G | 24/24 res | 16.68 tok/s | 377M | AWQ INT4 -> pure Q1 cubes |
| 43 | `43_hf_Qwen2.5-0.5B-GPTQ-Int4` | passed | `qwen2` | 0.44G | 24/24 res | 23.91 tok/s | 377M | GPTQ INT4 -> pure Q1 cubes |
| 44 | `44_hf_internlm2_5-1_8b-chat` | failed | — | 3.52G | — | — | — | load: `UnsupportedModelError`: custom `InternLM2Config` requires `trust_remote_code` |
| 45 | `45_hf_gpt-oss-20b-MXFP4` | passed | `gpt_oss` | 12.84G | 24/24 + MoE | 0.85 tok/s | 3,793M | 20B packed MXFP4 MoE in 3.8 GiB VRAM (newly unlocked) |
| 46 | `Qwen3.6-35B-A3B-uncensored-heretic-APEX-Balanced` | passed | `qwen3_5_moe_text` | 23.87G | 40/40 + MoE | 1.91 tok/s | 5,567M | 35B 256-expert MoE in 5.5 GiB VRAM |
| 47 | `Qwythos-9B-v2-MTP-BF16` | passed | `qwen3_5_text` | 17.14G | 32/32 res | 19.91 tok/s | 6,044M | 9B BF16 fully resident at `k=5` (~20 tok/s) |

## Historical 1.2.0 baseline

The earlier 2026-11-18 Q6 smoke census (31/46 passed) is preserved in [benchmarks/model_support_1_2_0.json](benchmarks/model_support_1_2_0.json), [benchmarks/RELEASE_1_2_0.md](benchmarks/RELEASE_1_2_0.md), and [benchmarks/CLI_MODELS_20261118.md](benchmarks/CLI_MODELS_20261118.md).

## Boundaries

- `failed` records an actual unresolved error; checkpoints requiring `trust_remote_code`, legacy pickle `.bin` files, or multimodal-only model classes without `AutoModelForCausalLM` are intentionally excluded.
- Source GGUF, FP8, AWQ, GPTQ, bitsandbytes 8-bit, and MXFP4 weights are adapted on demand to the runtime representation (`q1_cubes` by default on CUDA). Ultra-low-bit source quantizations (`IQ1_S`, `IQ3_XXS`) limit text quality regardless of runtime depth.
- Multimodal checkpoints exercise only the text decoder; image and audio towers are skipped.
- Native MTP speculation targets compatible trained Qwen3.5 text heads; other families use prompt lookup or standard decode.
