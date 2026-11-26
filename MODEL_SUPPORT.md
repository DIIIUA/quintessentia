# Model support in Quintessentia 2.2.0

**Support is checkpoint-specific.** Building on the **2026-11-26** local census (isolated **3-turn suite**: Run 1 cold factual greedy, Run 2 direct multi-turn KV/SSM prefix continuation, Run 3 post-reset structured generation at `temp=0.25, rep=1.05`) plus **2026-11-27** verification of **80B-A3B `qwen3_next`** and **resident Q1 native MTP** on an **NVIDIA GeForce RTX 3060 Ti 8 GB** (WSL 2 `ext4`) under **V4 pure Q1 cubes + 4-layer RSSF balance**:

- **40 / 46** present weight folders passed end-to-end multi-turn execution with **0 OOM, 0 timeouts, and 0 crashes** (38 clean `PASS` including **80B-A3B** + 2 base-model greedy-repeat notices on `gpt2-124m` and `bloom-560m` that generate cleanly under R3 sampling).
- **Newly verified in 2.2.0:**
  - **`Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated.i1-Q4_K_M` (`qwen3_next`, 46 GiB, 48 layers, 512 experts/layer = 24,576 experts):** fixed `qwen3_next` GGUF grouped SSM head order across all 36 GatedDeltaNet layers; boots in **4.82 s** with all 48 trunk layers resident and generates coherently in **5.38–5.93 GiB VRAM** (`2.61 / 2.50 tok/s`, `kv2_hit=182`, `8.96–14.53%` experts carved on demand).
  - **`Qwythos-9B-v2-MTP-BF16` (`qwen3_5_text`, 17.14 GiB):** native resident Q1 MTP (`blk.32.nextn.*`) + two-stage SWAR `__dp4a` multi-row CUDA kernels accelerated greedy 256-token decode from `17.61 tok/s` to **`21.17 tok/s`** (MTP off) and **`33.70 tok/s`** (auto-MTP `max_drafts=3`, `75.7%` accept, 0 replays, bit-exact `256/256` match — **1.91×**).
- **4 checkpoints unlocked in 2.1.0:** `10_gguf_MiniCPM3-4B-MLA-Q4_K_M` (62-layer MLA), `24_gguf_BitNet-b1.58-2B-TQ2_0`, `25_gguf_BitNet-b1.58-2B-i2_s` (native `I2_S` type 36), and `41_hf_Qwen2-VL-2B-Instruct-AWQ` (Qwen2 text backbone).
- **5** expected `UnsupportedModelError` rejections (`gemma3n`, `rwkv7`, `xlstm`, `OpenELM`, `internlm2`), **1** legacy pickle `.bin`-only folder (`RWKV4-169m-bin`), and **1** empty 0-byte transfer folder (`14_gguf_granite-4.0-tiny-hybrid-IQ3_XXS`, previously verified passing in 2.0.0; `13_gguf_Falcon-H1-0.5B-Q4_K_M` was verified in 2.0.0).
- Machine-readable 3-run census baseline: [benchmarks/model_support_2_1_0.json](benchmarks/model_support_2_1_0.json).

| # | Checkpoint | Result | Arch (`model_type`) | Disk | Q1 / Placement | Decode R1/R2/R3 (tok/s) | KV2 Hit | Peak VRAM | Notes |
| :-: | :--- | :---: | :--- | ---: | :---: | :---: | ---: | ---: | :--- |
| 1 | `01_gguf_Llama-3.2-1B-Instruct-Q4_K_M` | passed | `llama` | 0.75G | `k=8` 16/16 | 14.1 / 14.6 / 39.2 | 56 | 1,391M | Clean Paris / Eiffel / tips (`k=8` auto-refined) |
| 2 | `02_gguf_SmolLM3-3B-IQ3_XXS` | passed | `smollm3` | 1.18G | `k=5` 36/36 | 9.3 / 13.8 / 16.2 | 88 | 2,334M | Clean Paris / Eiffel / tips (fixed `LlamaRopeReversePermute`) |
| 3 | `03_gguf_Phi-4-mini-instruct-IQ3_XXS` | passed | `phi3` | 1.56G | `k=5` 32/32 | 8.7 / 17.0 / 16.4 | 18 | 3,043M | Clean crisp EOS stop (fixed `<|end|>` EOT) |
| 4 | `04_gguf_granite-3.3-2b-instruct-Q4_K_M` | passed | `granite` | 1.44G | `k=5` 40/40 | 12.8 / 17.5 / 13.4 | 79 | 2,022M | Clean Paris / Eiffel / tips (fixed RoPE permute + `refact` pre-tok) |
| 5 | `05_gguf_gemma-3-1b-it-Q4_K_M` | passed | `gemma3_text` | 0.75G | `k=8` 26/26 | 14.5 / 16.6 / 16.6 | 30 | 1,075M | Clean Paris / Eiffel / tips (`k=8` auto-refined) |
| 6 | `06_gguf_gemma-2-2b-it-Q4_K_M` | passed | `gemma2` | 1.59G | `k=5` 26/26 | 7.5 / 5.2 / 15.9 | 0 | 1,988M | Clean crisp EOS stop (fixed `<end_of_turn>` EOT) |
| 7 | `07_gguf_OLMo-2-0425-1B-Q4_K_M` | passed | `olmo2` | 0.87G | `k=8` 16/16 | 13.9 / 27.6 / 20.5 | 31 | 1,429M | Clean Paris / Eiffel / tips (`k=8` auto-refined) |
| 8 | `08_gguf_command-r7b-12-2024-IQ3_XS` | passed | `cohere2` | 3.47G | `k=5` 32/32 | 12.6 / 19.8 / 18.6 | 30 | 5,953M | 7B fully resident; crisp EOS stop (fixed `<|END_RESPONSE|>`) |
| 9 | `09_gguf_gemma-3n-E2B-it-Q2_K_L` | failed | — | 2.07G | — | — | — | — | load: `UnsupportedModelError`: GGUF architecture `'gemma3n'` is not yet mapped |
| 10 | `10_gguf_MiniCPM3-4B-MLA-Q4_K_M` | passed | `minicpm3` | 2.30G | `k=5` 62/62 | 5.3 / 6.7 / 6.0 | 26 | 2,943M | **Newly unlocked** 62-layer MLA fully resident + asymmetric MLA KV cache |
| 11 | `11_gguf_Nemotron-Mini-4B-IQ3_M` | passed | `nemotron` | 2.03G | `k=5` 32/32 | 20.0 / 21.7 / 22.1 | 31 | 2,576M | Clean Paris / Eiffel / tips |
| 12 | `12_gguf_Qwen3.5-0.8B-Q4_K_M` | passed | `qwen3_5_text` | 0.50G | `k=8` 24/24 | 8.2 / 7.8 / 27.3 | 32 | 855M | Clean Paris / Eiffel / tips (`k=8` auto-refined) |
| 13 | `14_gguf_granite-4.0-tiny-hybrid-IQ3_XXS` | skipped | — | 0.00G | — | — | — | — | 0-byte empty folder on disk (verified passing in 2.0.0: 40/40 res, 3.28 tok/s) |
| 14 | `15_gguf_LFM2-350M-Q4_K_M` | passed | `lfm2` | 0.21G | `k=8` 16/16 | 14.0 / 13.6 / 48.2 | 30 | 490M | Clean Paris / Eiffel / tips (48.2 tok/s at `k=8`) |
| 15 | `16_gguf_Quant_SmolLM2-135M-Q2_K` | passed | `llama` | 0.08G | `k=8` 30/30 | 6.8 / 12.2 / 19.8 | 50 | 203M | Coherent 135M `Q2_K` (`k=8` auto-refined) |
| 16 | `17_gguf_Quant_Qwen2.5-0.5B-Q3_K_M` | passed | `qwen2` | 0.40G | `k=8` 24/24 | 8.9 / 9.2 / 22.0 | 49 | 555M | Clean Paris / Eiffel / tips (`k=8` auto-refined) |
| 17 | `18_gguf_Quant_Qwen2.5-0.5B-Q4_0` | passed | `qwen2` | 0.40G | `k=8` 24/24 | 10.0 / 9.5 / 28.6 | 49 | 555M | Clean Paris / Eiffel / tips (`k=8` auto-refined) |
| 18 | `19_gguf_Quant_Llama-3.2-1B-Q4_1` | passed | `llama` | 0.78G | `k=8` 16/16 | 15.9 / 20.3 / 40.8 | 56 | 1,391M | Clean Paris / Eiffel / tips (40.8 tok/s at `k=8`) |
| 19 | `20_gguf_Quant_SmolLM2-135M-IQ4_XS` | passed | `llama` | 0.09G | `k=8` 30/30 | 6.8 / 13.0 / 14.4 | 50 | 203M | Coherent 135M `IQ4_XS` (`k=8` auto-refined) |
| 20 | `21_gguf_Quant_Llama-3.2-1B-IQ4_NL` | passed | `llama` | 0.72G | `k=8` 16/16 | 11.5 / 17.1 / 40.2 | 56 | 1,391M | Clean Paris / Eiffel / tips (40.2 tok/s at `k=8`) |
| 21 | `22_gguf_Quant_Llama-3.2-1B-IQ2_XXS` | passed | `llama` | 0.43G | `k=8` 16/16 | 21.3 / 14.8 / 36.2 | 56 | 1,391M | Coherent 2-bit `IQ2_XXS` (36.2 tok/s at `k=8`) |
| 22 | `23_gguf_Quant_Llama-3.2-1B-IQ1_S` | passed | `llama` | 0.39G | `k=8` 16/16 | 11.0 / 12.8 / 27.4 | 91 | 1,391M | Degraded source 1.56-bit `IQ1_S` weights (`k=8` auto-refined) |
| 23 | `24_gguf_BitNet-b1.58-2B-TQ2_0` | passed | `bitnet` | 1.12G | `k=6` 30/30 | 7.8 / 12.3 / 11.4 | 27 | 2,107M | **Newly unlocked** ternary BitNet 1.58b (`TQ2_0`), clean Paris / Eiffel / tips |
| 24 | `25_gguf_BitNet-b1.58-2B-i2_s` | passed | `bitnet` | 1.11G | `k=6` 30/30 | 14.5 / 17.0 / 15.7 | 72 | 2,108M | **Newly unlocked** native `bitnet.cpp` `I2_S` (ggml type 36) decoder |
| 25 | `26_hf_Qwen3-0.6B-FP8` | passed | `qwen3` | 0.99G | `k=8` 28/28 | 7.7 / 6.1 / 15.6 | 32 | 657M | Block FP8 -> Q1 cubes (`k=8`), clean Paris / Eiffel / tips |
| 26 | `27_hf_ERNIE-4.5-0.3B-PT` | passed | `ernie4_5` | 0.67G | `k=8` 18/18 | 25.6 / 32.2 / 31.9 | 27 | 526M | Clean Paris / Eiffel / tips (32.2 tok/s at `k=8`) |
| 27 | `28_hf_granite-3.1-1b-a400m-MoE` | passed | `granitemoe` | 2.49G | `k=5` 24/24 +MoE | 3.6 / 6.0 / 10.9 | 80 | 910M | Clean Paris / Eiffel / tips (Q1 MoE) |
| 28 | `29_hf_mamba-130m` | passed | `mamba` | 0.48G | `k=8` 24/24 | 19.8 / 14.4 / 28.3 | 0 | 236M | Base SSM 130M (`k=8` auto-refined) |
| 29 | `30_hf_mamba2-370m` | passed | `mamba2` | 0.69G | `k=8` 48/48 | 14.2 / 16.3 / 10.1 | 0 | 543M | Base SSM 370M (`k=8`); coherent Paris & Eiffel continuation |
| 30 | `31_hf_Zamba2-1.2B-instruct-v2` | passed | `zamba2` | 2.26G | `k=8` 38/38 | 17.1 / 15.6 / 11.9 | 31 | 1,478M | Clean 3/3 + fixed hybrid SSM/conv unpark (`kv2_hit=31`) |
| 31 | `32_hf_rwkv7-0.1B-g1` | failed | — | 0.36G | — | — | — | — | load: `UnsupportedModelError` (`auto_map` / `trust_remote_code=False`, safe fail) |
| 32 | `33_hf_recurrentgemma-2b-it-8bit` | passed | `recurrent_gemma` | 3.13G | `k=6` 26/26 | 16.3 / 22.5 / 18.1 | 30 | 2,372M | Direct bitsandbytes 8-bit `SCB` -> Q1 (`k=6`), clean Paris / Eiffel / tips |
| 33 | `34_hf_xlstm-german-wikipedia` | failed | — | 0.41G | — | — | — | — | load: `UnsupportedModelError`: non-standard config schema / external `xlstm` code |
| 34 | `35_hf_RWKV4-169m-bin` | failed | — | 0.63G | — | — | — | — | load: legacy pickle `.bin` only (no `.safetensors` / `.gguf`) |
| 35 | `36_hf_OpenELM-270M-Instruct` | failed | — | 0.51G | — | — | — | — | load: `UnsupportedModelError`: non-standard config schema / `trust_remote_code` |
| 36 | `37_hf_gpt2-124m` | passed (base) | `gpt2` | 0.51G | `k=8` 12/12 | 78.1 / 106.3 / 77.1 | 0 | 225M | Base `Conv1D` model (106.3 tok/s at `k=8`); greedy R1/R2 repeat, coherent R3 |
| 37 | `38_hf_bloom-560m` | passed (base) | `bloom` | 1.04G | `k=8` 24/24 | 31.1 / 28.4 / 24.8 | 0 | 629M | Base model (31.1 tok/s at `k=8` — fixed `k=5` gibberish); coherent Paris & R3 |
| 38 | `39_hf_pythia-410m` | passed | `gpt_neox` | 0.85G | `k=8` 24/24 | 42.6 / 42.0 / 34.7 | 0 | 467M | Base `GPT-NeoX` model (42.6 tok/s at `k=8`); coherent Paris & R3 |
| 39 | `40_hf_SmolVLM2-256M` | passed | `llama` | 0.96G | `k=8` 30/30 | 6.5 / 7.9 / 18.5 | 28 | 202M | Text decoder branch (`k=8`), clean Paris / Eiffel / tips |
| 40 | `41_hf_Qwen2-VL-2B-Instruct-AWQ` | passed | `qwen2_vl` | 2.74G | `k=8` 28/28 | 8.7 / 12.9 / 21.2 | 39 | 1,669M | **Newly unlocked** 1D-mRoPE-equivalent `Qwen2` AWQ text backbone, clean 3/3 |
| 41 | `42_hf_Qwen2.5-0.5B-AWQ` | passed | `qwen2` | 0.68G | `k=8` 24/24 | 11.2 / 11.8 / 19.2 | 49 | 555M | AWQ INT4 -> Q1 (`k=8`), clean Paris / Eiffel / tips |
| 42 | `43_hf_Qwen2.5-0.5B-GPTQ-Int4` | passed | `qwen2` | 0.43G | `k=8` 24/24 | 7.4 / 8.5 / 21.1 | 49 | 555M | GPTQ INT4 -> Q1 (`k=8`), clean Paris / Eiffel / tips |
| 43 | `44_hf_internlm2_5-1_8b-chat` | failed | — | 3.52G | — | — | — | — | load: `UnsupportedModelError` (`auto_map` / `trust_remote_code=False`, safe fail) |
| 44 | `45_hf_gpt-oss-20b-MXFP4` | passed | `gpt_oss` | 12.82G | `k=5` 24/24 +MoE | 1.0 / 1.0 / 0.9 | 127 | 3,772M | 20B packed MXFP4 MoE in 3.77 GiB VRAM (`kv2_hit=127`, native Harmony channels) |
| 45 | `Qwen3.6-35B-A3B-uncensored-heretic-APEX-Balanced` | passed | `qwen3_5_moe_text` | 23.86G | `k=5` 40/40 +MoE | 2.2 / 2.4 / 3.3 | 32 | 5,404M | 35B 256-expert MoE in 5.40 GiB VRAM (**3.3 tok/s**, +73% vs 2.0.0), clean 3/3 |
| 46 | `Qwythos-9B-v2-MTP-BF16` | passed | `qwen3_5_text` | 17.14G | `k=5` 32/32 | **21.2** OFF / **33.7** MTP | 62 | 5,861M | 9B BF16 fully resident at `k=5`; **33.70 tok/s** auto-MTP `d=3` (75.7% accept, 256/256 bit-exact) |
| 47 | `Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated.i1-Q4_K_M` | passed | `qwen3_next` | 46.00G | `k=5` 48/48 +MoE | **2.61 / 2.50** | 182 | 6,070M | **Newly verified in 2.2.0** — 80B 24,576-expert MoE in 5.38–5.93 GiB VRAM (4.82 s boot, fixed grouped SSM order) |

## Historical baselines

- **2.0.0 (2026-11-24, 37/47 passed):** [benchmarks/model_support_2_0_0.json](benchmarks/model_support_2_0_0.json), [benchmarks/RELEASE_2_0_0.md](benchmarks/RELEASE_2_0_0.md).
- **1.2.0 (2026-11-18, 31/46 passed):** [benchmarks/model_support_1_2_0.json](benchmarks/model_support_1_2_0.json), [benchmarks/RELEASE_1_2_0.md](benchmarks/RELEASE_1_2_0.md).

## Boundaries

- Checkpoints requiring `trust_remote_code=True` (`auto_map` outside native `transformers`), legacy pickle `.bin` weights without `.safetensors`/`.gguf`, or unmapped GGUF architectures (`gemma3n`) raise `UnsupportedModelError` cleanly without interactive prompts or process exit.
- Multimodal checkpoints (`SmolVLM2`, `Qwen2-VL`) execute the text decoder branch only; vision/audio towers are skipped.
- Ultra-low-bit source quantizations (`IQ1_S`) remain bounded by the information stored in the source weights even at `k=8`.
