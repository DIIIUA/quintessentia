# Changelog

## 2.2.0 — 2026-11-27

### Bilingual English & Russian CLI, Interactive TUI & Web UI
- **Button-driven zero-dependency Interactive TUI (`quintessentia_cli/tui.py` — `quint`, `quint menu` / `ui` / `tui`):**
  - Cross-platform raw single-key navigation (`msvcrt` + VT100 on Windows; unbuffered POSIX `os.read` PTY escape-sequence parser on Linux/WSL 2 fixing ConPTY split-arrow sequences).
  - Live searchable local model picker, quick presets (`auto`, `auto_kl`, `quality`, `balanced`, `speed`, `bf16`, `custom`), full Q1 controls (`--q1-planes`, `--cube-root-bits`, `--lm-head-planes`, `--min/max-q1-planes`, `--q1-kl-target`, `--no-auto-fit-q1`), persistent `cli_config.json`, and one-key launch of Chat, LAN Server, Benchmark, Doctor, and Hugging Face Download.
- **Full English & Russian (`EN / RU`) localization (`quintessentia_cli/i18n.py`, `cli.py`, `tui.py`, `lan_server.py`):**
  - Global `--lang en|ru` flag (before or after any subcommand, including `--help`), `QUINTESSENTIA_LANG=en|ru`, saved `cli_config.json` preference, and automatic OS locale detection.
  - Instant one-keypress **`[L]`** toggle and top-row **`Language / Язык`** selector in the TUI, `/lang en|ru` slash command in `quint chat`, localized LAN server console hotkeys/messages, and an **`EN / RU`** switcher button in the built-in LAN Web UI (`localStorage` + `/api/info` `"language"`).

### Resident Q1 Native MTP Acceleration (up to 1.91× speedup on 9B BF16)
- **Native GGUF NextN/MTP extraction & GPU-resident MTP head (`quintessentia_cli/model_source.py`, `src/native_mtp.py`):**
  - Extracts `blk.<layers>.nextn.*` weights (`eh_proj`, `enorm`, `hnorm`, `shared_head_norm`, and `mtp.layers.0.*`) directly from GGUF checkpoints (`Qwythos-9B-v2-MTP-BF16` `blk.32.*`) with zero-centered RMSNorm handling and raised 512 MiB default MTP head budget.
- **Two-stage zero-SMEM SWAR `__dp4a` multi-row (`M=2..4`) Q1 CUDA kernels (`src/quintessentia_q1_cube_nvrtc.py`):**
  - Splits Phase-1 FWHT + INT8 quantization (`ideal_q1_fwht_quant_i8_f32`) from zero-SMEM L1 `__ldg` register-only SWAR `__dp4a` multi-row GEMV (`ideal_q1_popc10_mr_gemv_f32`) with a zero-alloc `ctypes` argument cache, while keeping `M=1` on bank-conflict-free 10-bit `popc10_r2`.
- **Fused multi-row Q1 DeltaNet & zero-replay speculative rollback (`quintessentia_cli/unified_runtime.py`, `src/general_speculation.py`, `quintessentia_cli/thought_buffer.py`):**
  - Added `fused_deltanet_q1_multirow`, `install_q1_qwen35_fast_decode_unifications`, `HybridSpecSnapshotManager` zero-replay slots, fast direct-clone linear-attention cache snapshot/restore, raised default `--max-drafts 3`, and enabled automatic native MTP on resident `q1_cubes` (`doctor` reports `MTP head: weights present, on`).
  - Verified on RTX 3060 Ti 8 GB (`Qwythos-9B-v2-MTP-BF16`, `k=5`, 256 tokens greedy): **`21.17 tok/s`** (MTP off) → **`30.82 tok/s`** (`d=2`) → **`33.70 tok/s`** (auto-MTP `d=3`, `75.7%` accept, `0` replays, bit-exact `256/256` match — **1.91×** vs 2.1.0 `17.61 tok/s`).

### 80B-A3B `qwen3_next` GGUF Fix & Verification on 8 GB VRAM
- **Fixed `qwen3_next` GGUF SSM head ordering (`quintessentia_cli/gguf_arch_registry.py`):**
  - Removed erroneous `TiledToGroupedRows` / `TiledToGroupedInputs` transforms on `qwen3_next` `A_log`, `dt_bias`, `conv1d.weight`, and `out_proj.weight` (unlike `qwen35`, `qwen3next` GGUFs preserve Hugging Face grouped `[16, 2]` head order across all 6 SSM tensors).
  - Verified live on **`Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated.i1-Q4_K_M`** (`46 GiB`, 48 layers, 512 experts/layer = `24,576` experts) on an 8 GB RTX 3060 Ti: boots in **`4.82 s`** with all 48 trunk layers resident and generates coherently in **`5.38–5.93 GiB` VRAM** (**`2.61 / 2.50 tok/s`**, `kv2_hit=182`, `8.96–14.53%` experts carved on demand, `0` OOM).

### Disk-Backed Token Ledger, Live 0-ms Q1 Controls & Resizable Live HUD
- **Two-layer persistent Token Ledger & context compaction (`quintessentia_cli/token_ledger.py`, `unified_runtime.py`, `tool_calling.py`, `lan_server.py`):**
  - Append-only `ledger.jsonl` archive + atomic `active_window.json` slice bound by tokenizer SHA-256 fingerprint for deterministic 100% retokenization across restarts, `.strip()` whitespace differences, `<think>` stripping, and mid-chat edits (`partial_turn_prefix` + pure-attention KV truncation).
  - Non-destructive epoch compaction via `/compact <summary>` in `quint chat` and `POST /api/compact`, incremental turn suffix fallback for base checkpoints without `chat_template`, and preserving explicit `<tool_call>` blocks even on unknown tool names.
- **Live 0-ms Q1 precision switching, KL calibration & `--rssf on|off` ablation (`unified_runtime.py`, `lan_server.py`, `src/rssf_dynamic_operators.py`):**
  - Switch active `q1_planes` on the fly in 0 ms via `/k <1..loaded>`, LAN console hotkeys `1..9`, Web UI buttons, or `POST /api/q1`; auto-calibrate smallest `k` within a KL budget via `--q1-kl-target` (`calibrate_q1_planes_by_kl`) without modifying `embed_tokens`.
  - Added clean process-level `--rssf on|off` (`QUINTESSENTIA_RSSF`) ablation switch across all active RSSF/Volterra/FDS scales (`M0–M8` + `KV`).
- **Resizable Desktop Live HUD & console lifecycle binding (`quintessentia_cli/hud/`, `live_hud_beacon.py`, `cli.py`):**
  - Uniform vector-scaled resizable WPF HUD (`CanResize`, `MinWidth=440`, `MinHeight=414`, persisted geometry), clean Windows-native host RAM & branding without misleading WSL2 labels, expanded 8,000-char answer / 4,000-char thinking scrollable preview, and automatic `console_owner_alive.flag` heartbeat so HUD closes cleanly when its parent CLI/TUI console exits.
  - Added interactive LAN server console hotkeys (`[M/Q/Esc]` stop & return to menu, `[O]` open Web UI, `[H]` Desktop HUD, `[1..9]` live Q1 `K`) and raised default `--max-request-bytes` to 16 MiB.

### Verified test results (RTX 3060 Ti 8 GB, WSL 2 ext4 & Windows Native)
- **Windows Native `.venv` (CUDA + CPU):** **84 passed, 1 skipped** (POSIX PTY skip) across all 10 core CLI/TUI/i18n/Token Ledger/RSSF/Tool Calling/LAN/VRAM/v21-22 suites (`0` failed).
- **WSL 2 (CUDA + CPU):** **223 / 223** passing across the 19 core release suites (`test_tui_and_q1_controls`, `test_token_ledger`, `test_v21_fixes`, `test_cli_lan`, `test_rssf_dynamic_operators`, `test_tool_calling`, `test_lan_server_cancel`, `test_vram_plan`, `test_gguf_tokenizer_types`, `test_context_budget`, `test_binary_halving_q1`, `test_mirror_scales_v4`, `test_q1_cube_engine`, `test_general_speculation`, `test_speculation_joint`, `test_thought_buffer`, `test_session_state`, `test_download_preparation`, `test_model_management`); full repository census `460 passed, 10 skipped` (`0` regressions from 2.2.0).
- **Static checks:** `120 / 120` Python files compile cleanly (`py_compile`, `0` errors); PowerShell AST syntax check of `quintessentia_hud.ps1` passed (`0` errors).

## 2.1.0 — 2026-11-26

### Unified 4-layer solver-free multi-scale balance (M0–M8 + KV + Prefill/Turn)
- **Complete 4-layer balance operators (`src/rssf_dynamic_operators.py`):**
  - Layer 1: O(1) solver-free 8-pair force balance ($\Delta_3$-margin, scale-invariant across MB and GB regimes).
  - Layer 2: Energy-conserving Volterra archive (`VolterraFDSMonitor`) tracking memory integral and drift.
  - Layer 3: Fluctuation-Dissipation balance ($\dot{E} - \lambda\dot{S}$) distinguishing calm expansion from genuine drift/overload.
  - Layer 4: Explicit `StrategyCollapse` flags (`loss_of_archive`, `scale_break`, `regime_overload`, `non_conservative_drift`).
- **Wired across every runtime scale with strict V4 isolation preserved:**
  - `M0/M5b` (`src/binary_halving_q1.py`, `src/mirror_scales_v4.py`), `M1` load balancing, `M2` region pool breathing, `M3` cold/post-prefill warmup latency drift, `M4` depth coverage/scar balance, `M6` probability-path survivors (`src/survivor_probability_path.py`), `M7` speculative depth gating, `M8` zero-sync tensor thought-buffer balance (`quintessentia_cli/thought_buffer.py`), hybrid `KV` VRAM/pinned-RAM placement (`quintessentia_cli/kv_offload.py`), and `unified_runtime.py` prefill chunk damping & turn lifecycle telemetry.
- **Bidirectional mirror-balance Q1 depth auto-fit (`auto_fit_q1_planes`):**
  - Retains whole-plane step-down (`5 -> 4 -> 3`) when large weights exceed VRAM, and adds automatic step-up refinement (`5 -> 6..8`) when a compact dense checkpoint fits inside `<= 35%` of available VRAM budget (leaving `>= 65%` headroom for KV cache), eliminating low-bit degradation on outlier-heavy small models (`BLOOM-560m` improved from gibberish at `k=5` to `31.1 tok/s` clean English at `k=8`).

### Universal architecture expansion & bug fixes (39 / 45 present checkpoints passing 3x suite)
- **4 newly unlocked checkpoints & formats:**
  - **`MiniCPM3-4B-MLA` (`minicpm3` GGUF):** native MLA config mapping (`q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`), accurate compressed MLA `kv_bytes_per_token` planning (fitting **62/62 layers resident** in 2.94 GiB VRAM), and asymmetric `dk != dv` support across `PreallocatedGPUDynamicLayer`, `PinnedOffloadedDynamicLayer`, and `_SharedGPUStagingPool` (`quintessentia_cli/kv_offload.py`).
  - **`BitNet-b1.58-2B` (`TQ2_0` and `I2_S`):** added `bitnet` and `bitnet-b1.58` GGUF architecture mappings (`attn_sub_norm`, `ffn_sub_norm`, `LlamaRopeReversePermute`) plus native **`GGML_TYPE_I2_S` (type 36)** 128-element interleaved ternary decoder with per-tensor trailing float32 scale (`quintessentia_cli/gguf_reader.py`, `quintessentia_cli/model_source.py`).
  - **`Qwen2-VL-2B-Instruct-AWQ`:** unwraps `qwen2_vl` / `qwen2_5_vl` pure-text 1D-mRoPE-equivalent backbone directly onto native `Qwen2` CausalLM (`21.2 tok/s`, `kv2_hit=39`, 1.67 GiB VRAM).
- **Universal GGUF & runtime bug fixes:**
  - **Interleaved RoPE reverse-permutation (`quintessentia_cli/gguf_arch_registry.py`):** extended `LlamaRopeReversePermute` to `granite`, `granitemoe`, `granitehybrid`, `smollm3`, `bitnet`, and `bitnet-b1.58`, and registered `"refact"` pre-tokenizer regex — fixing garbled `SmolLM3-3B` and stuttering `Granite-3.3-2B` into clean factual and structured outputs.
  - **Universal chat-template EOT stop tokens (`quintessentia_cli/unified_runtime.py`, `src/general_speculation.py`):** `_resolve_effective_eos_ids` automatically stops on chat-template turn terminators (`<|end|>`, `<end_of_turn>`, `<|END_RESPONSE|>`, `<|eot_id|>`, `<|im_end|>`, `<|return|>`) while protecting Harmony multi-channel `<|end|>` separators (`GPT-OSS-20B`). Eliminates leaked stop tags in `Phi-4-mini`, `Gemma-2`, and `Command-R7B`.
  - **Hybrid SSM/conv session unpark (`quintessentia_cli/unified_runtime.py`):** safely initializes `None` slots and restores `conv_kernel_size` in `QuintessentiaKVPrefixSession.unpark_to_gpu`, fixing multi-turn Run 2 crashes on `Zamba2-1.2B` (`kv2_hit=31`).
  - **Safe rejection of custom `auto_map` checkpoints (`quintessentia_cli/model_source.py`):** passes `trust_remote_code=False` and rejects unregistered `auto_map` architectures (`rwkv7`, `internlm2`) with `UnsupportedModelError` instead of triggering an interactive EOF `sys.exit(0)`.

### Desktop Live HUD Widget (Windows Native & WSL 2) & CLI upgrades
- **Bundled `Quintessentia Live HUD` (`quintessentia_cli/hud/` & `quintessentia_cli/live_hud_beacon.py`):**
  - Zero-dependency NVML + Win32 `GlobalMemoryStatusEx` + Win32 `GetSystemTimes` + `vmmemWSL` telemetry backend supporting both **Windows Native** (including CPU-only safe mode) and **WSL 2**, with PCIe wrap-glitch and micro-burst `peak_tok_s` filtering and automatic peak reset on model switch.
  - Polished dark WPF widget (`660×580`) with non-clipping KPI cards, dark thin scrollbar, `×` close button, right-click menu, truthful `WIN native` vs `WSL2` badge, `HOST CPU LOAD` bar when WSL 2 is idle, live **`BALANCE + KV`** strip (`d3` margin, `KV` GPU/RAM layers & promotions, active scales `M0..M8`, archive/scale-break alert badges), and instant stop via `stop_hud.flag` + `hud_ui.pid`.
- **CLI & Web UI upgrades (`quintessentia_cli/cli.py`, `quint.cmd`, `quintessentia_cli/lan_server.py`):**
  - Fixed `NameError: os is not defined` in `launch_desktop_hud` and added automatic local staging (`%USERPROFILE%\.quintessentia\hud`) when launching from WSL 2.
  - Added `quint hud --stop | --status | --install-desktop`, `quint bench --runs N`, piped `stdin` support for `quint ask`, `/balance` and `/hud` slash commands in `quint chat`, UTF-8 `-X utf8 -u` in `quint.cmd`, and live `KV` / `Balance` telemetry badges in the LAN Web UI and console summary.

### Verified hardware & test results (RTX 3060 Ti 8 GB, WSL 2 ext4 & Windows Native)
- **39 / 45** present checkpoints passed isolated **3-turn** execution (`R1` cold factual, `R2` KV/SSM continuation, `R3` post-reset structured) with **0 OOM, 0 timeouts, and 0 crashes**: `Qwythos-9B-BF16` **18.9 tok/s** (`5.86 GiB`, `kv2_hit=62`; `60K` decode `13.06 tok/s`, `80K` `10.17 tok/s`), `Qwen3.6-35B-A3B` **3.3 tok/s** (+73% vs 2.0.0, `5.40 GiB`), `Command-R-7B` **19.8 tok/s**, `MiniCPM3-4B-MLA` **6.7 tok/s** (`62/62` resident), `BitNet-b1.58-2B` **12.3–17.0 tok/s**, `Qwen2-VL-2B-AWQ` **21.2 tok/s**, `Zamba2-1.2B` **17.1 tok/s**, `LFM2-350M` **48.2 tok/s**, `GPT-2` **106.3 tok/s**. See [MODEL_SUPPORT.md](MODEL_SUPPORT.md) and [benchmarks/model_support_2_1_0.json](benchmarks/model_support_2_1_0.json).
- **194 / 194** `pytest` passing across 17 CUDA/CPU suites in WSL 2 plus Windows native regression verification (`test_v21_fixes`, `test_vram_plan`, `test_cli_lan`, `test_rssf_dynamic_operators`, `test_tool_calling`, `test_lan_server_cancel`, `test_gguf_tokenizer_types`, `test_context_budget`).

## 2.0.0 — 2026-11-24

### Pure Q1 Cubes V4 as default CUDA runtime
- **Default `--weight-format auto` promoted to pure `Q1` cubes V4 (`q1_cubes`, `k=5`, `root_bits=8`) on CUDA:** covers dense models, hybrid SSM/Attention models, and small/large MoE checkpoints automatically. `--feedback-mode off`, explicit `--weight-format q1|q2|q4|q6|q8|bf16`, and `--device cpu` opt-outs remain fully intact.
- **High-throughput self-contained NVRTC Q1 CUDA kernels:**
  - Bank-conflict-free `POPC10` 2-row-per-warp GEMV (`Kernel #6-R2`, batched MoE `Kernel #6c`, and `ideal_q1_popc10_r2`) plus cached native C-launcher dispatch (`_fast_k6`).
  - Single-pass CUDA Ballot Q1 carving (`Kernel #7`, >200× faster than Python-loop carving), GPU Sinkhorn scaling without large temporary FP32 buffers, and chunked pure-Q1 `lm_head` carving for large vocabularies (saving >1.1 GiB VRAM on 248k-vocab models).
  - Adaptive V4 GPU/host/disk streaming breather (`M1` load dispatcher + `M2` CUDA-event region pool + prefix-pinned host cache + `ext4` disk L3 cache with Linux `O_DIRECT` / `POSIX_FADV_DONTNEED`) replacing the rigid 3-slot conveyor ring in V4.
- **Complete V4 subsystem stack (M0–M8) & CLI ergonomics:**
  - Isolated V4 runtime components covering trunk carving, load balancing, region pooling, cold/post-prefill warmup, adaptive depth, probability-path tracking, speculative verification, and thought-buffer retention.
  - One-button launchers (`./quint` on Linux/WSL 2, `quint.cmd` / `quintessentia.cmd` on Windows), `--mode default|custom` generation profiles with GPU anti-loop sampling, and updated `doctor`, `chat`, `prepare`, `download`, and LAN server passport/UI reporting active backend/format and `--q1-planes 1..16`.

### Universal architecture & format expansion (37 / 47 checkpoints verified)
- **Hugging Face loaders (`quintessentia_cli/model_source.py`, `hf_quantized_source.py`, `src/binary_halving_q1.py`):**
  - Native `Conv1D` Q1 support (`GPT-2`: 74.31 tok/s, +30% over BF16).
  - Universal `resolve_decoder_layers` and module-level `_tied_weights_keys` sharing (`Mamba`, `Mamba2`, `Zamba2`).
  - Native packed HF **MXFP4** MoE expert streaming (`gpt-oss-20b-MXFP4` running in 3.8 GiB VRAM).
  - Direct HF **bitsandbytes 8-bit (`SCB`)** decoding without requiring `bitsandbytes` (`RecurrentGemma-2B-8bit`).
- **GGUF hybrid & SSM support (`quintessentia_cli/gguf_arch_registry.py`, `model_source.py`):**
  - Added SSM config mapping, trailing singleton squeeze for 2D `ssm_a`/`ssm_d`, `head_dim` resolution, and virtual `#cat0:` merging of split shared-MLP `gate`/`up` tensors (`Falcon-H1-0.5B`, `Granite-4.0-tiny-hybrid`).
- **`transformers 5.x` recurrent cache contract (`src/quintessentia_integrated.py`):**
  - Added native `cache` parameter handling and `_setup_cache` initialization for architectures whose forward does not return `past_key_values` (`RecurrentGemma`), plus `model.final_norm.weight` / `model.norm_f.weight` norm resolution.
- **Linux / WSL 2 NVRTC compatibility:** automatic discovery of PyTorch-bundled `libnvrtc.so*` and `libnvrtc-builtins.so*` on Linux alongside Windows CUDA support.

### Verified hardware & test results (RTX 3060 Ti 8 GB, WSL 2 ext4)
- **37 / 47** checkpoints passing end-to-end 2-turn chat on pure Q1 cubes V4 across **26 `model_type` families**: `Qwythos-9B-BF16` **19.91 tok/s** (32/32 resident, 6.0 GiB VRAM), `Command-R-7B` **15.68 tok/s**, `Llama-3.2-1B` **39.79 tok/s**, `GPT-2` **74.31 tok/s**, `Zamba2-1.2B` **14.22 tok/s**, `RecurrentGemma-2B-8bit` **15.89 tok/s**, `Qwen3.6-35B-A3B` **1.91 tok/s** (5.5 GiB VRAM), `gpt-oss-20b-MXFP4` **0.85 tok/s** (3.8 GiB VRAM). See [MODEL_SUPPORT.md](MODEL_SUPPORT.md).
- **99 / 99** `pytest` passing across core V4, Q1, CLI, download/preparation, quantized-source, and VRAM-plan suites in WSL 2.

## 1.2.0 — 2026-11-18

- Add a numbered local model catalog, selection by number/name/path, Hugging Face
  page/file links, managed downloads into a chosen directory and `prepare`.
  Downloads verify a finite first forward and automatic runtime packing by default;
  complete local snapshots can be reused offline. Split GGUF selection is explicit.
- Keep help/catalog independent of Torch initialization. Register external model
  directories, isolate revisions and retain partial downloads for retry.
- Use the lower CUDA/NVML startup free-memory reading for the GPU matched by UUID;
  expose counter/fallback diagnostics in status and doctor. Live two-process testing
  caught and corrected a Windows accounting discrepancy. This is startup sampling,
  not continuous global memory rebalancing.
- Add a fresh local roster census and a controlled resident/half/full-streaming
  comparison. Llama1B Q6 medians: 30.55 / 11.98 / 7.34 tok/s, warm RAM cache,
  MTP off; all 24 output hashes matched across two pressure settings. See
  `MODEL_SUPPORT.md` and `benchmarks/RELEASE_1_2_0.md` for results and limits.

### Common CPU/GPU execution

- Keep the streamed CPU graph running without CUDA or when GPU preparation fails;
  retire partially adopted GPU layers at a forward boundary without losing live cache.
- Add `--device cpu` / `device="cpu"` and CPU-aware doctor/session parking. Keep GPU
  preparation diagnostics visible; do not retry failed preparation on every token.
- Defer the unused CPU-start staging arena; retain its geometry until GPU transfers
  need real buffers, rather than reserving three complete layer-sized RAM slots.
- Release CPU and GPU preparation together; ready devices take work without a forced CPU-first dependency. GPU transfers use a bounded background queue.
- Adopt ready dense layers and their live cache on the same graph, then hand off to
  the existing GPU ring without replaying the prompt. Keep `--no-progressive-start` as a GPU-first reference.
- Carry the original CPU prefill map and probability-anchor values through GPU adoption;
  protect auxiliary probes against a device change and drain preparation on cancellation.
- Add common streamed draft/verify/commit: prompt lookup and an optional bounded native
  Qwen3.5 text MTP head sharing base embeddings/output projection. Verify token rows
  with decode arithmetic under one layer lease; roll back rejected KV/recurrent rows.
- Preserve retained-map preparation during verification; commit only accepted routes.
  Add execution-scoped Granite/OLMoE zero-route exclusions before reads and current-work
  preparation cuts. Probability forecasts cannot suppress nonzero required routes.
- Add `--speculation`, `--mtp-head`, `--mtp-head-budget-mib` and `--[no-]region-pruning`.
  General large-model, arbitrary dense pruning and universal native-MTP coverage remain unverified.

### Memory and probability experiments

- Bound thought-anchor log-odds influence, retain FP32 logits and the complete original probability distribution, and allow explicit input corrections. Lexical alignment remains diagnostic; it no longer verifies notes or triggers stronger-bias retries. Direct recurrent-state mixing is opt-in.
- Reduce dynamic-memory pinned staging to two expert slots, reuse views/layouts, preserve recently used experts during dense-page reclamation, share observed router rows with dispatch, and wake the producer only for missing preparation work. Dynamic placement remains experimental; a general speedup is not established.
- Use bounded positioned reads for HF globals and layer metadata, avoiding whole-shard mappings and the reproduced Windows error 1455. Global staging and physical read sizes obey the configured cap.
- Add a resumable current-model census and paired thought-buffer acceptance probes. See [development evidence](benchmarks/DEVELOPMENT_20261118.md) for scope and results; probability retention is not a semantic-memory certification.

## 1.1.0 — 2026-11-18

- Universal Q4/Q6 packed projections and compatible SwiGLU expert kernels; Q6 is the generic default. Source GGUF and runtime affine formats remain distinct and lossy.
- Bounded adapters for HF FP8 block, AWQ GEMM INT4 and GPTQ INT4; repaired SentencePiece/ByteLevel tokenizer selection and conv/recurrent/sliding session parking.
- Request context admission now checks prompt plus requested output against the planned and declared model limits before prefill or session unpark. LAN clients receive a context-length error instead of entering an oversized generation.
- Experimental shared dynamic weight arena: A owns active work/cache, B prepares a bounded next region, C reserves no payload; dense and expert pages share CUDA reader fences and transfer ownership without extra copies.
- Retained prefill-map prediction now combines normalized initial and recent activity (25%/75%). Actual routing always determines required expert computation.
- Experimental thought-buffer probes now restore complete sliding-window and conv/recurrent state. Unknown state containers fail explicitly; recurrent telemetry no longer counts nonexistent recurrent states. Original probability fields remain preserved. Lexical alignment is not a correctness verifier and reasoning improvements remain unproven.
- Added reproducible small-model benchmarking, a dated 48-checkpoint support matrix, and regression coverage for the combined memory/probability mechanisms. Not all architectures or checkpoints are supported; see `MODEL_SUPPORT.md` and `benchmarks/RELEASE_1_1_0.md`.

Changes accumulated since 1.0.0 (architecture adapters below are implementation coverage, not blanket checkpoint certification):

- Broadened Hugging Face CausalLM and VLM checkpoint support (`quintessentia_cli/model_source.py`, `src/quintessentia_integrated.py`, `src/bounded_memory_manager.py`, `src/gpu_ring_pipeline.py`): unwraps `text_config` from multimodal VLM checkpoints while ignoring vision/audio towers, supports single custom-named `.safetensors` files and all decoder root/norm/head conventions (`model.layers`, `transformer.h`, `gpt_neox.layers`, `model.decoder.layers`, `backbone.layers`, `rwkv.blocks`, `transformer.blocks`), handles Hub leaf renamings (`Mixtral`, `PhiMoE`, `Nemotron-H`, `Ernie-4.5-MoE`), streams 2D per-expert and 3D packed standard/transposed/biased MoE experts (`Qwen3/3.5-MoE`, `Mixtral`, `DeepSeek-V2/V3`, `Llama-4`, `GPT-OSS`, `OLMoE`, `GLM-4-MoE`, `MiniMax-M2`) via positioned reads + VRAM LRU cache, supports variable-shape same-name layer views (`Gemma 4` sliding vs global attention), co-existing parallel mixers (`Falcon-H1`), scaled input embeddings (`Gemma 1–4`), SSM `cache_params` / `RWKV` `state`, and base-model chat-template fallback while keeping `Qwen3.5-9B` on its fast Q4 + MTP path as default.
- Broadened zero-`mmap` GGUF support (`quintessentia_cli/gguf_arch_registry.py`, `quintessentia_cli/model_source.py`) from 3 architectures to 35+ model families (47+ GGUF architecture IDs), added multi-shard split GGUF (`*-00001-of-0000N.gguf`) discovery while ignoring `mmproj*.gguf`, added `Llama`/`DeepSeek` interleaved RoPE Q/K un-permutation, `GPT-2` `Conv1D` 2D transpose, `BLOOM` QKV un-interleaving, `Phi-2` fused QKV virtual row slicing, `Q8_1` and ternary `TQ1_0`/`TQ2_0` block dequantization, and extended pure-GGUF tokenizer building without `gguf` PyPI or sidecar files.
- Multi-turn KV and GatedDeltaNet session prefix reuse (`quintessentia chat` and LAN web chat) now works across BF16 Hugging Face and GGUF checkpoints as well as Q4 Qwen3.5.
- Self-contained NVRTC CUDA kernels (`src/quintessentia_deltanet_nvrtc.py`) accelerate `causal_conv1d_fn`, `causal_conv1d_update`, `chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule`, and gated/zero-centered `RMSNorm` on Windows and Linux without `causal-conv1d`, `flash-linear-attention`, or Triton, eliminating `transformers` PyTorch reference fallback warnings and multi-kernel `solve_triangular` overhead.
- Coalesced positioned disk reads and batched H2D token/expert routing transfers accelerate MoE multi-token prefill across GGUF and sharded Hugging Face checkpoints.
- LAN server exposes context-window metadata on `GET /v1/models`, `GET /api/v0/models`, `GET /props`, and `GET /api/extra/true_max_context_length` for OpenAI, LM Studio, llama.cpp, and KoboldCPP clients, and logs SSE client disconnects accurately as `cancelled`.
- Cooperative cancellation (`POST /v1/chat/stop`, client disconnect, preemption) now polls at every streamed decoder layer boundary and resets in-flight ring slots cleanly.

## 1.0.0 — 2026-11-14

- Public release: the repository now contains only the program, the CLI and its tests.
- Reads standard Hugging Face shards directly; recommended model `Qwen/Qwen3.5-9B`.
- VRAM modes `normal` and `turbo` replace the fixed `4gb`/`8gb` profiles.
- Fixed CUDA out-of-memory on long prompts: chunked prefill, last-position logits only.
- LAN server: immediate stop (`POST /v1/chat/stop`, closed connections), one generation at a time,
  guaranteed VRAM cleanup, request size limit, rotating log `logs/server.log`.
