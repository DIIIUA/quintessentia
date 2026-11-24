# Changelog

## Unreleased (2.0.0-dev) — 2026-11-24

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
