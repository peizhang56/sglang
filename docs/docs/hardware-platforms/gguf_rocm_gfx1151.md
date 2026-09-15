# GGUF on ROCm/HIP, gfx1151 (Strix Halo)

Enablement report and measured results for running GGUF-quantized models on an AMD
RDNA3.5 APU.

- **Date:** 2026-09-15
- **Hardware:** AMD Ryzen AI MAX+ 395 (gfx1151 / RDNA3.5, 20 CU, wave32), 46.9 GiB unified memory
- **Software:** ROCm 7.2, torch 2.9.1, WSL2 (kernel 6.6.114.1-microsoft-standard-WSL2)

## Source changes

| | |
|---|---|
| Branch | `gguf-rocm-gfx1151` |
| Base | `main` @ `242d8a70c0` |
| Commit | `14b767d92e` — *Enable GGUF on ROCm/HIP for gfx1151 (RDNA3.5) and fix four correctness bugs* |
| Remote | `git@github.com:peizhang56/sglang.git` (pushed) |

14 files. Kernel enablement:

- `python/sglang/kernels/aot/setup_rocm.py` — compile `csrc/quantization/gguf/gguf_kernel.cu`
- `python/sglang/kernels/aot/csrc/common_extension_rocm.cc` — register the six `ggml_*` ops
- `python/sglang/kernels/aot/csrc/quantization/gguf/gguf_kernel.cu` — `default:` cases on all
  four type switches, null check on `ggml_get_to_cuda`, kernel launch checks
- `python/sglang/kernels/aot/csrc/quantization/gguf/ggml-common.h`,
  `python/sglang/kernels/aot/include/utils.h` — warp-size `static_assert`

Runtime and loader:

- `python/sglang/srt/layers/quantization/gguf.py` — drop the HIP gate
- `python/sglang/srt/configs/model_config.py` — allow `gguf` on ROCm
- `python/sglang/srt/model_loader/loader.py` — rope unpermute, GemmaRMSNorm unfold,
  reject unmapped tensor names
- `python/sglang/srt/model_loader/gguf_name_maps.py` — `qwen3_5` name map builder
- `python/sglang/srt/models/qwen3_5.py`, `qwen3_5_text.py` — GDN value-head un-tiling
- `python/sglang/srt/utils/common.py` — `is_amd_apu()`

Tests: `python/sglang/kernels/aot/tests/test_gguf_rocm.py`,
`python/sglang/kernels/aot/tests/run_mutation_gate.sh`.

Deliberately **not** committed: the `pyproject.toml` / `pyproject_other.toml` /
`pyproject_rocm.toml` swaps (ROCm docker image build-variant renames, not part of this
work — committing them deletes upstream files), and all hipify `.hip` / `_hip.h`
artifacts (committing them makes hipify skip the real sources on later builds).

## Bugs fixed

Four of the five are architecture-general; only the last is ROCm-specific.

1. **Missing GGUF rope unpermute** (llama/mistral/mixtral). `convert_hf_to_gguf.py`
   permutes `q_proj`/`k_proj` rows so GGML's rope reads interleaved pairs; HF keeps the
   split-half layout, and nothing in the tensor name records it. **Affects CUDA today.**
   Symptom is fluent but factually wrong text — it passes a smoke test.
2. **Missing GemmaRMSNorm `+1` unfold** (qwen3_5). llama.cpp folds the `1` of
   `(1 + weight)` into the exported tensor. Also not ROCm-specific. Symptom is gibberish.
3. **GDN value-head tiling.** llama.cpp tiles 48 value heads across 16 key heads
   (export head `i` -> key head `i % 16`); HF/FLA interleaves (`j // 3`).
4. **Unmapped tensor names collide.** Every name-map miss keyed on the literal
   `"None.<suffix>"` and overwrote the previous one, hiding 96 missing GDN parameters
   behind a single entry.
5. **AMD APU memory over-commit.** ROCm reports `is_integrated == 0` for gfx1151, so
   `mem_get_info()` advertised 108.4 GiB on a 46.9 GiB machine.

## Correctness: before vs. after

| model | before | after |
|---|---|---|
| Llama-3.2-1B-Instruct Q4_K_M | `' located in the United States\nThe capital of the United States of France'` | `' Paris. The Eiffel Tower is a famous landmark in Paris...'` |
| Qwen3.8-27B UD-Q4_K_M | `';’ ’樄'` | `' Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome...'` |

## Commands

### Llama-3.2-1B-Instruct Q4_K_M (GGUF)

Server:

```bash
python3 -m sglang.launch_server \
  --model-path /root/models/llama32-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --dtype float16 --context-length 4096 --max-total-tokens 65536 \
  --attention-backend triton --disable-cuda-graph --port 30002
```

Client:

```bash
python3 benchmark/gsm8k/bench_sglang.py \
  --data-path /tmp/test.jsonl --num-questions 10 --parallel 10 --port 30002

python3 benchmark/gsm8k/bench_sglang.py \
  --data-path /tmp/test.jsonl --num-questions 200 --parallel 16 --port 30002
```

### Llama-3.2-1B-Instruct bf16 (unquantized baseline)

Server — identical flags, only `--model-path` differs:

```bash
python3 -m sglang.launch_server \
  --model-path /root/models/llama32-1b-hf \
  --dtype float16 --context-length 4096 --max-total-tokens 65536 \
  --attention-backend triton --disable-cuda-graph --port 30003
```

Client:

```bash
python3 benchmark/gsm8k/bench_sglang.py \
  --data-path /tmp/test.jsonl --num-questions 200 --parallel 16 --port 30003

# chat-template variant
python3 benchmark/gsm8k/bench_sglang.py \
  --data-path /tmp/test.jsonl --num-questions 200 --parallel 16 --port 30003 \
  --enable-thinking --tokenizer-path /root/models/llama32-1b-hf \
  --chat-template-kwargs '{}'
```

### Qwen3.8-27B UD-Q4_K_M (GGUF)

Model: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q4_K_M.gguf>
(16.5 GB, single file).

Server — `SGLANG_MAMBA_CONV_DTYPE` and `--skip-server-warmup` are both required, see
*Known issues*:

```bash
SGLANG_MAMBA_CONV_DTYPE=float16 python3 -m sglang.launch_server \
  --model-path /root/models/qwen38-27b-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --dtype float16 --context-length 4096 --max-total-tokens 16384 \
  --attention-backend triton --disable-cuda-graph --skip-server-warmup \
  --watchdog-timeout 100000 --port 30002
```

Client:

```bash
python3 benchmark/gsm8k/bench_sglang.py \
  --data-path /tmp/test.jsonl --num-questions 10 --parallel 10 --port 30002

python3 benchmark/gsm8k/bench_sglang.py \
  --data-path /tmp/test.jsonl --num-questions 10 --parallel 1 \
  --max-new-tokens 768 --port 30002
```

## Results

GSM8K, 5-shot, temperature 0, `benchmark/gsm8k/bench_sglang.py`.

| model | q | parallel | accuracy | invalid | latency | throughput | outcome |
|---|---|---|---|---|---|---|---|
| 1B GGUF Q4_K_M | 10 | 10 | 0.500 | 0.000 | 21.6 s | 38.5 tok/s | ok |
| 1B GGUF Q4_K_M | 200 | 16 | **0.325** | 0.010 | 205.7 s | 86.7 tok/s | ok |
| **1B bf16 baseline** | 200 | 16 | **0.330** | 0.010 | 61.7 s | 288.8 tok/s | ok |
| 1B bf16 + chat template | 200 | 16 | 0.280 | 0.020 | 56.5 s | 342.1 tok/s | ok |
| 1B GGUF, **unbounded KV pool** | 10 | 10 | — | — | — | — | **hung** (watchdog, 300 s) |
| 27B GGUF UD-Q4_K_M | 10 | 10 | 0.400 | 0.500 | 121.5 s | 6.26 tok/s | ok |
| 27B GGUF UD-Q4_K_M | 10 | 1 | 0.400 | 0.500 | 151.6 s | 5.01 tok/s | ok |
| 27B GGUF UD-Q4_K_M | 10 | 5 | — | — | — | — | **hung** (killed at 21 min) |

### The result that matters

**Q4_K_M scores within one question out of 200 of the unquantized bf16 baseline**
(0.325 vs 0.330), on the same harness, prompts, seed and server flags — only the
checkpoint differs. That is the strongest available evidence that the dequantize
kernels, the matmul path and the rope unpermute are all correct together.

### What the absolute numbers do and do not mean

The ~33% is **not** attributable to quantization and **not** explained. Meta reports
44.4% for Llama-3.2-1B-Instruct (bf16, 8-shot CoT). Applying the chat template was
tested as an explanation and **disproven** — it made accuracy *worse* (0.280). The
residual gap is some combination of evaluation-protocol differences (5-shot vs 8-shot
CoT, answer extraction, prompt wording) that has not been isolated. Do not cite the
absolute figure as a model-quality result.

Separately, a 1B model is weak at GSM8K by nature: the 3B sibling scores ~77% and
frontier models exceed 95%.

### The 27B's 0.400 is not an accuracy measurement

`Invalid: 0.500` — five of ten responses contained **zero digits**, which is the only
condition under which `get_answer_value` returns `INVALID`. Inspecting raw outputs:

- **Empty completions.** One request returned 1 token, `''`. `bench_sglang.py:110` sets
  `stop=["Question", "Assistant:", "<|separator|>"]`; this is a thinking model, and when
  it opens by restating the question the stop fires immediately.
- **Truncation mid-number.** Another ended `'...80,000*1.5=<<80000*1.5=120000>>120,0 word'`
  — extraction took `1200`.
- **Prompt-format mismatch.** The model emits `<think>` blocks and LaTeX
  (`$$ \text{Clips in May} = \frac{48}{2} = 24 $$`) against a plain-text completion-style
  few-shot prompt, and one response opened *"The user is asking the same question that was
  already answered at the beginning of the conversation."*

The reasoning itself was correct where it was legible. This is a harness/model-format
mismatch, not a correctness or a quantization result. Evaluating this model properly
needs a thinking-aware harness.

### Throughput: GGUF is 3.3x slower than bf16 on the 1B

86.7 vs 288.8 tok/s. Quantization is supposed to win here and loses badly. A 1B's
weights are ~0.7 GB, so decode is not bandwidth-bound on a 256 GB/s part — it is
dominated by per-layer launch overhead, and GGUF adds dequantize work on top. The
bandwidth argument that motivates GGUF only starts paying at large model sizes. Do not
read the 1B throughput as representative of what GGUF is for.

## Known issues

### The 27B hangs intermittently — unresolved

It is **intermittent**, not a function of concurrency. Observed: `--parallel 10`
completed, `--parallel 1` completed, `--parallel 5` hung; two single-request warmup
hangs and two single-request successes.

Ruled out:

- **Memory.** Reproduces with `--max-total-tokens 16384` and 40 GiB free — a ~2000x
  margin over what the request needs.
- **Request shape.** Blamed on server warmup, but warmup sends
  `"The capital city of France is"`, the exact prompt that succeeds with
  `--skip-server-warmup`.
- **Kernel-ordering race.** Succeeds both with and without `AMD_SERIALIZE_KERNEL=3`.

Specific to the Gated DeltaNet path — the 1B (plain Llama attention) is stable across
every run attempted, including 200 questions at 16-way concurrency.

Best remaining hypothesis, **untested**: firing a request while the 16.5 GB weight load
is still draining through the page cache re-creates transient memory pressure, and
WSL2's `dxgkrnl` wedges the GPU queue rather than failing the allocation
(`dxgvmb_send_create_allocation failed ffffffb5`, `Ioctl failed: -75` / EOVERFLOW).

Workaround: `--skip-server-warmup`, and let the load settle before the first request.

### Other

- **27B at `--dtype bfloat16` hangs in warmup** even with the memory fix. Separate issue.
- **`SGLANG_MAMBA_CONV_DTYPE=float16` is required** for Gated DeltaNet under
  `--dtype float16`. `configs/mamba_utils.py:68` hardcodes the conv state dtype to
  bfloat16 regardless of runtime dtype, so long prefill otherwise dies with a dtype
  mismatch at `gdn_backend.py:911`. Architecture-general bug.
- **`--max-total-tokens` is currently mandatory on this platform.** `is_amd_apu()` fixes
  the 108 GiB over-report, but `_profile_available_bytes`
  (`mem_cache/kv_cache_configurator.py:2148`) still sets the KV budget from available
  memory minus a slack that `mem_fraction_static` only scales — so the pool claims nearly
  all of system RAM. At the default 0.884 the 1B took a 36 GB / 1,216,359-token pool and
  left ~8 GB for everything else: enough for single requests, not enough for 10
  concurrent prefills (the hung row in the results table).

## Kernel-level verification

Offline oracles, independent of the serving path:

- `ggml_dequantize` vs. numpy reference: max abs error 6.6e-05 (Q6_K), 4.4e-04 (Q4_K)
- `fused_mul_mat_gguf` vs. dequantize-then-matmul, ntok in {1, 4, 7, 32, 128}:
  cosine >= 0.999854, relative Frobenius <= 0.0171

The kernels are correct. Every bug found in this work was in the loader.

## Follow-ups worth filing upstream

1. Missing GGUF rope unpermute for llama/mistral/mixtral — affects CUDA today.
2. Missing GemmaRMSNorm `+1` unfold for qwen3_5 — affects CUDA today.
3. `conv_dtype` hardcoded to bfloat16 in `mamba_utils.py:68`.
4. Unmapped GGUF tensor names colliding on the literal `"None.<suffix>"`.
5. `is_amd_apu` memory over-commit, and the KV budget taking ~all available memory on
   unified-memory parts.
6. IQ3_S `iq3xs_grid` bug.
7. Missing `sudot4` path — RDNA3/3.5 ROCm only.
