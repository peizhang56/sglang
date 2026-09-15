# Qwen3.8-27B GGUF on ROCm/HIP, gfx1151 (Strix Halo)

Running `unsloth/Qwen3.8-27B-GGUF` (UD-Q4_K_M, 16.5 GB) on an AMD Ryzen AI MAX+ 395
(gfx1151 / RDNA3.5, wave32, 46.9 GiB unified memory), ROCm 7.2, torch 2.9.1, WSL2.

Model: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q4_K_M.gguf>

Q4_K_M is the only quantization win available here: FP8 is out because `torch._scaled_mm`
rejects gfx1151 (RDNA3.5 has no FP8 matrix instructions), and bf16 would read ~54 GB per
token on a 46.9 GB machine.

## Status

Loads, generates correct text, completes a GSM8K run. **Not production-ready:** it hangs
intermittently, and the standard GSM8K harness cannot score it (50% extraction failure).

## Source changes

Branch `gguf-rocm-gfx1151`, base `main` @ `242d8a70c0`, pushed to
`git@github.com:peizhang56/sglang.git`.

- `14b767d92e` — enable the GGUF kernels on HIP and fix four correctness bugs
- `6bfc6d47b7` — this report

GGUF was hard-gated off on HIP; the ROCm code paths existed but were never compiled
(`setup_rocm.py`, `common_extension_rocm.cc`, plus `default:` cases, a null check and
launch checks in `gguf_kernel.cu`). Built for gfx1151/wave32 only.

Four loader bugs stood between a working kernel and correct output — before this work the
model emitted `';’ ’樄'`:

1. **GemmaRMSNorm `+1` folding.** llama.cpp folds the `1` of `(1 + weight)` into the
   exported tensor. Symptom: gibberish. **Affects CUDA equally.**
2. **GDN value-head tiling.** llama.cpp tiles 48 value heads across 16 key heads
   (`i % 16`); HF interleaves (`j // 3`). Also `ssm_a` stores `-exp(A_log)`.
3. **Unmapped tensor names collided** on the literal `"None.<suffix>"`, hiding 96 missing
   Gated DeltaNet parameters silently.
4. **AMD APU memory over-commit.** ROCm reports `is_integrated == 0` for gfx1151, so
   `mem_get_info()` advertised 108.4 GiB on a 46.9 GiB machine.

Files: `loader.py`, `gguf_name_maps.py`, `qwen3_5.py`, `qwen3_5_text.py`,
`quantization/gguf.py`, `model_config.py`, `utils/common.py`, plus the kernel build.

## Commands

All three of `SGLANG_MAMBA_CONV_DTYPE`, `--max-total-tokens` and `--skip-server-warmup`
are currently required — see *Known issues*. Load takes ~4 min.

```bash
SGLANG_MAMBA_CONV_DTYPE=float16 python3 -m sglang.launch_server \
  --model-path /root/models/qwen38-27b-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --dtype float16 --context-length 4096 --max-total-tokens 16384 \
  --attention-backend triton --disable-cuda-graph --skip-server-warmup \
  --watchdog-timeout 100000 --port 30002
```

```bash
python3 benchmark/gsm8k/bench_sglang.py \
  --data-path /tmp/test.jsonl --num-questions 10 --parallel 10 --port 30002
```

## Results

GSM8K, 5-shot, temperature 0.

| questions | parallel | accuracy | invalid | latency | throughput | outcome |
|---|---|---|---|---|---|---|
| 10 | 10 | 0.400 | 0.500 | 121.5 s | 6.26 tok/s | ok |
| 10 | 1 | 0.400 | 0.500 | 151.6 s | 5.01 tok/s | ok |
| 10 | 5 | — | — | — | — | **hung**, killed at 21 min |

Single request: 150 tokens in 26.7 s = **5.6 tok/s**.

Output is correct where legible — on the Natalia clips question it produces
`$$ \text{Clips in May} = \frac{48}{2} = 24 $$` and the right total.

### The 0.400 is not an accuracy measurement

`Invalid: 0.500` — five of ten responses contained **zero digits**, the only condition
under which `get_answer_value` returns `INVALID`. From the raw outputs:

- **Empty completions.** `bench_sglang.py:110` sets `stop=["Question", ...]`; this is a
  thinking model, and when it opens by restating the question the stop fires immediately
  (one request returned 1 token, `''`).
- **Truncation mid-number** — `'...>>120,0 word'` extracted as `1200`.
- **Format mismatch** — `<think>` blocks and LaTeX against a plain-text few-shot prompt.

Scoring this model needs a thinking-aware harness. Don't quote 0.400 in either direction.

There is also no Q4_K_M-vs-bf16 comparison for this model and cannot be one here (bf16
needs ~54 GB). The quantization-loss check for this branch was done on a 1B proxy —
0.325 vs 0.330 on GSM8K 200q, one question apart — recorded in `14b767d92e`.

## Known issues

**Intermittent hang — the main blocker, unresolved.** Not concurrency-driven:
`--parallel 10` and `--parallel 1` completed, `--parallel 5` hung; two single-request
warmup hangs and two successes. Ruled out: memory (reproduces with 40 GiB free), request
shape (warmup sends the exact prompt that succeeds with `--skip-server-warmup`), and
kernel-ordering races (succeeds with and without `AMD_SERIALIZE_KERNEL=3`). Specific to
the Gated DeltaNet path — a 1B GGUF on the same build was stable across every run.
Untested hypothesis: requesting while the 16.5 GB load still drains through the page
cache wedges WSL2's `dxgkrnl` (`dxgvmb_send_create_allocation failed ffffffb5`,
`Ioctl failed: -75`) instead of returning OOM.

**Required flags.** `SGLANG_MAMBA_CONV_DTYPE=float16` — `configs/mamba_utils.py:68`
hardcodes the conv-state dtype to bfloat16, so GDN under `--dtype float16` dies at
`gdn_backend.py:911`. `--max-total-tokens` — `is_amd_apu()` fixes the 108 GiB
over-report, but `_profile_available_bytes` still claims nearly all system RAM for KV,
leaving too little for activations.

**`--dtype bfloat16` hangs in warmup.** Not investigated; use `float16`.

## Upstream follow-ups

GemmaRMSNorm `+1` unfold and the GGUF rope unpermute for llama/mistral/mixtral both
affect CUDA today. Also: `conv_dtype` hardcoded to bfloat16, unmapped GGUF names
colliding, `is_amd_apu` over-commit, the IQ3_S `iq3xs_grid` bug, and the missing `sudot4`
path (RDNA3/3.5 only).
