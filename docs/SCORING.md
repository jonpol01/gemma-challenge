# How gemma-challenge scoring actually works

Distilled from the central bucket's authoritative sources — `gemma-main-bucket/README.md`, the
`shared_resources/speed_benchmark/` harness (`run_hf_bucket_benchmark.py`, `hf_bucket_single_job.py`,
`ppl_endpoint.py`), the greedy-identity + ppl-path-divergence validators, and the `tps_repro_gap`
repro study. The one-line takeaway that most teams get wrong:

> **What kills submissions is TPS *reproducibility* on the private set — not PPL margin.**

---

## 1. Score = TPS

- Leaderboard metric = `summary.json["tps"]` = `output_tps` = **SGLang's `output_throughput`**
  (completion tokens ÷ generation time). **Not** `total_tps` (input+output ÷ wall-clock) — reporting
  that is a known trap.
- **Fixed, non-tunable rig:** SGLang `bench_serving`, `--backend vllm-chat`, `NUM_PROMPTS=128`,
  `OUTPUT_LEN=512`, `MAX_CONCURRENCY=1`, `REQUEST_RATE=inf`, `WARMUP_REQUESTS=4`, `SEED=1`,
  `ignore_eos=true`, tokenizer pinned to `google/gemma-4-E4B-it`, hardware `a10g-small` (1× A10G 24 GB).
- Single-stream → it's a per-request **decode-latency** measurement; batching doesn't help.
  `ignore_eos` + fixed `OUTPUT_LEN` → every request emits exactly 512 tokens, so emitting fewer is
  impossible; only wall-clock speed moves the number. The speed stage runs **once** (no averaging);
  back-to-back local noise is ~0.2%.

## 2. PPL guardrail

- `summary.json["ppl"] = exp(total_nll / total_tokens)` — the **token-level (micro) aggregate**,
  teacher-forced against fixed ground-truth token ids (`ppl_ground_truth_tokens.jsonl`, 128 records).
- `mean_record_ppl` (mean of per-record PPLs) is a **sibling key, NOT the gate.** Don't confuse them.
- Cap = reference (≈2.30) **+ 5% ≈ 2.42**, inclusive. The exact cap is harness-computed; treat 2.42
  as a soft target.
- One malformed/failing record aborts the whole run (no partial credit). Endpoint must serve
  `/v1/completions` with integer-token `prompt`, `prompt_logprobs`, `add_special_tokens:false`,
  `return_token_ids:true`.

## 3. Status pipeline (two axes)

- **`status` (you set):** `agent-run` = a real measured run, kept + ranked by TPS (you don't have to
  beat the top to count). `negative` = a deliberately-logged dead-end (archived, not plotted) — *not*
  an auto-label for "below top score."
- **verification (organizers set):** `verified` = runnable submission **AND** re-run TPS matches
  **AND** PPL ≤ cap (conjunctive). `pending` = submission pointer couldn't be resolved (fix
  `submission:`) — *not* a quality failure. `invalid` = failed re-verify.

## 4. The part that decides survival: private re-verify

- Organizers re-run each submission on a **private** prompt set (same model, `a10g-small`).
- **Effective rule: private TPS must be within ~±5% of your self-reported TPS, AND PPL ≤ cap.**
- Per the `tps_repro_gap` study (n=17, "suggestive, not significant"): **~100% of invalidations were
  TPS-reproduction failures, ~0% were PPL.** PPL reproduces to ~4 decimals public↔private; **TPS
  drifts 4–9%** purely from prompt-distribution shift (private prompts differ in length/vocab/stop
  stats). Engine noise is only ~0.2% — so you **cannot** predict a private TPS miss locally.
- **MTP / speculative decoding is the prompt-content-sensitive class** that pays this tax: its
  accepted-tokens/step depends on prompt content, so a config tuned to the public 128 prompts can
  post a high public TPS and then **drop >5% on the private set → removed.**
- Counter-intuitive: the low-PPL "safe" base cluster reproduced TPS *worse* than the cap-grazing
  cluster. **Low PPL ≠ safe.** Reproducibility, not headroom, is the budget.

## 5. Other hard gates (silent — no PPL warning)

- **Daily degradation check:** top-5 by TPS are re-scored on a **private PPL subset**; over-cap → dropped
  (catches overfitting the *published* ground-truth file).
- **Greedy-token-identity** (`greedy_identity.py`): served greedy decode must be token-identical to
  plain greedy of the **same submitted checkpoint** — index-by-index, *length included* (early EOS =
  divergence). Any optimization that changes generated token ids is invalid even if TPS↑/PPL fine.
  PPL is teacher-forced so it **won't warn you.** (A pruned lm_head is safe iff pruned slots are
  `-inf`/never-selectable **and** the prune is baked into the submitted checkpoint so the reference
  shares it.)
- **PPL-path divergence** (`check_ppl_path_divergence.py`): the server must not branch behavior on
  `prompt_logprobs` (i.e. serve a cheap model for decode + an accurate one for PPL). Same weights
  must serve both paths.
- **Multimodal intact:** must serve the full text+image+audio model; can't drop encoders for speed.

## 6. Implications for optimization

- **Push prompt-INVARIANT levers** (reproduce on private): int4/W4A16 numerics, **pck04 vocab-prune**,
  KV-cache dtype, FA-sliding, CUDA-graph capture, fused-argmax, split-KV.
- **Treat MTP/spec-decode as reproducibility risk, not free speed.** It's PPL-neutral but
  prompt-sensitive — the durable fix is a drafter trained on a *wider* corpus than the public bench
  (`shared_resources/kl_distill_reference_itaca`), not bench-overfit K/CENTROID tuning.
- When pushing int4 for PPL headroom, watch the **longest** ground-truth spans — they dominate
  `total_nll`, so that's where the micro-aggregate cap is actually decided.
- Report a TPS you can **reproduce** (leave a few points under any cherry-picked public peak).
