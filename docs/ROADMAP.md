# Verify-safe headroom past 506.74 — investigation + roadmap

Multi-agent investigation across all shared knowledge + the top verified competitor stacks,
graded through the real gate (**TPS reproducibility**, not PPL margin — see [`SCORING.md`](SCORING.md)).

## TL;DR

**We are at / within ~1–2% of the verify-safe ceiling for this architecture.** Config, quant,
lm_head and decode-kernel headroom is **essentially spent** — confirmed by audit: our stack already
runs the full frontier config and the body is already **int4 g128** (channel-wise int4 lm_head).
Realistic verify-safe gain in remaining benchmark slots is **~+3 to +10 tok/s (config-grade)**, and
even that is uncertain. Anything materially bigger (**+50 class**) requires a **different drafter
architecture** — real R&D, multi-session, with genuine reproduction risk.

The verified 494–506 frontier *is our own shared osoi5 stack*; competitors differ only by config.

## Why config is tapped (audited)

- ✅ already running: `FEOPT_ORJSON`, `DETOK_ENDONLY`, `SPLITKV_VERIFY`, `FA_SLIDING` (w192),
  `ONEGRAPH`/loopgraph, `FUSED_SPARSE_ARGMAX`, `DIXIE_FUSED_ACCEPT_PREP`, `PLE_FOLD`, MTP K=7.
- Body is **already g128** → the "g32→g128 re-quant (+7%)" lever does **not** apply to us.
- lm_head already untied + channel int4 + pruned to 12k (0.03 ms, <2% of step) → harder pruning ≈ **0 tok/s**.
- Decode is ~92% weight-GEMM bandwidth-bound; attention/KV/sampling/graph levers are harvested.
- Sub-4-bit (3/2-bit, FP4, fp8-KV) is **hard-blocked on vLLM 0.22 + Ampere sm_86**.

## Ranked opportunities

### Quick wins (config, 1 slot) — small
| lever | gain | invariant? | risk | verdict |
|---|---|---|---|---|
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | single-digit % on the int4 body | yes (weight-GEMM) | atomicAdd reorders float reduction → **must pass a spec-off reference greedy-identity check first** | the *only* untried config knob; modest, worth one gated slot |

### Dead ends — do NOT spend slots
K7→8 / dynamic-K (regressed −9.8; killed 511.69) · window 192→160/128 (prompt-sensitive; broke
sparkgemma's CUDA-graph tiling) · precache (canonical private-Δ fail) · 12k→8k lm_head prune
(~0 tok/s + OOV risk) · sub-4-bit / NF4 / fp8-KV (blocked on Ampere) · g256/int3/all-channel scales
(PPL-only) · **wider-corpus KL-distill drafter (falsified: −1.5 tok/s; the offline acceptance metric
is invalid — it scores HF numerics that drift 1.3–1.5%/token from the int4 serve kernels).**

### R&D — the only real ceiling-raisers (multi-session)
| project | ceiling | risk |
|---|---|---|
| **QAT drafter matched to the EXACT served int4 numerics** | open (+20–60 if it transfers) | medium — fixes the HF↔int4 drift that killed KL; but acceptance gains are the canonical thing that *doesn't reproduce*, and the "conservation law" hints e1 may already be near serve-optimal for the MTP family |
| **PARD / EAGLE-3 flat-acceptance parallel drafter** | highest (mean-accept ~3.4→~7-8) | high — unbuilt, integration risk; the second bet if QAT-match stalls |

## Scoped R&D — QAT drafter aligned to served numerics (the "increase scope" project)

**Root cause it fixes:** the prescribed KL-distill failed because acceptance was optimized against
HF-numerics argmaxes that drift ~1.3–1.5%/token from the int4 serve kernels (+0.12 acc-tok/step
offline → −1.5 tok/s served). A drafter QAT'd to the *exact served* target removes the proxy gap.

1. **Corpus** (the expensive part): ≥9k distribution-matched prompts across the 4 eval distributions;
   capture ~1.08M propose-call traces **from the actual served int4 stack** (osoi5 g128), *not* HF softmax.
2. **Train:** small drafter (~4 layers / hidden-256 / ~150M), warm-start from kenyan-duma e1, **QAT**
   so draft numerics match the served target. ~1 epoch ≈ a few H100-hours (~$5–15). Compute is cheap;
   corpus capture is the cost.
3. **Offline durability gate (before any benchmark slot):** score acceptance vs **served int4 argmaxes**
   on a **held-out split disjoint from training** (private-set proxy); require durable **+≥0.05 acc-tok/step**
   holding across *all 4* distributions. `offline_acceptance.py` as-shipped is **invalid here** (HF proxy).
4. **Final gate:** served `/metrics` accept counters on a real run — never the offline simulator.
5. **+1 accepted token ≈ +107 tok/s**, so a fractional durable lift is the whole game — but unquantified
   on the 506.74 stack and gated entirely on held-out/served durability.

**Honest EV:** uncertain. The lever with the most leverage (acceptance) is also the one most prone to
private-set reproduction failure. Treat this as a research bet, not a sure gain.
