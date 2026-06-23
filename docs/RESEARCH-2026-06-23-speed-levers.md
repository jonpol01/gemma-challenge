# Speed-lever research sweep — 2026-06-23

A from-scratch, adversarial re-investigation of "what can still beat 506.74 tok/s on the verified
board," prompted by the board moving past us. Method: a multi-agent workflow — **7 expert research
avenues + a completeness critic → 48 candidate levers → 32 unique → a 2-pass adversarial grade
(grade, then red-team each survivor against the private-reproducibility gate) → synthesis.** Every
agent was fed the full constraint set ([`SCORING.md`](SCORING.md)) and the documented dead-list so
it could not hand back reheated dead ends.

**Bottom line: the sweep CONFIRMS [`ROADMAP.md`](ROADMAP.md).** Config/quant/kernel/lm_head headroom
is spent. The only real ceiling-raiser is still a different drafter — and the sweep adds the
*quantified* reason that bet is a coin-flip, plus one genuinely-new prompt-invariant sliver worth
scoping. 3 of 32 levers survived grading, all "marginal."

---

## 1. The board is not what "we lost #1" implies

Live verified board (best-per-agent) at time of writing:

| TPS | status | agent | method |
|---:|---|---|---|
| 513.77 | **pending** | inifinityoptimizer | **w160**-ctk42 |
| 512.59 | **pending** | gemma-slayer | **w160**-ctk44 |
| **506.94** | valid | vidraft-darwin | fw192-**ctk49** |
| **506.74** | valid | **mikasa-inbound (us)** | hayai w192-ctk48-mtp-k7 |
| 506.63 | valid | sparkgemma-s46b | w192-ctk48 |

- The two numbers "above" us are both **`w160` `pending`** — the perpetual non-converting churn this
  repo has documented from the start. No `w160` has *ever* converted on the private gate. They are
  not real standing.
- The actual **verified** board is a **3-way tie at the ceiling: 506.63 / 506.74 / 506.94 — a spread
  of 0.3 tok/s ≈ 0.06%**, below the single-run noise floor (~0.2%; the speed stage runs **once**, no
  averaging). vidraft-darwin's +0.20 over us is a lucky roll, not a better stack — and more centroids
  (`ctk49` vs our `ctk48`) means *more* work, not less, so the knob doesn't even explain it.
- Practical consequence: **reclaiming verified #1 is a variance re-roll, not an R&D problem** (see §6).

---

## 2. The one quantified new finding (why a better drafter is a coin-flip)

`+1 accepted token/step ≈ +107 tok/s`, so acceptance is the whole game — but it is also the lever the
private gate kills. The sweep put a number on it:

> **Gemma-4 EAGLE-3 shows a 64% cross-task acceptance spread (1.05×–1.72× from SWEBench↔MT-Bench).
> The private re-verify gate is ±5%.**

The gap between a 64% cross-distribution spread and a 5% gate means the private re-verify outcome is
essentially a bet on whether the organizers' hidden prompts happen to be MT-Bench-like (best case) or
SWEBench-like (worst case) — a factor entirely outside our control, and one no EAGLE-3.1 robustness
fix addresses (FC-normalization targets attention drift at depth, not domain-level predictability).
This is the *same* mechanism that busts every `w160` and that regressed K=8. It is the structural
reason the frontier is stuck at ~506.

---

## 3. What survived grading (3 marginals)

| lever | realistic | survives private? | feasible now? | note |
|---|---|---|---|---|
| **Parallel-draft (PARD / mask-token)** | **~tens of tok/s *if* acceptance holds** | **gated on private — NOT separable** ⚠️ | **vLLM-ready; needs a PARD drafter retrain** | **corrected by the 2026-06-23 sm_86 spike — see §3.1** |
| Broaden drafter corpus + training-time-test | +0 standalone | likely | yes | only a *prerequisite* for a PARD/EAGLE-3 drafter |
| Remove per-step D2H accept-count sync | +0–1 tok/s | likely | yes | `nsys`-first; likely already pipelined |

> ⚠️ **Update (2026-06-23 spike):** the original "Claim A" idea below was that the parallel-draft
> *overhead* gain is prompt-invariant and separable from acceptance. **An sm_86 spike disproved that.**
> §3.1 is rewritten with the measured result; the old text is kept struck-through for the record.

### 3.1 The parallel-draft lever — characterized on sm_86 (RTX-3080 spike, 2026-06-23)

A spike on the RTX-3080 (Ampere **sm_86**, same arch as the A10G; torch 2.11/cu130, triton 3.6,
CUDA-graph capture all work) settled this lever. **Bottom line: the speed prize is real (~400 µs) and
vLLM is already wired for it — but the prize is NOT prompt-invariant or separable from acceptance, so
it's gated on the private re-verify like every other acceptance lever.**

- **Confirmed in our own code (`sitecustomize.py:174-193`):** the loopgraph runs the drafter as
  **K=7 sequential full forwards** per token (`self.model(...)` in `for index in range(token_count)`,
  one position each), autoregressively.
- **`parallel_drafting` is real and already inherited by `Gemma4Proposer`.** It subclasses
  `SpecDecodeBaseProposer` (`llm_base_proposer.py`), which holds the entire one-forward-pass subsystem
  (mask-token Triton kernels in `utils.py`, `only_one_forward_pass = is_graph_capturing or
  self.parallel_drafting`). **No vLLM engine surgery is needed** — contrary to the earlier "not wired"
  guess. BUT it only activates for a drafter trained with a **mask-token scheme** — it reads
  `pard_token` / `mask_token_id` / `ptd_token_id` from the drafter config (lines 331-335). Our
  kenyan-duma drafter (H=256, 4 layers, centroid-masked full-vocab head, **fp16 ~73M params**) has
  none, so flipping the flag would feed it mask tokens it was never trained on → ~0 acceptance.
  **The blocker is a PARD-style drafter *retrain*, not code.**
- **The speed prize is ~400 µs, not 60–90 µs** (microbench, *real* drafter dims, sm_86, inside a CUDA
  graph): 7 sequential drafter steps = **481 µs** vs 1 parallel pass = **83 µs** → **~398 µs (83%)
  saved**, roughly **10–20% of a 2 ms token budget**. At hidden-256 each forward is launch/latency-
  bound (many tiny kernels × 4 layers), so 7 sequential pay that 7× *even captured in a graph* — the
  graph can't remove it. (An earlier proxy with the wrong dims — hidden-2560, dense head — overstated
  this ~8×; corrected here.)
- **The catch — the prize is NOT separable from acceptance.** The only way to collapse 7 forwards → 1
  is predict-K-from-1 (mask-token PARD), which changes the draft distribution → **acceptance becomes
  prompt-sensitive** → gated on the ±5% private re-verify, exactly like w160/EAGLE-3. There is no
  prompt-invariant route to the speedup. *(This retracts the original "Claim A is prompt-invariant and
  separable" framing.)*
- **int4-quantizing the drafter is dead:** the drafter is latency-bound at hidden-256, not bandwidth-
  bound, so fewer bytes don't speed up its (tiny, launch-bound) kernels.

**Net:** more real (hundreds of µs) and more integrable (vLLM-ready, no surgery) than first thought —
but it's a PARD-drafter **retrain** whose entire payoff rides on the private gate. It does **not**
escape §2; it's decided by the same acceptance-variance pre-gate (§4). Run that before building anything.

---

## 4. The free go/no-go before ANY drafter R&D

Before spending a session or a slot on a new drafter, run the **acceptance-variance pre-gate** —
**0 benchmark slots**, uses the existing [`drafter-rnd/`](../drafter-rnd) pipeline:

1. Capture current **kenyan-duma K=7** acceptance length across **4 held-out distributions**
   (code / math / chat / reasoning) **on the served int4 stack** (`capture_argmax.py` +
   `measure_accept.py` — never the offline HF proxy).
2. **Variance > 8%** → any acceptance-based lever is dead-on-arrival on the ±5% gate → **hold 506.74**.
   **Variance < 5%** → the parallel-draft bet is justified; proceed to the §3.1 kernel spike.

This converts the open "is the drafter bet worth it?" question into a cheap, served-numerics
measurement instead of a paid benchmark gamble.

---

## 5. Confirmed dead — do NOT spend slots (with reasons)

Everything below was graded and red-teamed to **dead**.

**Speculative / drafter architectures**
- **EAGLE-3 / EAGLE-3.1** trained on served int4 — mechanism is sound, but the 64% cross-task spread
  (§2) structurally violates the ±5% gate; +overhead disproportionate at a 2 ms/token budget; no E4B
  drafter exists; piecewise CUDA graph conflicts with ONEGRAPH/loopgraph.
- **MatFormer E2B nested self-drafter** — fatal cost math at batch=1: an E2B pass costs ~0.55× a full
  forward *per draft token* (vs ~0.05× for an MTP head); MatFormer's own paper shows only 1.14×.
- **Self-speculative early-exit (Kangaroo / LayerSkip)** — acceptance collapse drafting from int4 PLE
  shallow layers; replaces an already serve-tuned external drafter (−20 to −50 tok/s).

**Kernels / runtime**
- **TensorRT-LLM swap** — greedy-identity-unsafe, multimodal untested, can't reuse the pck04 lm_head
  byte win; its headline 20–40% is vs *unoptimized* vLLM (no graphs/Marlin/FA-sliding) we already bank.
- **Megakernel / Mirage-MPK (and AutoMegaKernel)** — batch-1 *slower* than a graph-captured vLLM loop;
  int4 path immature → regression, not gain.
- **GemLite GEMV_RevSplitK**, **weight re-swizzle**, **hand-written lm_head split-K GEMV**,
  **FlashInfer fused decode** — flat-to-negative at M=1. (The lm_head is **~1.5% of the token budget**
  — deleting it *entirely* is ≤ +7 tok/s; not worth touching.)

**Quant**
- **AWQ re-bake**, **official Gemma-4 QAT-INT4 ckpt**, **int4 g64 head + deeper row-prune** — pure
  PPL-headroom instruments; the headroom budget is already spent, so they net 0 tok/s.
- **Mixed-precision 3-bit bulk + sensitive 4-bit** — same drafter accept-collapse that killed g256.
- **QuaRot / SpinQuant W4A4** — online Hadamard adds per-token FLOPs; non-power-of-2 gemma dims block
  the fast transform (−10 to −30 tok/s).

**lm_head / arch / host**
- **SVD low-rank lm_head** (2nd-GEMV launch overhead > bytes saved), **12k→10k prune** (≈0, consistent
  with the measured 12k→8k=0), **MatFormer body-slice as checkpoint** (PPL gate kills any meaningful
  trim), **AltUp/LAuReL**, **PLE residency**, **decouple multimodal towers** — all ~0 tok/s.
- **Fused dequant+matmul+RMSNorm+residual**, **async-scheduling (V1)**, **drafter/target stream
  overlap**, **compact-logit-space**, **fixed-rig host-free replay**, **DFloat11 entropy coding**,
  **SM-clock/power lock** — already captured by ONEGRAPH, incompatible with MTP K=7, or net-negative.

---

## 6. Recommended sequence

1. **Variance re-roll (free, scratch-only — a long shot, not a free bump).** Re-running the *exact*
   `vllm-hayai-repro-v1` stack through the free org bench is zero-cost and zero-risk *as an
   observation* (scratch bucket, not auto-promoted). But the real cross-job spread on this infra is
   **~3%, not ~0.2%** — measured: identical-stack re-runs drew **491–498 tok/s**, and 506.74 was a
   *favorable* historical draw — so clearing **506.94** is unlikely. And **promoting** a high draw is
   a private-reverify gamble: a high-public/low-private result is the classic invalidation death mode,
   and our 506.74 is already verified and locked (it survives regardless). Net: roll to *observe*, but
   do **not** post a noise-high draw without accepting the re-verify risk.
2. **Acceptance-variance pre-gate (free, §4).** Decides whether the drafter bet is alive at all.
3. **Only if (2) clears: the parallel-draft *overhead* kernel spike (§3.1)** on the sm_86 box — the
   only path to a real, gate-safe, beyond-noise verified gain.

**Honest EV:** near-zero for a *bankable* gain beyond the re-roll. Hold remains the dominant play for
the verified line; the parallel-draft overhead lever is the one bet that could move it without
betting on the private prompt distribution. Spend **zero paid credits** until the free gates clear.
