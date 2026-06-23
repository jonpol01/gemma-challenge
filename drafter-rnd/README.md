# Drafter R&D — the MTP speculative-decoding lever

The **only** lever with the leverage to move the verified frontier by a meaningful margin
(+1 accepted token/step ≈ **+107 tok/s** on this stack) is a better MTP drafter. Everything else
— quant, lm_head prune, decode kernels, config — is [tapped](../docs/ROADMAP.md). This dir is the
pipeline we built to chase it, plus the honest result: **no verified gain over the shared
`kenyan-duma` e1 drafter yet.** It's here as the foundation for the one open bet (PARD / EAGLE-3 —
see [ROADMAP](../docs/ROADMAP.md#rd--the-only-real-ceiling-raisers-multi-session)).

## The core insight (why the prescribed approach failed)

The challenge's shipped `offline_acceptance.py` scores a drafter against **HF-softmax** argmaxes.
But we serve an **int4** target (osoi5 g128) whose greedy tokens **drift ~1.3–1.5%/token** from HF
numerics. So a drafter optimized offline against HF can show **+0.12 acc-tok/step offline yet
−1.5 tok/s served** — the proxy is *invalid*. This killed the wider-corpus KL-distill drafter
([`train_kl_drafter.py`](train_kl_drafter.py), falsified). **The fix:** train and gate against the
**exact served int4 argmaxes**, never HF.

## Pipeline (stages)

| stage | script | what it does |
|---|---|---|
| 1 · corpus | [`build_corpus.py`](build_corpus.py) | distribution-matched prompt corpus across the 4 eval distributions (the expensive part) |
| 2 · capture | [`capture_argmax.py`](capture_argmax.py) | capture argmax trajectories **from the actual served int4 stack** (training targets) — *not* HF softmax |
|   · traces | [`build_traces.py`](build_traces.py) | assemble per-position next-token training traces from the captured greedy trajectory |
| 3 · train | [`train_eagle.py`](train_eagle.py) | EAGLE-style joint trainer: drafter predicts the int4 target's next greedy token from `cat(embed, last_hidden)` + cropped `shared_kv_states`; target frozen, drafter trainable |
|   · (dead) | [`train_kl_drafter.py`](train_kl_drafter.py) | the **falsified** KL-distill-against-HF approach — kept as the documented dead-end |
| 4 · gate | [`measure_accept.py`](measure_accept.py) | served mean-acceptance-length for a (target, drafter) pair via the real serve path |
|   · bench | [`bench_tps.py`](bench_tps.py) | tok/s bench (`--ignore-eos` → fixed length, pure speed) |
| model | [`modeling_gemma4_assistant.py`](modeling_gemma4_assistant.py) | the Gemma4 MTP drafter (single-position multi-token candidate generator) definition |

## EAGLE training recipe (`train_eagle.py`)

Mirrors `transformers` `SinglePositionMultiTokenCandidateGenerator.get_candidates`: run the
target text model with `return_shared_kv_states=True` → `last_hidden_state` +
`shared_kv_states = {"full_attention":(K,V), "sliding_attention":(K,V)}` (last layer of each
attention type). To predict the token after position `p`:

```
emb    = target.embed_tokens(full[p])                 # (1,1,2560) raw, unscaled
hid    = last_hidden_state[:, p, :]                    # (1,1,2560)
inp    = cat([emb, hid], -1)                           # (1,1,5120) -> drafter.pre_projection
skv_p  = {k: (K[:,:,:p+1,:], V[:,:,:p+1,:]) ...}       # shared_kv cropped to current length
logits = drafter(inputs_embeds=inp, position_ids=[[p]], shared_kv_states=skv_p).logits
label  = full[p+1]                                     # target's greedy next token (captured)
CE(logits, label); backprop drafter only.
```

`--smoke` runs the stock drafter (no training) and reports argmax match-rate vs label — a wiring
check: high rate ⇒ the reconstruction matches what the drafter was served with; near-zero ⇒ wiring
is wrong.

## Result (honest)

- The **EAGLE-from-served-numerics** design is correct and removes the proxy gap that killed KL.
- But within the work done, it **did not produce a durable, verified acceptance gain** over the
  shared `kenyan-duma` e1 drafter (already MTP K=7, serve-optimal for this family). The
  "conservation law" hint in the ROADMAP holds: e1 looks near the ceiling for the *single-position*
  MTP family.
- **The real ceiling-raiser is a different architecture** — PARD / EAGLE-3 flat-acceptance parallel
  drafter (mean-accept ~3.4 → ~7–8), which this pipeline (corpus + served-capture + served gate)
  is the substrate for. High EV, high risk, multi-session. See
  [`docs/ROADMAP.md`](../docs/ROADMAP.md).

> ⚠️ Gate every drafter on **served int4 acceptance on a held-out split**, never the offline
> HF-proxy simulator. That mistake is the single most expensive lesson in this dir.
