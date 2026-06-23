#!/usr/bin/env python3
"""EAGLE-style joint trainer for the Gemma4 MTP drafter — exact-match to the int4 target.

Recipe (from transformers SinglePositionMultiTokenCandidateGenerator.get_candidates):
  target text model with return_shared_kv_states=True -> last_hidden_state, shared_kv_states
    (shared_kv_states = {"full_attention":(K,V), "sliding_attention":(K,V)}, last layer of each type)
  to predict the token after position p (prefix = full[:p+1]):
    emb   = target.embed_tokens(full[p])            # (1,1,2560)  raw, unscaled
    hid   = last_hidden_state[:, p, :]              # (1,1,2560)
    inp   = cat([emb, hid], -1)                     # (1,1,5120)  -> drafter.pre_projection
    skv_p = {k:(K[:,:,:p+1,:], V[:,:,:p+1,:]) ...}  # cropped to current length
    logits= drafter(inputs_embeds=inp, position_ids=[[p]], shared_kv_states=skv_p, use_cache=False).logits
    label = full[p+1]                               # target's greedy next token (captured)
  CE(logits, label); backprop drafter only (target frozen).

--smoke: stock drafter, no training, report argmax match-rate vs label. High rate => the
reconstruction matches what the drafter was trained/served with; near-zero => wiring is wrong.
"""
import argparse, json, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma4ForConditionalGeneration


def load_jsonl(path, limit):
    rows = []
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
        if limit and len(rows) >= limit:
            break
    return rows


def find_text_model(m):
    """Locate the Gemma4 text decoder (has embed_tokens + layers, emits shared_kv_states)."""
    for path in ("model.language_model", "language_model", "model.model", "model"):
        o = m
        ok = True
        for a in path.split("."):
            if hasattr(o, a):
                o = getattr(o, a)
            else:
                ok = False
                break
        if ok and hasattr(o, "layers") and hasattr(o, "embed_tokens"):
            return o, path
    raise RuntimeError("could not locate text decoder (embed_tokens+layers)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int4", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--traces", required=True)        # {prompt, target_token_ids} per line
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0)   # prompts (0=all)
    ap.add_argument("--max-pos", type=int, default=64)  # positions/prompt (0=all)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=50)
    a = ap.parse_args()
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    print(f"[e] loading target int4: {a.int4}", flush=True)
    target = Gemma4ForConditionalGeneration.from_pretrained(
        a.int4, dtype=torch.bfloat16, trust_remote_code=True,
        ignore_mismatched_sizes=True, device_map="cuda").eval()
    for p in target.parameters():
        p.requires_grad_(False)
    text, where = find_text_model(target)
    print(f"[e] text decoder at '{where}'; layers={len(text.layers)} hidden={text.embed_tokens.embedding_dim}", flush=True)
    print(f"[e] GPU after target load: {torch.cuda.memory_allocated()/1e9:.1f}GB alloc", flush=True)
    emb = text.embed_tokens

    print(f"[e] loading drafter: {a.drafter}", flush=True)
    drafter = AutoModelForCausalLM.from_pretrained(a.drafter, dtype=torch.bfloat16, trust_remote_code=True).to(dev)
    print(f"[e] GPU after drafter load: {torch.cuda.memory_allocated()/1e9:.1f}GB alloc", flush=True)
    opt = None
    if a.smoke:
        drafter.eval()
    else:
        drafter.train()
        # Train against the FULL lm_head logits (stable CE). The ordered-embeddings/masked head
        # fills non-selected tokens with ~-inf -> CE explodes (300+) when the true token is masked
        # out -> divergence. Weights are shared, so full-CE training still improves the serving head.
        if getattr(drafter.config, "use_ordered_embeddings", False):
            drafter.config.use_ordered_embeddings = False
            print("[e] training with FULL lm_head logits (ordered-emb OFF for stable CE)", flush=True)
        opt = torch.optim.AdamW([p for p in drafter.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.95))
        print(f"[e] trainable {sum(p.numel() for p in drafter.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)

    rows = load_jsonl(a.traces, a.limit)
    print(f"[e] {len(rows)} prompts | smoke={a.smoke} max_pos={a.max_pos} accum={a.accum}", flush=True)

    match = tot = steps = 0
    running = 0.0
    t0 = time.time()
    for ri, row in enumerate(rows):
        traj = row.get("target_token_ids") or []
        if len(traj) < 2:
            continue
        ptoks = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                        add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
        if ptoks and isinstance(ptoks[0], list):
            ptoks = ptoks[0]
        full = list(ptoks) + list(traj)
        ids = torch.tensor([full], device=dev)
        T = len(full)
        p0 = len(ptoks) - 1                       # predict first traj token from last prompt tok
        p1 = T - 1
        if a.max_pos:
            p1 = min(p1, p0 + a.max_pos)
        with torch.no_grad():
            tout = text(input_ids=ids, return_shared_kv_states=True, use_cache=False)
            last_hidden = tout.last_hidden_state          # (1,T,2560)
            skv = tout.shared_kv_states
        for p in range(p0, p1):
            e = emb(ids[:, p:p + 1])                      # (1,1,2560)
            h = last_hidden[:, p:p + 1, :]                # (1,1,2560)
            inp = torch.cat([e, h], dim=-1)               # (1,1,5120)
            skv_p = {k: (K[:, :, :p + 1, :], V[:, :, :p + 1, :]) for k, (K, V) in skv.items()}
            pos = torch.tensor([[p]], device=dev, dtype=torch.long)
            label = torch.tensor([full[p + 1]], device=dev)
            if a.smoke:
                with torch.no_grad():
                    lo = drafter(inputs_embeds=inp, position_ids=pos, shared_kv_states=skv_p, use_cache=False).logits
                match += int((lo[:, -1, :].argmax(-1) == label).item())
                tot += 1
            else:
                lo = drafter(inputs_embeds=inp, position_ids=pos, shared_kv_states=skv_p, use_cache=False).logits
                loss = F.cross_entropy(lo[:, -1, :], label) / a.accum
                loss.backward()
                tot += 1
                if tot % a.accum == 0:
                    torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    L = float(loss) * a.accum
                    running = 0.99 * running + 0.01 * L if steps else L
                    steps += 1
                    if steps % a.log_every == 0:
                        print(f"[e] step{steps} loss={L:.4f} ema={running:.4f} pos={tot} t={time.time()-t0:.0f}s", flush=True)
        if a.smoke and (ri < 5 or ri % 20 == 0):
            print(f"[e] prompt{ri}: match {match}/{tot} = {match/max(tot,1):.3f}", flush=True)

    if a.smoke:
        print("RESULTJSON " + json.dumps({"match": match, "tot": tot, "rate": round(match / max(tot, 1), 4)}), flush=True)
    else:
        if opt is not None:
            opt.step()
            opt.zero_grad(set_to_none=True)
        if hasattr(drafter.config, "use_ordered_embeddings"):
            drafter.config.use_ordered_embeddings = True   # restore fast serving head for vLLM
        if a.out:
            drafter.save_pretrained(a.out + "/final", safe_serialization=True)
            tok.save_pretrained(a.out + "/final")
            print(f"[e] saved {a.out}/final steps={steps} pos={tot} t={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
