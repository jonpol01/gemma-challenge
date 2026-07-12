#!/usr/bin/env python3
"""CE exact-match trainer for the Gemma4 MTP drafter (QAT-exact-match recipe).
Trimmed from the reference KL trainer: pure cross-entropy on the int4 target
argmax — no top-k logits needed. Trace line: {prefix_token_ids, target_argmax_id}."""
import argparse, json, os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class TraceJSONL(IterableDataset):
    def __init__(self, path, max_prefix_len=2048):
        self.path, self.m = path, max_prefix_len

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        nw = wi.num_workers if wi else 1
        with open(self.path) as fh:
            for i, line in enumerate(fh):
                if i % nw != wid:
                    continue
                r = json.loads(line)
                p = r["prefix_token_ids"][-self.m:]
                if len(p) < 1:
                    continue
                yield torch.tensor(p, dtype=torch.long), int(r["target_argmax_id"])


def collate(batch, pad):
    sm = max(p.size(0) for p, _ in batch)
    B = len(batch)
    pref = torch.full((B, sm), pad, dtype=torch.long)
    am = torch.zeros(B, sm, dtype=torch.bool)
    lp = torch.zeros(B, dtype=torch.long)
    tgt = torch.zeros(B, dtype=torch.long)
    for i, (p, a) in enumerate(batch):
        L = p.size(0)
        pref[i, :L] = p; am[i, :L] = True; lp[i] = L - 1; tgt[i] = a
    return pref, am, lp, tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--init-revision", default=None)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-prefix-len", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--freeze-embed", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[t] device={dev} init={a.init}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.init, revision=a.init_revision, trust_remote_code=True)
    pad = tok.pad_token_id or 0
    model = AutoModelForCausalLM.from_pretrained(
        a.init, revision=a.init_revision, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev)
    model.config.use_cache = False
    if a.grad_checkpoint:
        try:
            model.gradient_checkpointing_enable()
            print("[t] gradient checkpointing ON", flush=True)
        except Exception as e:
            print(f"[t] grad checkpoint unavailable: {e}", flush=True)
    if a.freeze_embed:
        nf = 0
        for n, p in model.named_parameters():
            if "embed" in n:
                p.requires_grad_(False)
                nf += p.numel()
        print(f"[t] froze embed params ({nf/1e6:.1f}M)", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    tot = sum(p.numel() for p in trainable)
    print(f"[t] trainable params: {tot/1e6:.1f}M across {len(trainable)} tensors", flush=True)
    model.train()
    loader = DataLoader(TraceJSONL(a.corpus, a.max_prefix_len), batch_size=a.batch_size,
                        num_workers=a.workers, collate_fn=lambda b: collate(b, pad), pin_memory=True)
    opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    step, ema = 0, 0.0
    for ep in range(a.epochs):
        for pref, am, lp, tgt in loader:
            pref, am, lp, tgt = pref.to(dev), am.to(dev), lp.to(dev), tgt.to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = model(input_ids=pref, attention_mask=am, use_cache=False)
                B = pref.size(0)
                ll = out.logits[torch.arange(B), lp]
                loss = F.cross_entropy(ll, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema = 0.99 * ema + 0.01 * float(loss) if step else float(loss)
            step += 1
            if step % a.log_every == 0:
                print(f"[t] ep{ep} step{step} loss={float(loss):.4f} ema={ema:.4f}", flush=True)
    ck = os.path.join(a.out, "final")
    os.makedirs(ck, exist_ok=True)
    model.save_pretrained(ck, safe_serialization=True)
    tok.save_pretrained(ck)
    print(f"[t] saved {ck} (steps={step})", flush=True)


if __name__ == "__main__":
    main()
