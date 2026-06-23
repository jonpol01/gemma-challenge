#!/usr/bin/env python
"""Build a distribution-matched drafter-training corpus from NON-HF (GitHub) sources
— the box's general net is fast; only HF is throttled. Dedups vs the public bench by
512-char prefix hash, reserves a held-out split. CPU only, no model needed."""
import json, urllib.request, hashlib, random, os
random.seed(1)
OUT = "/mnt/e/gemma/corpus"
os.makedirs(OUT, exist_ok=True)

def fetch(url):
    print("fetch", url, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "corpus-bot"})
    return urllib.request.urlopen(req, timeout=180).read()

def pkey(t):
    return hashlib.md5(t.strip()[:512].encode("utf-8", "ignore")).hexdigest()

# --- bench prompts → dedup keys ---
bench_keys = set()
try:
    bench = json.load(open(f"{OUT}/eval_prompts_sharegpt.json"))
    def texts(o):
        r = []
        if isinstance(o, str): r.append(o)
        elif isinstance(o, dict):
            for k in ("prompt", "text", "content", "instruction", "question", "value"):
                if isinstance(o.get(k), str): r.append(o[k])
            for v in o.values(): r += texts(v)
        elif isinstance(o, list):
            for v in o: r += texts(v)
        return r
    for t in texts(bench):
        bench_keys.add(pkey(t))
    print("bench dedup keys:", len(bench_keys), flush=True)
except Exception as e:
    print("bench load warn:", repr(e), flush=True)

prompts = []
# Alpaca (instruct / conversational proxy)
try:
    data = json.loads(fetch("https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"))
    for d in data:
        instr = (d.get("instruction") or "").strip()
        inp = (d.get("input") or "").strip()
        p = (instr + "\n" + inp).strip() if inp else instr
        if len(p) >= 12: prompts.append(("alpaca", p))
    print("alpaca:", sum(s == "alpaca" for s, _ in prompts), flush=True)
except Exception as e:
    print("alpaca FAIL", repr(e), flush=True)
# GSM8K (math)
try:
    txt = fetch("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl").decode("utf-8", "ignore")
    for line in txt.splitlines():
        if line.strip():
            q = (json.loads(line).get("question") or "").strip()
            if len(q) >= 12: prompts.append(("gsm8k", q))
    print("gsm8k:", sum(s == "gsm8k" for s, _ in prompts), flush=True)
except Exception as e:
    print("gsm8k FAIL", repr(e), flush=True)

# dedup (vs bench + internal)
seen, uniq = set(), []
for src, p in prompts:
    k = pkey(p)
    if k in bench_keys or k in seen: continue
    seen.add(k); uniq.append({"source": src, "prompt": p})
print("after dedup:", len(uniq), flush=True)

# stratified cap → ~9k
random.shuffle(uniq)
caps = {"alpaca": 6000, "gsm8k": 3000}
cnt, corpus = {}, []
for r in uniq:
    s = r["source"]
    if cnt.get(s, 0) < caps.get(s, 0):
        corpus.append(r); cnt[s] = cnt.get(s, 0) + 1
print("corpus mix:", cnt, "total", len(corpus), flush=True)

# held-out split (>=900)
random.shuffle(corpus)
nh = max(900, int(0.1 * len(corpus)))
held, train = corpus[:nh], corpus[nh:]
with open(f"{OUT}/corpus_train.jsonl", "w") as f:
    for r in train: f.write(json.dumps(r) + "\n")
with open(f"{OUT}/corpus_heldout.jsonl", "w") as f:
    for r in held: f.write(json.dumps(r) + "\n")
print(f"WROTE train={len(train)} heldout={len(held)} -> {OUT}", flush=True)
