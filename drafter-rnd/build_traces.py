#!/usr/bin/env python3
"""Build CE training traces from captured int4 argmax trajectories.
Input line:  {prompt, target_token_ids}
Output line: {prefix_token_ids, target_argmax_id}
prefix = chat_template(prompt) + trajectory[:i]; target = trajectory[i]. One row per
position — this is plain next-token training on the int4 target's greedy trajectory,
so the drafter learns to predict exactly what the int4 model emits (exact-match)."""
import argparse, json
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--in-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)    # cap prompts (0 = all)
    ap.add_argument("--max-pos", type=int, default=0)  # cap positions/prompt (0 = all)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    npr = nrec = 0
    with open(a.in_file) as f, open(a.out, "w") as o:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if a.limit and npr >= a.limit:
                break
            d = json.loads(line)
            traj = d.get("target_token_ids") or []
            if not traj:
                continue
            enc = tok.apply_chat_template(
                [{"role": "user", "content": d["prompt"]}],
                add_generation_prompt=True, tokenize=True, return_dict=True)
            ptoks = enc["input_ids"]
            if ptoks and isinstance(ptoks[0], list):  # nested batch -> take first
                ptoks = ptoks[0]
            T = len(traj) if not a.max_pos else min(len(traj), a.max_pos)
            for i in range(T):
                o.write(json.dumps({
                    "prefix_token_ids": ptoks + traj[:i],
                    "target_argmax_id": int(traj[i]),
                }) + "\n")
                nrec += 1
            npr += 1
    print(f"BUILT {nrec} records from {npr} prompts -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
