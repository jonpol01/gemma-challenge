#!/usr/bin/env python3
"""Refresh the README leaderboard block from the live gemma-challenge board.

Fetches the public `GET /v1/leaderboard` (tokenless), rewrites only the text
between the <!-- LEADERBOARD:START --> / <!-- LEADERBOARD:END --> markers, and
saves the raw JSON to data/leaderboard.json for history. Run hourly by
.github/workflows/update-leaderboard.yml (or manually).

Stdlib only — no pip deps, so CI needs nothing but Python.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

API = os.environ.get("GEMMA_API", "https://gemma-challenge-gemma-bucket-sync.hf.space")
AGENT = os.environ.get("GEMMA_AGENT", "mikasa-inbound")
TOP_N = 8

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_chart import render as render_chart  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")
SNAPSHOT = os.path.join(ROOT, "data", "leaderboard.json")
CHART = os.path.join(ROOT, "assets", "score-evolution.svg")
START = "<!-- LEADERBOARD:START -->"
END = "<!-- LEADERBOARD:END -->"
EMO = {"valid": "✅ valid", "pending": "⏳ pending", "invalid": "❌ invalid"}


def fetch(path: str):
    req = urllib.request.Request(API + path, headers={"User-Agent": "gemma-readme-bot"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def standing(raw_rank, verified_rank, our) -> str:
    if not our:
        return "**Our standing:** not currently on the board."
    tps = f'{our.get("tps", 0):.2f}'
    v = our.get("verification")
    if verified_rank == 1 and v == "valid":
        return (f"**Our standing:** **#{raw_rank} raw · #1 verified** 🥇 "
                f"({tps} tok/s, `valid`) — every higher score is unverified `pending`.")
    tail = f"; #{verified_rank} on the verified board" if verified_rank else ""
    return f"**Our standing:** #{raw_rank} raw ({tps} tok/s, `{v}`){tail}."


def main() -> int:
    data = fetch("/v1/leaderboard?best_per_agent=true")
    rows = data.get("rows", [])
    meta = data.get("meta", {})
    if not rows:
        print("no rows returned; aborting", file=sys.stderr)
        return 1

    our = next((r for r in rows if r.get("agent") == AGENT), None)
    raw_rank = our.get("rank") if our else None
    valid_rows = sorted((r for r in rows if r.get("verification") == "valid"),
                        key=lambda r: r.get("tps", 0), reverse=True)
    verified_rank = next((i + 1 for i, r in enumerate(valid_rows)
                          if r.get("agent") == AGENT), None)

    shown = list(rows[:TOP_N])
    if our and our not in shown:
        shown.append(our)

    lines = ["| # | agent | tok/s | verif |", "|--:|-------|------:|:-----:|"]
    for r in shown:
        agent = r.get("agent", "?")
        tps = f'{r.get("tps", 0):.2f}'
        verif = EMO.get(r.get("verification"), str(r.get("verification")))
        if agent == AGENT:
            tag = ("✅ **valid — #1 verified** 🥇"
                   if verified_rank == 1 and r.get("verification") == "valid" else f"**{verif}**")
            lines.append(f'| **{r.get("rank")}** | **{agent} (us)** | **{tps}** | {tag} |')
        else:
            lines.append(f'| {r.get("rank")} | {agent} | {tps} | {verif} |')
    table = "\n".join(lines)

    gen = str(meta.get("generated_at", ""))[:16].replace("T", " ")
    considered = meta.get("results_considered", "?")
    invalid = meta.get("excluded", {}).get("verification_invalid", "?")

    block = (f"{START}\n"
             f"_Auto-updated hourly from `GET /v1/leaderboard` · live snapshot **{gen} UTC**_\n\n"
             f"{standing(raw_rank, verified_rank, our)}\n\n"
             f"{table}\n\n"
             f"_{considered} results considered · {invalid} invalid excluded · "
             f"{len(valid_rows)} verified entries._\n"
             f"{END}")

    txt = open(README, encoding="utf-8").read()
    if START not in txt or END not in txt:
        print("markers not found in README", file=sys.stderr)
        return 2
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _m: block, txt, flags=re.DOTALL)

    os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
    with open(SNAPSHOT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

    # score-evolution chart from the full (every-attempt) history
    try:
        allrows = fetch("/v1/leaderboard?best_per_agent=false").get("rows", [])
        os.makedirs(os.path.dirname(CHART), exist_ok=True)
        if render_chart(allrows, AGENT, CHART):
            print(f"chart rendered ({len(allrows)} results)")
    except Exception as ex:  # never let the chart break the README refresh
        print(f"chart render skipped: {ex!r}", file=sys.stderr)

    if new != txt:
        with open(README, "w", encoding="utf-8") as f:
            f.write(new)
        print(f"README updated (raw #{raw_rank}, verified #{verified_rank}, {len(valid_rows)} valid)")
    else:
        print("README unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
