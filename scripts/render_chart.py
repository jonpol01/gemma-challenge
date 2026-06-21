#!/usr/bin/env python3
"""Render the gemma-challenge score-evolution chart as a self-contained SVG.

Stdlib only (no matplotlib) so CI needs no pip installs. Called by
update_leaderboard.py with the full result list (best_per_agent=false):
every result as a faint dot, verified results as navy diamonds, our results
in gold, plus two running-max frontiers — verified (bold, ends at us) and
raw incl. pending (dashed).
"""
from __future__ import annotations

import datetime

W, H = 1200, 440
L, R, T, B = 60, 168, 46, 44           # margins (R is wide for end labels)
NAVY, NAVY2, GRAY, GOLD = "#2b3a7a", "#6675b0", "#b9bdca", "#c5961a"
BG, GRID, TEXT, MUT = "#f3f4fb", "#e3e5ef", "#3a3f55", "#8a8fa3"


def _parse(rows):
    pts = []
    for r in rows:
        ts, tps = r.get("timestamp"), r.get("tps")
        if not ts or not isinstance(tps, (int, float)):
            continue
        try:
            dt = datetime.datetime.strptime(str(ts).replace(" UTC", "").strip(), "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        pts.append((dt, float(tps), r.get("verification"), r.get("agent")))
    pts.sort(key=lambda p: p[0])
    return pts


def _diamond(cx, cy, r, fill, opacity=1.0, stroke="none", sw=0):
    return (f'<path d="M{cx:.1f} {cy-r:.1f}L{cx+r:.1f} {cy:.1f}L{cx:.1f} {cy+r:.1f}'
            f'L{cx-r:.1f} {cy:.1f}Z" fill="{fill}" opacity="{opacity}"'
            f' stroke="{stroke}" stroke-width="{sw}"/>')


def render(rows, our_agent, out_path):
    pts = _parse(rows)
    if not pts:
        return False
    t0, t1 = pts[0][0], pts[-1][0]
    span = max((t1 - t0).total_seconds(), 1.0)
    ymax = (int(max(p[1] for p in pts) / 50) + 1) * 50
    pw, ph = W - L - R, H - T - B

    def X(dt):
        return L + (dt - t0).total_seconds() / span * pw

    def Y(v):
        return T + (1 - v / ymax) * ph

    def frontier(pred):
        out, best = [], None
        for dt, v, verif, _ in pts:
            if pred(verif) and (best is None or v > best):
                best = v
                out.append((dt, v))
        return out

    def step_path(fr):
        if not fr:
            return ""
        d = [f"M{X(fr[0][0]):.1f} {Y(fr[0][1]):.1f}"]
        for i in range(1, len(fr)):
            d.append(f"L{X(fr[i][0]):.1f} {Y(fr[i-1][1]):.1f}")
            d.append(f"L{X(fr[i][0]):.1f} {Y(fr[i][1]):.1f}")
        d.append(f"L{X(t1):.1f} {Y(fr[-1][1]):.1f}")     # extend flat to right edge
        return "".join(d)

    e = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="ui-sans-serif,system-ui,-apple-system,sans-serif">',
         f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
         f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="{BG}"/>']

    # y gridlines + labels
    v = 0
    while v <= ymax:
        yy = Y(v)
        e.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{L+pw}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        e.append(f'<text x="{L-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="{MUT}">{v}</text>')
        v += 100

    # x date ticks (every 2 days)
    day = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= t1 + datetime.timedelta(days=1):
        xx = X(day)
        if L - 1 <= xx <= L + pw + 1:
            e.append(f'<line x1="{xx:.1f}" y1="{T}" x2="{xx:.1f}" y2="{T+ph}" stroke="{GRID}" stroke-width="1"/>')
            e.append(f'<text x="{xx:.1f}" y="{T+ph+18:.1f}" text-anchor="middle" font-size="11" fill="{MUT}">'
                     f'{day.strftime("%b ") + str(day.day)}</text>')
        day += datetime.timedelta(days=2)

    # all results = faint dots
    for dt, val, verif, ag in pts:
        if ag != our_agent:
            e.append(f'<circle cx="{X(dt):.1f}" cy="{Y(val):.1f}" r="2.3" fill="{GRAY}" opacity="0.5"/>')
    # verified (not ours) = navy diamonds
    for dt, val, verif, ag in pts:
        if verif == "valid" and ag != our_agent:
            e.append(_diamond(X(dt), Y(val), 3.6, NAVY, opacity=0.8))

    # frontiers
    raw = frontier(lambda _v: True)
    ver = frontier(lambda v: v == "valid")
    e.append(f'<path d="{step_path(raw)}" fill="none" stroke="{NAVY2}" stroke-width="1.6" '
             f'stroke-dasharray="5 3" opacity="0.75"/>')
    e.append(f'<path d="{step_path(ver)}" fill="none" stroke="{NAVY}" stroke-width="2.6"/>')

    # our results = gold diamonds (drawn last, on top)
    ours = [(dt, val) for dt, val, verif, ag in pts if ag == our_agent]
    for dt, val in ours:
        e.append(_diamond(X(dt), Y(val), 6.5, GOLD, stroke="#ffffff", sw=1.4))

    # end labels (stagger to avoid overlap)
    if raw:
        ry = Y(raw[-1][1])
        e.append(f'<text x="{L+pw+10}" y="{ry-2:.1f}" font-size="11.5" fill="{NAVY2}">'
                 f'– – raw best {raw[-1][1]:.1f}</text>')
        e.append(f'<text x="{L+pw+10}" y="{ry+12:.1f}" font-size="10" fill="{MUT}">unverified (pending)</text>')
    if ver:
        vy = Y(ver[-1][1])
        top_is_ours = ours and max(o[1] for o in ours) >= ver[-1][1] - 1e-6
        e.append(_diamond(X(ver[-1][0]), vy, 6.5, GOLD if top_is_ours else NAVY, stroke="#ffffff", sw=1.4))
        e.append(f'<text x="{L+pw+10}" y="{vy+28:.1f}" font-size="12.5" fill="{NAVY}" font-weight="bold">'
                 f'◆ verified {ver[-1][1]:.1f}</text>')
        lab = f"{our_agent} · #1 verified" if top_is_ours else "top verified"
        e.append(f'<text x="{L+pw+10}" y="{vy+44:.1f}" font-size="10.5" fill="{TEXT}">{lab}</text>')

    # title + legend
    e.append(f'<text x="{L}" y="24" font-size="13" font-weight="bold" fill="{TEXT}" '
             f'letter-spacing="1.2">SCORE EVOLUTION — gemma-challenge (tok/s)</text>')
    lx = L + 392
    e.append(f'<circle cx="{lx}" cy="20" r="3" fill="{GRAY}"/>')
    e.append(f'<text x="{lx+8}" y="24" font-size="10.5" fill="{MUT}">all results</text>')
    e.append(_diamond(lx + 92, 20, 3.6, NAVY))
    e.append(f'<text x="{lx+100}" y="24" font-size="10.5" fill="{MUT}">verified</text>')
    e.append(_diamond(lx + 168, 20, 4.5, GOLD, stroke="#fff", sw=1))
    e.append(f'<text x="{lx+177}" y="24" font-size="10.5" fill="{MUT}">mikasa-inbound (us)</text>')

    e.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(e))
    return True
