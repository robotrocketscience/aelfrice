"""Illustrative schematic of aelfrice's retrieval lanes activating over a
belief graph. Layer = color; distance from center = graph-walk depth.

Lanes mirror the real design (README "How it works"); for legibility the
figure draws the lanes that fan out visibly from the query:
  L0    locked       - always returned, pinned at the query
  L1    FTS5 / BM25  - keyword seeds
  L3    graph walk   - typed-edge BFS from the seed set (opt-in; depth = hops)
  HRR   structural   - Plate-FFT bridge to vocab-gap matches keyword missed
                       (a separate retrieve_v2 lane, not on the default path)
The L2.5 entity-index lane is omitted from the drawing for legibility.
Not a trace of any real store. Deterministic (fixed seed)."""
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

rng = np.random.default_rng(11)

BG = "#070912"
C_QUERY = "#ffffff"
C_L0 = "#ffd24d"     # locked (always returned)
C_L1 = "#4d9bff"     # keyword / BM25 seeds
C_L3 = "#3ddc97"     # graph walk (typed-edge BFS, opt-in)
C_HRR = "#b07cff"    # structural HRR bridge (retrieve_v2 lane)
C_DORM = "#222838"


def lighten(hexc, f=0.55):
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    c = c + (1 - c) * f
    return tuple(c)


def lightning(p0, p1, segs=14, jit=0.22):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0, 1, segs)
    pts = (1 - t)[:, None] * p0 + t[:, None] * p1
    d = p1 - p0
    L = np.hypot(*d) + 1e-9
    n = np.array([-d[1], d[0]]) / L
    disp = rng.normal(0, jit, segs)
    disp[0] = disp[-1] = 0
    disp = np.convolve(disp, [0.25, 0.5, 0.25], mode="same")
    return pts + n * (disp * L)[:, None]


def streak(p0, p1, color, intensity=1.0, segs=14, jit=0.22):
    pts = lightning(p0, p1, segs, jit)
    x, y = pts[:, 0], pts[:, 1]
    for lw, a in [(7.0, 0.035), (4.0, 0.07), (2.3, 0.16)]:
        ax.plot(x, y, color=color, lw=lw * intensity, alpha=a, zorder=3,
                solid_capstyle="round")
    ax.plot(x, y, color=lighten(color, 0.6), lw=0.9, alpha=0.95, zorder=4,
            solid_capstyle="round")
    return pts[-1]


def glow(x, y, color, size):
    for r, a in [(size * 6, 0.05), (size * 3.2, 0.09), (size * 1.8, 0.18)]:
        ax.scatter([x], [y], s=r, color=color, alpha=a, zorder=5, edgecolors="none")
    ax.scatter([x], [y], s=size, color=color, zorder=6,
               edgecolors=lighten(color, 0.7), linewidths=0.5)


fig, ax = plt.subplots(figsize=(11, 8), dpi=140)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

P = rng.uniform(-5.2, 5.2, size=(440, 2))
P = P[np.hypot(P[:, 0], P[:, 1]) > 0.6]
for i in range(len(P)):
    d = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
    for j in np.argsort(d)[1:4]:
        if j > i:
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color="#11151f", lw=0.5, zorder=1)
ax.scatter(P[:, 0], P[:, 1], s=10, color=C_DORM, zorder=2)

center = np.array([0.0, 0.0])

for k in range(3):
    a = np.pi / 2 + (k - 1) * 0.6
    p = np.array([0.85 * np.cos(a), 0.85 * np.sin(a)])
    streak(center, p, C_L0, intensity=1.0, jit=0.06)
    glow(*p, C_L0, 40)

seeds = []
sang = np.linspace(0, 2 * np.pi, 7, endpoint=False) + 0.3 + rng.uniform(-0.08, 0.08, 7)
for a in sang:
    r = rng.uniform(1.3, 1.8)
    p = np.array([r * np.cos(a), r * np.sin(a)])
    end = streak(center, p, C_L1, intensity=1.15, jit=0.1)
    glow(*end, C_L1, 44)
    seeds.append((end, a))

for base, a in seeds:
    frontier = [(base, a)]
    for hop, (rad, inten) in enumerate([(2.6, 0.9), (3.6, 0.72), (4.5, 0.55)]):
        nxt = []
        for bp, ba in frontier:
            for _ in range(int(rng.integers(1, 3 if hop == 0 else 2) + 1)):
                aa = ba + rng.uniform(-0.45, 0.45)
                p = np.array([rad * np.cos(aa), rad * np.sin(aa)])
                end = streak(bp, p, C_L3, intensity=inten, jit=0.18 + 0.05 * hop)
                glow(*end, C_L3, max(14, 34 - 9 * hop))
                nxt.append((end, aa))
        frontier = nxt[:2]

for a in np.linspace(0, 2 * np.pi, 5, endpoint=False) + 0.9:
    r = rng.uniform(3.8, 4.9)
    p = np.array([r * np.cos(a), r * np.sin(a)])
    end = streak(center, p, C_HRR, intensity=0.8, segs=18, jit=0.33)
    glow(*end, C_HRR, 26)

glow(*center, C_QUERY, 95)

ax.set_xlim(-5.3, 5.3)
ax.set_ylim(-4.0, 4.0)
ax.set_aspect("equal")
ax.axis("off")

handles = [
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=C_QUERY, markeredgecolor="none", markersize=9, label="query"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=C_L0, markeredgecolor="none", markersize=9, label="L0  locked (always returned)"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=C_L1, markeredgecolor="none", markersize=9, label="L1  FTS5 keyword / BM25"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=C_L3, markeredgecolor="none", markersize=9, label="L3  graph walk (typed-edge BFS, opt-in)"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=C_HRR, markeredgecolor="none", markersize=9, label="structural HRR (retrieve_v2 lane)"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=C_DORM, markeredgecolor="none", markersize=9, label="dormant belief"),
]
ax.legend(handles=handles, loc="upper left", frameon=False, labelcolor="#c7ccd6",
          fontsize=8.5, bbox_to_anchor=(-0.02, 1.0))
ax.set_title("aelfrice retrieval lanes activating over a belief graph (illustrative)",
             color="#e6e9f0", fontsize=12, pad=12)
plt.tight_layout()
_out = pathlib.Path(__file__).resolve().parent / "retrieval-lanes.png"
plt.savefig(_out, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(f"wrote {_out}")
