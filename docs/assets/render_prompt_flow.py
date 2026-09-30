"""Schematic of one aelfrice prompt: what happens between typing a message
and the model reading it. Same palette and glow as render_retrieval_lanes.py.

  1. You type a prompt.
  2. The UserPromptSubmit hook runs before the model sees it.
  3. The hook retrieves from the local SQLite store: locked rules (L0), the
     entity index (L2.5), and BM25 full-text search (L1).
  4. It prepends an <aelfrice-memory> block to the prompt.
  5. The model reads block and prompt as one message.

Illustrative, not a trace. Deterministic: no randomness.
Render with: uv run --with matplotlib python docs/assets/render_prompt_flow.py
"""
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BG = "#070912"
C_TEXT = "#e8ecf6"
C_DIM = "#8a93ab"
C_PROMPT = "#ffffff"
C_HOOK = "#b07cff"
C_L0 = "#ffd24d"
C_L25 = "#ff8a5c"
C_L1 = "#4d9bff"
C_MODEL = "#3ddc97"


def lighten(hexc: str, f: float = 0.55) -> tuple[float, float, float]:
    rgb = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(c + (1 - c) * f for c in rgb)  # type: ignore[return-value]


def box(ax, x, y, w, h, color, title, sub=""):
    for pad, alpha in [(0.06, 0.05), (0.035, 0.09), (0.015, 0.16)]:
        ax.add_patch(FancyBboxPatch(
            (x - pad, y - pad), w + 2 * pad, h + 2 * pad,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color, edgecolor="none", alpha=alpha, zorder=2))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=BG, edgecolor=lighten(color, 0.3), lw=1.6, zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.13 if sub else 0), title,
            ha="center", va="center", color=C_TEXT, fontsize=13,
            fontweight="bold", zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.19, sub, ha="center", va="center",
                color=C_DIM, fontsize=9.5, zorder=4)


def arrow(ax, p0, p1, color, rad=0.0):
    for lw, alpha in [(7, 0.06), (3.5, 0.14)]:
        ax.add_patch(FancyArrowPatch(
            p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
            color=color, lw=lw, alpha=alpha, zorder=1))
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
        mutation_scale=14, color=lighten(color, 0.4), lw=1.4, zorder=5))


fig, ax = plt.subplots(figsize=(12.4, 5.4), dpi=160)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12.4)
ax.set_ylim(-0.2, 5.2)
ax.axis("off")

# The main path, left to right.
box(ax, 0.3, 3.3, 2.1, 1.0, C_PROMPT, "Your prompt", '"push the release"')
box(ax, 3.2, 3.3, 2.4, 1.0, C_HOOK, "aelfrice hook", "runs before the model")
box(ax, 6.4, 3.3, 3.0, 1.0, C_HOOK, "<aelfrice-memory>", "matched beliefs + prompt")
box(ax, 10.2, 3.3, 1.95, 1.0, C_MODEL, "Model", "reads one message")
arrow(ax, (2.45, 3.8), (3.15, 3.8), C_PROMPT)
arrow(ax, (5.65, 3.8), (6.35, 3.8), C_HOOK)
arrow(ax, (9.45, 3.8), (10.15, 3.8), C_MODEL)

# The local store the hook reads, below it.
ax.add_patch(FancyBboxPatch(
    (1.2, -0.05), 6.4, 2.25, boxstyle="round,pad=0.02,rounding_size=0.12",
    facecolor="none", edgecolor=C_DIM, lw=1.0, ls=(0, (4, 4)), zorder=1))
ax.text(4.4, 0.13, "local SQLite store — no network", color=C_DIM,
        fontsize=9.5, ha="center", va="center", zorder=4)
box(ax, 1.5, 0.5, 1.8, 1.2, C_L0, "L0 locked", "always returned")
box(ax, 3.55, 0.5, 1.8, 1.2, C_L25, "L2.5 entities", "exact + stem")
box(ax, 5.6, 0.5, 1.8, 1.2, C_L1, "L1 BM25", "full-text search")
for x, color in [(2.4, C_L0), (4.45, C_L25), (6.5, C_L1)]:
    arrow(ax, (x, 1.75), (4.4, 3.25), color, rad=0.0)

out = pathlib.Path(__file__).with_name("prompt-flow.png")
fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
print(out)
