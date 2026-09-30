"""A small belief graph: beliefs colored by type, linked by labeled edges.

Renders docs/assets/belief-graph-light.png and belief-graph-dark.png.

Color does one job: it names a belief's type. Four types get four slots
of a validated categorical palette (blue, yellow, magenta, green), checked
all-pairs for color-vision deficiency in both modes. No five-slot set
passes that check, so the fifth type, speculative, is drawn as a hollow
outline instead of a fill: it is a phantom, a belief not yet trusted.
Every node also names its type in text, so color is never the only cue.

The beliefs and edges are illustrative, not a trace of a real store.
Each edge type shown has a writer in the code; the README's edge table
says which writers are on by default. Deterministic: no randomness.

Render with: uv run --with matplotlib python docs/assets/render_belief_graph.py
"""
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "rule": "#c3c2b7",
        "factual": "#2a78d6", "preference": "#eda100",
        "requirement": "#e87ba4", "correction": "#008300",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "rule": "#383835",
        "factual": "#3987e5", "preference": "#c98500",
        "requirement": "#d55181", "correction": "#008300",
    },
}
SANS = ["Helvetica Neue", "Arial", "DejaVu Sans"]

# key: (x, y, type, text, locked)
NODES = {
    "lock": (2.2, 6.2, "factual", "never push directly to main", True),
    "pub": (7.8, 6.2, "factual", "the publish script runs\nthe release checks", False),
    "chk": (13.4, 6.2, "requirement", "the release checks\nmust include pyright", False),
    "wonder": (2.2, 3.4, "speculative", "cache the wheel build\nbetween releases?", False),
    "fix": (7.8, 3.4, "correction", "do not deploy staging from\nmain; use the release branch", False),
    "commit": (13.4, 3.4, "factual", "publish.sh runs pytest\nand pyright (commit)", False),
    "old": (7.8, 0.6, "factual", "staging deploys from main", False),
    "pref": (13.4, 0.6, "preference", "I prefer small,\natomic commits", False),
}
# (src, (side, dx), dst, (side, dx), label, label offset)
EDGES = [
    ("chk", ("left", 0), "pub", ("right", 0), "DERIVED_FROM", (0, 0.2)),
    ("commit", ("top", 0), "chk", ("bottom", 0), "SUPPORTS", (0.12, 0)),
    ("commit", ("left", 0), "pub", ("bottom", 1.0), "IMPLEMENTS", (0.25, 0.05)),
    ("wonder", ("right", 0), "pub", ("bottom", -1.0), "RELATES_TO", (-1.45, 0.05)),
    ("fix", ("top", 0), "pub", ("bottom", 0), "TEMPORAL_NEXT", (0.12, 0)),
    ("fix", ("bottom", 0), "old", ("top", 0), "SUPERSEDES", (0.12, 0)),
    ("pref", ("left", 0), "fix", ("bottom", 1.2), "TEMPORAL_NEXT", (0.55, -0.1)),
]
W_BOX, H_BOX = 3.6, 1.1


def render(mode: str) -> pathlib.Path:
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(16, 8.6), dpi=130)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.7, 8.2)
    ax.axis("off")

    def text(x, y, s, size=10, color=None, weight="normal", ha="center"):
        ax.text(x, y, s, fontsize=size, color=color or t["ink"], ha=ha,
                va="center", fontweight=weight, family=SANS, zorder=6)

    def anchor(key, spec):
        side, dx = spec
        x, y = NODES[key][:2]
        return {"left": (x - W_BOX / 2 - 0.05, y),
                "right": (x + W_BOX / 2 + 0.05, y),
                "top": (x + dx, y + H_BOX / 2 + 0.05),
                "bottom": (x + dx, y - H_BOX / 2 - 0.05)}[side]

    text(0.4, 7.8, "A few beliefs, colored by type and linked by typed edges "
         "(illustrative)", 14, weight="bold", ha="left")

    for x, y, kind, body, locked in NODES.values():
        hollow = kind == "speculative"
        color = t["ink2"] if hollow else t[kind]
        ax.add_patch(FancyBboxPatch(
            (x - W_BOX / 2, y - H_BOX / 2), W_BOX, H_BOX,
            boxstyle="round,pad=0,rounding_size=0.14",
            facecolor=t["surface"], edgecolor=color,
            lw=1.4 if hollow else 1.2, zorder=3))
        if not hollow:
            # A colored band on the left carries the type; text stays ink.
            ax.add_patch(FancyBboxPatch(
                (x - W_BOX / 2 + 0.08, y - H_BOX / 2 + 0.12), 0.14,
                H_BOX - 0.24, boxstyle="round,pad=0,rounding_size=0.05",
                facecolor=color, edgecolor="none", zorder=4))
        tag = kind + ("  ·  locked" if locked else "")
        text(x + 0.1, y + 0.3, tag, 8.5, color=t["ink2"])
        text(x + 0.1, y - 0.12, body, 9.5)

    for src, s_spec, dst, d_spec, label, (ox, oy) in EDGES:
        p0, p1 = anchor(src, s_spec), anchor(dst, d_spec)
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=12, color=t["ink2"],
            lw=1.3, zorder=2))
        text((p0[0] + p1[0]) / 2 + ox, (p0[1] + p1[1]) / 2 + oy, label, 8,
             color=t["muted"], ha="left" if ox else "center")

    text(0.4, -0.5, "Each arrow runs from an edge's source to its target. A "
         "hollow outline is a speculative (phantom) belief. Locked is a flag "
         "on a belief of any type, and a lock needs no edge: it is injected "
         "whether or not anything links to it.", 9, color=t["ink2"], ha="left")

    out = pathlib.Path(__file__).with_name(f"belief-graph-{mode}.png")
    fig.savefig(out, facecolor=t["surface"], bbox_inches="tight",
                pad_inches=0.2)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for m in THEMES:
        print(render(m))
