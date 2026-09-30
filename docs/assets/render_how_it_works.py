"""How aelfrice works, from an empty context window through one turn.

Renders docs/assets/how-it-works-light.png and how-it-works-dark.png.

Color does one job: it tells the three paths out of the prompt hook apart.
The palette is three slots of a validated categorical palette, checked
all-pairs for color-vision deficiency in both modes; everything else is
neutral ink, and every colored mark carries a text label, so color is
never the only cue.

  violet   retrieval: every prompt, the hook reads the store and adds a block
  magenta  lock: a typed /aelf:lock writes a rule that comes back every time
  green    capture: each turn is logged and ingested as typed beliefs

The quoted blocks are real hook output for a five-belief demo store (two
locks, three ordinary beliefs); the two ranked lists are real `aelf search`
orderings on that store. Deterministic: no randomness.

Render with: uv run --with matplotlib python docs/assets/render_how_it_works.py
"""
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "hair": "#e1e0d9", "rule": "#c3c2b7",
        "host": "#e1e0d9", "host2": "#ecebe6", "code_bg": "#f4f3ef",
        "violet": "#4a3aa7", "magenta": "#e87ba4", "green": "#008300",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "hair": "#2c2c2a", "rule": "#383835",
        "host": "#383835", "host2": "#2c2c2a", "code_bg": "#232322",
        "violet": "#9085e9", "magenta": "#d55181", "green": "#008300",
    },
}
SANS = ["system-ui", "-apple-system", "Segoe UI", "Helvetica Neue", "Arial",
        "DejaVu Sans"]
MONO = ["Menlo", "DejaVu Sans Mono", "monospace"]

BASELINE = [
    "<aelfrice-baseline>",
    "The memory store contents below are in two trust tiers. …",
    '<belief id="47265028cd9e4904" lock="user">never push directly to main; use scripts/publish.sh</belief>',
    '<belief id="ce301406f38fa7e0" lock="user">commits must be signed</belief>',
    "</aelfrice-baseline>",
]
MEMORY = [
    "<aelfrice-memory>",
    "The memory store contents below are in two",
    "trust tiers. …",
    '<belief id="ce301406f38fa7e0" lock="user">',
    "  commits must be signed</belief>",
    '<belief id="47265028cd9e4904" lock="user">',
    "  never push directly to main; …</belief>",
    '<belief id="c112fa18b21874bf" lock="none">',
    "  the publish script runs the release",
    "  checks before tagging</belief>",
    "…",
    "</aelfrice-memory>",
]
TURN1 = ("push the release", [
    ("lock", "commits must be signed"),
    ("lock", "never push directly to main; use scripts/publish.sh"),
    ("", "the publish script runs the release checks before tagging"),
    ("", "the release checks include the full test suite and pyright"),
    ("", "staging deploys from the release branch, not main"),
])
TURN2 = ("why did staging deploy from main", [
    ("lock", "commits must be signed"),
    ("lock", "never push directly to main; use scripts/publish.sh"),
    ("", "staging deploys from the release branch, not main"),
])


def render(mode: str) -> pathlib.Path:
    t = THEMES[mode]
    W, H = 16.0, 17.2
    fig, ax = plt.subplots(figsize=(W, H), dpi=130)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def text(x, y, s, size=11, color=None, weight="normal", ha="left",
             va="center", family=None):
        ax.text(x, y, s, fontsize=size, color=color or t["ink"],
                fontweight=weight, ha=ha, va=va,
                family=family or SANS, zorder=6)

    def panel(x, y, w, h, edge=None, fill=None, lw=1.0):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=fill or t["surface"], edgecolor=edge or t["rule"],
            lw=lw, zorder=2))

    def step(x, y, w, h, color, title, sub=""):
        # A thin colored rule on the left carries identity; text stays ink.
        panel(x, y, w, h)
        ax.add_patch(FancyBboxPatch(
            (x + 0.06, y + 0.1), 0.08, h - 0.2,
            boxstyle="round,pad=0,rounding_size=0.04",
            facecolor=color, edgecolor="none", zorder=3))
        text(x + 0.3, y + h / 2 + (0.16 if sub else 0), title, 11.5,
             weight="bold")
        if sub:
            text(x + 0.3, y + h / 2 - 0.18, sub, 9.5, color=t["ink2"])

    def arrow(p0, p1, color, rad=0.0):
        ax.add_patch(FancyArrowPatch(
            p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
            mutation_scale=13, color=color, lw=2.0, zorder=5))

    def code(x, y, w, lines, size=8.4, step_y=0.25):
        h = step_y * len(lines) + 0.22
        panel(x, y - h, w, h, edge=t["hair"], fill=t["code_bg"])
        for i, line in enumerate(lines):
            text(x + 0.14, y - 0.22 - step_y * i, line, size,
                 color=t["ink2"], family=MONO)
        return y - h

    def heading(y, s):
        text(0.4, y, s, 14, weight="bold")

    # 1. Session start: the context window before your first prompt.
    heading(16.75, "1   Session start: the context window, before you type")
    bar_y, bar_h = 15.75, 0.5
    for x, w, color, label in [
        (0.4, 5.6, t["host"], "Host system prompt + tool definitions"),
        (6.05, 1.6, t["host2"], "CLAUDE.md"),
        (7.7, 1.2, t["magenta"], "aelfrice locked rules"),
    ]:
        ax.add_patch(FancyBboxPatch(
            (x, bar_y), w, bar_h, boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=color, edgecolor="none", zorder=3))
        text(x + 0.05, bar_y - 0.28, label, 9.5, color=t["ink2"])
    ax.add_patch(FancyBboxPatch(
        (8.95, bar_y), 6.65, bar_h, boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=t["surface"], edgecolor=t["rule"], lw=1.0, zorder=3))
    text(12.27, bar_y + bar_h / 2, "free for the conversation", 10,
         color=t["muted"], ha="center")
    text(15.6, bar_y - 0.28, "widths are illustrative", 8.5,
         color=t["muted"], ha="right")
    text(0.4, 15.05, "What aelfrice adds at session start (real hook output):",
         10, color=t["ink2"])
    code(0.4, 14.85, 15.2, BASELINE)

    # 2. One turn: the prompt hook and its three paths.
    heading(12.75, "2   Each turn: the prompt hook runs before the model reads you")
    panel(0.4, 11.6, 3.4, 0.8)
    text(0.65, 12.17, "You type", 11.5, weight="bold")
    text(0.65, 11.83, "a prompt, or /aelf:lock <rule>", 9.5, color=t["ink2"])
    panel(4.6, 11.6, 3.4, 0.8)
    text(4.85, 12.17, "Prompt hook", 11.5, weight="bold")
    text(4.85, 11.83, "UserPromptSubmit", 9.5, color=t["ink2"])
    arrow((3.85, 12.0), (4.55, 12.0), t["ink2"])

    col = [0.4, 5.75, 11.1]
    width = 4.5
    heads = [
        (t["violet"], "Every prompt: retrieve"),
        (t["magenta"], "A typed /aelf:lock: store a rule"),
        (t["green"], "Every turn: capture"),
    ]
    rads = [0.0, 0.0, 0.0]
    for x, (color, head), rad in zip(col, heads, rads):
        arrow((6.3, 11.55), (x + width / 2, 11.05), color, rad=rad)
        text(x, 10.8, head, 11.5, weight="bold")

    # Retrieval path.
    step(col[0], 9.65, width, 0.8, t["violet"], "Search stack",
         "locks always · entity index · BM25 full text")
    step(col[0], 8.5, width, 0.8, t["violet"], "Ranking engine",
         "locks first, then relevance × confidence")
    arrow((col[0] + width / 2, 9.6), (col[0] + width / 2, 9.35), t["violet"])
    text(col[0], 8.2, "Added before your prompt (real hook output):", 9.5,
         color=t["ink2"])
    code(col[0], 8.02, width, MEMORY, size=7.6, step_y=0.22)

    # Lock path.
    step(col[1], 9.65, width, 0.8, t["magenta"], "The hook runs the lock",
         "/aelf:lock never push directly to main")
    step(col[1], 8.5, width, 0.8, t["magenta"], "Written as a locked belief",
         "pinned as ground truth")
    step(col[1], 7.35, width, 0.8, t["magenta"], "Always injected",
         "every session start and every prompt")
    arrow((col[1] + width / 2, 9.6), (col[1] + width / 2, 9.35), t["magenta"])
    arrow((col[1] + width / 2, 8.45), (col[1] + width / 2, 8.2), t["magenta"])

    # Capture path, ending in a small typed graph.
    step(col[2], 9.65, width, 0.8, t["green"], "Turn logged",
         "your prompt and the reply")
    step(col[2], 8.5, width, 0.8, t["green"], "Ingested as typed beliefs",
         "sentences become beliefs linked by typed edges")
    arrow((col[2] + width / 2, 9.6), (col[2] + width / 2, 9.35), t["green"])
    gx, gy = col[2], 5.6
    panel(gx, gy, width, 2.55, edge=t["hair"])
    text(gx + 0.15, gy + 2.3, "A small typed graph (illustrative)", 9,
         color=t["muted"])
    nodes = {
        "a": (gx + 1.0, gy + 1.55, "publish script\nruns the checks"),
        "b": (gx + 3.5, gy + 1.55, "checks include\nthe full suite"),
        "c": (gx + 1.0, gy + 0.5, "turn 14"),
        "d": (gx + 3.5, gy + 0.5, "staging deploys\nfrom release"),
    }
    for nx, ny, label in nodes.values():
        ax.add_patch(FancyBboxPatch(
            (nx - 0.72, ny - 0.26), 1.44, 0.52,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor=t["surface"], edgecolor=t["green"], lw=1.6, zorder=4))
        text(nx, ny, label, 7.8, ha="center")
    edges = [
        (("a", "right"), ("b", "left"), "RELATES_TO", (0, 0.14)),
        (("a", "bottom"), ("c", "top"), "DERIVED_FROM", (0.1, 0)),
        (("b", "bottom"), ("d", "top"), "RELATES_TO", (0.1, 0)),
    ]

    def anchor(key, side):
        nx, ny, _ = nodes[key]
        return {"right": (nx + 0.74, ny), "left": (nx - 0.74, ny),
                "top": (nx, ny + 0.28), "bottom": (nx, ny - 0.28)}[side]

    for (a, sa), (b, sb), label, (ox, oy) in edges:
        p0, p1 = anchor(a, sa), anchor(b, sb)
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=10, color=t["ink2"],
            lw=1.2, zorder=3))
        text((p0[0] + p1[0]) / 2 + ox, (p0[1] + p1[1]) / 2 + oy, label, 7.2,
             color=t["muted"], ha="left" if ox else "center")

    # 3. The ranking engine, turn by turn: real `aelf search` orderings.
    heading(4.75, "3   The ranking follows each prompt (real search order)")
    for x, (prompt, rows) in zip((0.4, 8.2), (TURN1, TURN2)):
        text(x, 4.3, f'Prompt: "{prompt}"', 10, weight="bold")
        y = 4.0
        for i, (tag, body) in enumerate(rows, 1):
            mark = "lock" if tag else "    "
            text(x + 0.1, y, f"{i}. {mark}  {body}", 8.6, color=t["ink2"],
                 family=MONO)
            y -= 0.27

    # 4. The store every path reads or writes.
    panel(0.4, 0.3, 15.2, 1.25)
    text(0.65, 1.2, "Local SQLite store: the belief graph", 12.5,
         weight="bold")
    text(0.65, 0.78, "beliefs with an origin and a Bayesian confidence "
         "(α, β), typed edges, and locks, in one file under your "
         "repository's .git directory", 9.5, color=t["ink2"])
    for x, color, label, up in [
        (col[0], t["violet"], "read on every prompt", True),
        (col[1], t["magenta"], "locks written", False),
        (col[2], t["green"], "turns written", False),
    ]:
        cx = x + width / 2
        p_top, p_bot = (cx, 2.15), (cx, 1.6)
        arrow(p_bot if up else p_top, p_top if up else p_bot, color)
        text(cx + 0.15, 1.88, label, 9, color=t["ink2"])

    out = pathlib.Path(__file__).with_name(f"how-it-works-{mode}.png")
    fig.savefig(out, facecolor=t["surface"], bbox_inches="tight",
                pad_inches=0.2)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for m in THEMES:
        print(render(m))
