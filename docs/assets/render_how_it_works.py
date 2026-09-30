"""How aelfrice works, from an empty context window through one turn.

Renders docs/assets/how-it-works-light.png and how-it-works-dark.png.

Color does one job: it tells the three paths apart.
The palette is three slots of a validated categorical palette, checked
all-pairs for color-vision deficiency in both modes; everything else is
neutral ink, and every colored mark carries a text label, so color is
never the only cue.

  violet   retrieval: on every prompt, and on every search the agent runs,
           a hook reads the store and adds a block
  magenta  lock: a typed /aelf:lock writes a rule that comes back every time
  green    capture: each turn is logged, and your sentences are ingested
           as beliefs linked by DERIVED_FROM edges

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
    heading(12.75, "2   Each turn: hooks run before the model reads you, and after it replies")
    row = [
        (0.4, "You type", "a prompt, or /aelf:lock <rule>"),
        (4.4, "Prompt hook", "UserPromptSubmit"),
        (8.4, "The model replies", "its searches fire the search hook"),
        (12.2, "Stop hook", "runs after each reply"),
    ]
    for x, title, sub in row:
        panel(x, 11.6, 3.4, 0.8)
        text(x + 0.25, 12.0 + (0.17 if sub else 0), title, 11.5,
             weight="bold")
        if sub:
            text(x + 0.25, 11.83, sub, 9.5, color=t["ink2"])
    for x in (3.85, 7.85, 11.85):
        arrow((x, 12.0), (x + 0.5, 12.0), t["ink2"])

    col = [0.4, 5.75, 11.1]
    width = 4.5
    heads = [
        (t["violet"], "Every prompt and search: retrieve"),
        (t["magenta"], "A typed /aelf:lock: store a rule"),
        (t["green"], "After each reply: capture"),
    ]
    # Retrieve and lock leave the prompt hook; capture leaves the Stop hook.
    starts = [(6.1, 11.55), (6.1, 11.55), (13.9, 11.55)]
    for x, (color, head), p0 in zip(col, heads, starts):
        arrow(p0, (x + width / 2, 11.05), color)
        text(x, 10.8, head, 11.5, weight="bold")

    # Retrieval path.
    step(col[0], 9.65, width, 0.8, t["violet"], "Search stack",
         "locks always · entity index · BM25 full text")
    step(col[0], 8.5, width, 0.8, t["violet"], "Ranking engine",
         "locks, then entity matches, then BM25 × confidence")
    arrow((col[0] + width / 2, 9.6), (col[0] + width / 2, 9.35), t["violet"])
    text(col[0], 8.2, "Added to what the model reads (real hook output):", 9.5,
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
         "your prompt, then the reply")
    step(col[2], 8.5, width, 0.8, t["green"], "Ingested as beliefs",
         "every 12 turns and at compaction; replies skipped")
    arrow((col[2] + width / 2, 9.6), (col[2] + width / 2, 9.35), t["green"])
    gx, gy = col[2], 5.6
    panel(gx, gy, width, 2.55, edge=t["hair"])
    text(gx + 0.15, gy + 2.3, "Each belief DERIVED_FROM the one before (illustrative)", 9,
         color=t["muted"])
    nodes = {
        "a": (gx + 1.05, gy + 1.55, "turn 14: release from\nthe publish script"),
        "b": (gx + 3.45, gy + 1.55, "turn 14: run the\nfull suite first"),
        "c": (gx + 3.45, gy + 0.5, "turn 15: staging\ndeploys from release"),
        "d": (gx + 1.05, gy + 0.5, "turn 16: keep\nmain protected"),
    }
    for nx, ny, label in nodes.values():
        ax.add_patch(FancyBboxPatch(
            (nx - 0.85, ny - 0.26), 1.7, 0.52,
            boxstyle="round,pad=0,rounding_size=0.1",
            facecolor=t["surface"], edgecolor=t["green"], lw=1.6, zorder=4))
        text(nx, ny, label, 7.8, ha="center")
    # Every edge is DERIVED_FROM, which the panel title names, so the
    # arrows carry no labels: later belief -> the one before it.
    edges = [
        (("b", "left"), ("a", "right")),
        (("c", "top"), ("b", "bottom")),
        (("d", "right"), ("c", "left")),
    ]

    def anchor(key, side):
        nx, ny, _ = nodes[key]
        return {"right": (nx + 0.87, ny), "left": (nx - 0.87, ny),
                "top": (nx, ny + 0.28), "bottom": (nx, ny - 0.28)}[side]

    for (a, sa), (b, sb) in edges:
        ax.add_patch(FancyArrowPatch(
            anchor(a, sa), anchor(b, sb), arrowstyle="-|>", mutation_scale=10,
            color=t["ink2"], lw=1.2, zorder=3))

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
        (col[0], t["violet"], "read on every prompt and search", True),
        (col[1], t["magenta"], "locks written", False),
        (col[2], t["green"], "beliefs written", False),
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
