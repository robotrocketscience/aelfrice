# Eval fixture corpus — captured vs synthetic policy

**Status:** decided.
**Closes:** [#142](https://github.com/robotrocketscience/aelfrice/issues/142).
**Blocks:** [#136](https://github.com/robotrocketscience/aelfrice/issues/136) (eval-harness scaffolding), [#141](https://github.com/robotrocketscience/aelfrice/issues/141) (rebuilder trigger modes — threshold-default calibration).
**Spec context:** [`docs/design/context_rebuilder.md`](context_rebuilder.md), [`benchmarks/context-rebuilder/README.md`](../benchmarks/context-rebuilder/README.md), [`docs/design/transcript_ingest.md § PII scrubbing`](transcript_ingest.md), [`docs/user/PRIVACY.md § INEDIBLE`](../user/PRIVACY.md).

This memo decides what kind of fixture transcripts the v1.4.0 context-rebuilder eval harness reads, where they live, and what the public CI guarantees against. It is a decision memo. There is no code change in the issue that closes — the contract is what subsequent fixture-producing PRs honour.

## TL;DR

1. **Boundary policy: hybrid, with the public/private split aligned to the existing two-repo separation.** Synthetic, generator-built fixtures are committed to the public repo at `benchmarks/context-rebuilder/fixtures/`. Captured real-session transcripts (any `turns.jsonl` produced from an actual working session) live in `~/projects/aelfrice-lab` and are never pushed to GitHub. The boundary is the directory, not a filter.
2. **Sanitization (defensive, not load-bearing):** if a captured fixture is ever published — by mistake or by deliberate one-off contribution — it must be redacted against the same identifier-shaped patterns the v1.2.0 triple extractor already recognises, plus path tokens and branch names. This is contingency policy; the structural property is that captured fixtures don't go to GitHub.
3. **Reproducibility: synthetic is canonical.** CI runs against the synthetic fixture set. The headline continuation-fidelity number from `docs/design/context_rebuilder.md § Headline metric` is computed on synthetic. Captured corpus is **offline-only** calibration; it never gates a release.
4. **Calibration for #141 threshold mode:** the v1.4.0 ship default for the `threshold` trigger mode is calibrated from **synthetic alone**. Captured corpus calibration is optional and v1.5-grade; if it produces a meaningfully different threshold, the v1.5.x release retunes. v1.4.0 does not block on captured-corpus calibration.

## Why hybrid, not "captured-only" or "synthetic-only"

The issue framed three options. Walking through why hybrid wins:

**Option 1 — captured real transcripts only.** Captured `turns.jsonl` files contain user prose, project paths, branch names, internal identifiers, occasionally inferred beliefs about the user's intent or the user's project. The two-repo separation policy in `~/.claude/CLAUDE.md § Two-repo workflow` exists precisely so this content does not land on GitHub. Putting captured transcripts into the public repo would punch a hole in that property — and the hole would be the eval-harness directory, which is high-traffic. Rejected.

**Option 2 — synthetic transcripts only.** Tractable, reproducible, no boundary issue. The cost is that synthetic transcripts under-represent the long-tail messiness of real sessions: ambiguous references, implicit topic switches, partial plans, the agent re-deriving state from prior turns. That long tail is exactly where the rebuilder's continuation-fidelity claim is most fragile. Pure-synthetic ships, but loses calibration signal. Rejected as the *whole* policy; kept as the public-repo half.

**Option 3 — hybrid (chosen).** Synthetic carries CI and the headline number. Captured corpus, held lab-side, calibrates the long tail offline when needed. The split mirrors the rest of the two-repo arrangement: the public repo is what we publish and reproduce; the lab repo is research workspace.

## Decision 1 — Boundary policy

### Public repo (`~/projects/aelfrice`)

- `benchmarks/context-rebuilder/fixtures/` is the synthetic fixture directory. **Tracked in git.** Committed alongside the harness code that consumes it. Reproducibility-friendly (anyone with the repo can run `uv run pytest -q` or the equivalent benchmark runner and get the same numbers).
- Fixtures here are produced by a deterministic generator (seedable; the generator script lives next to the fixtures so the corpus is regenerable on demand). The generator emits synthetic `turns.jsonl` records that exercise the rebuilder's algorithmic surface: triple-extractable prose, multi-turn topic threads, locked-belief references, mid-session topic switches, "long-tail" recall questions.
- `benchmarks/context-rebuilder/eval_corpus/` (the existing gitignored captured-transcript directory referenced in `benchmarks/context-rebuilder/README.md`) is **retained as the local-developer escape hatch** for ad-hoc captured fixtures. It stays gitignored. It is never the source of truth for the public CI run. The new `fixtures/` directory and the existing `eval_corpus/` directory coexist by purpose: `fixtures/` is committed-and-reproduced, `eval_corpus/` is local-only.

### Lab repo (`~/projects/aelfrice-lab`)

- Captured `turns.jsonl` files held under a lab-side path. Naming and layout are a lab-side concern; the only public-side requirement is that these files do not appear in the public working tree.
- Per `~/.claude/CLAUDE.md § Two-repo workflow`, the lab repo's `.githooks/pre-push` blocks any remote URL containing `github.com`. That is the structural enforcement for "captured corpus stays off GitHub". This memo does not invent a new mechanism; it inherits the existing one.

### What this means for #136

When #136 stands up the eval harness, its loader reads from `benchmarks/context-rebuilder/fixtures/` for the public/CI path. If a developer wants to additionally point the harness at `eval_corpus/` for local calibration, that's a CLI flag on the harness, not a default. The default — and what CI runs — is synthetic.

## Decision 2 — Sanitization (contingency policy)

The structural rule is: captured fixtures do not go to GitHub. Sanitization is the **belt** to that **suspenders**: if a captured fixture is ever published — for example, a one-off public reproduction case for a bug report, or a user contributing a corpus snippet — these are the redactions that must run first.

The patterns are intentionally aligned with surfaces aelfrice already recognises, so the redaction tooling can reuse the existing regex bank rather than inventing a parallel one.

### What must be redacted

1. **Path tokens.** Absolute filesystem paths (`/Users/<name>/...`, `/home/<name>/...`, `C:\Users\<name>\...`), repo-root-prefixed paths that leak the user's directory structure, hostnames in URLs that point at private infra. Replace with `<path>`.
2. **Branch names.** Branch identifiers from the captured session that encode private project codenames or organisational naming conventions. Replace with `<branch>`. Public conventional-prefix names (`main`, `feat/foo`, `release/v1.4.0`) may be retained at the contributor's discretion since they leak nothing project-specific.
3. **Identifier-like strings.** This is where the triple extractor's regex bank carries weight. The v1.2.0 extractor at [`src/aelfrice/triple_extractor.py`](../src/aelfrice/triple_extractor.py) defines `_TOKEN = r"[A-Za-z][\w-]*"` (a word-class identifier with dashes/underscores) and `_NP` (a noun phrase of up to five such tokens). The same `_TOKEN` regex is what surfaces the identifier-shaped strings most likely to be project-specific names: `session_id`, `aelf-hook`, `RobotRocketScience-internal-tool`. The redaction tool runs the same `_TOKEN` regex over the captured transcript and flags any token that does not appear in a public allowlist (English stop words, the public aelfrice surface, common identifiers like `session_id` that appear in the public docs). Each flagged token is shown to the contributor for accept/redact. **The point of reusing the extractor's regex is that the redactor's "what looks like an identifier" definition matches what the extractor would have indexed** — so we redact at the same granularity at which we'd otherwise have leaked.
4. **Triple-extracted entity surfaces.** Run `extract_triples()` over the candidate transcript before publishing. The subjects and objects of every emitted triple are exactly the noun phrases that would land as belief content if the transcript were ingested. Each one gets a contributor-facing review.
5. **Inferred beliefs.** Captured transcripts may contain assistant turns that articulate inferred beliefs about the user (`<aelfrice-memory>` blocks, `<aelfrice-baseline>` blocks). The redaction tool strips those blocks entirely before publishing.

### What is not in scope here

- Email addresses, API keys, credentials, IP addresses. The lab repo's existing gitleaks-based pre-commit pipeline (`scripts/check-commit-msg.py` + `.githooks/pre-push`) already catches these. The fixture-publishing path inherits whatever scanning the lab repo runs. This memo does not duplicate that policy.
- Live retrieval of captured fixtures by the public harness. The harness in `benchmarks/context-rebuilder/` does not know about the lab path; it cannot accidentally read it.

### When does the sanitization rule fire?

It fires only on the deliberate, one-off path of contributing a captured snippet to the public repo. The default expectation is: this rarely happens. The paragraph exists so when it does happen, the redaction step is documented and consistent rather than ad-hoc.

## Decision 3 — Reproducibility (synthetic is canonical)

- **CI runs synthetic.** The eval-harness CI invocation (whenever #138/#136 wire it into `.github/workflows/`) reads from `benchmarks/context-rebuilder/fixtures/` only. No path inside CI knows about `eval_corpus/` or the lab repo.
- **The headline number is synthetic.** When v1.4.0 publishes a continuation-fidelity number against the regression band in `docs/design/context_rebuilder.md § Headline metric`, that number is the synthetic-corpus number. The release notes say so explicitly. If captured-corpus fidelity differs (and it likely will), that's a calibration footnote, not the headline.
- **Captured corpus is offline-only.** A developer with access to the lab repo can point the harness at lab-side captured transcripts to sanity-check whether synthetic-derived defaults transfer. The output of those runs is a calibration artefact, not a release artefact, and lives in the lab repo.

The regression-band guarantee — "v1.4.0 must hit ≥80% continuation fidelity at ≤30% of full-replay token cost" — is enforced against synthetic. That is what "reproducible on a fresh clone with `uv sync`" means. A user cloning the public repo can reproduce every published number; they cannot reproduce calibration runs that touched private corpus, and the project does not claim they should be able to.

## Decision 4 — #141 threshold-mode calibration

Issue #141 requires that the threshold-trigger default for the v1.4.0 rebuilder be **derived from eval-harness data, not hardcoded**. The question this memo answers: what corpus produces that derivation?

**v1.4.0: synthetic alone.** The threshold default for the v1.4.0 ship is calibrated from the synthetic fixture set. The reasoning:

- The synthetic generator can be tuned to span the relevant axes (task type, session length, locked-belief density, topic-switch frequency). The corpus that calibrates the threshold can be made representative-by-construction.
- Locking the v1.4.0 ship behind captured-corpus calibration would block the release on a private-corpus access requirement, which is contrary to the reproducibility property in Decision 3.
- The eval harness is built before the rebuilder (per #136). The synthetic corpus is what the harness has on day one.

**v1.5 (optional): captured-corpus retune.** If, after v1.4.0 ships, captured-corpus calibration runs in the lab show the synthetic-derived threshold is materially miscalibrated against real sessions (defined: continuation fidelity drops by more than the regression band's slack on captured corpus), v1.5.x retunes. The retune does not require publishing the captured corpus — only the resulting threshold value, with a footnote that the calibration evidence lives lab-side.

**v1.5 (also optional): a published "real-shaped synthetic" corpus.** If captured-corpus runs reveal long-tail patterns the synthetic generator misses, a follow-up PR can teach the generator those patterns (without copying real session content) and republish a richer synthetic corpus. That keeps the public reproducibility property intact while narrowing the synthetic-vs-captured gap.

## Open follow-ups

These are not blocking #142 closing; they're tracked here so the next person picking up the harness work knows where to look:

- **Synthetic generator script.** Lives in #136's scope. Must be deterministic (single seed input → identical fixture set). Should produce enough variety to span at least the four task types in `docs/design/context_rebuilder.md` (debugging, planning, code-review, exploratory).
- **Redaction tool implementation.** Not on the v1.4.0 critical path. If/when the contingency-publishing case actually arises, build the tool then; the spec above is what it must implement. Until then, the `~/projects/aelfrice-lab/.githooks/pre-push` block is sufficient.
- **CI wiring.** When #138 (fidelity scorer) and #136 (harness skeleton) both land, a `.github/workflows/eval.yml` step runs the synthetic-corpus regression at PR time. Out of scope for this memo.

## Why this is a decision memo, not a code change

The contract is what fixture-producing PRs (starting with #136) honour. There is no current PR producing fixtures; this memo lands ahead of those PRs so the policy is in place when they're written. Atomic commits in this PR: (a) this memo; (b) the ROADMAP cross-reference. No tests; the green CI on this PR just confirms nothing else broke.
