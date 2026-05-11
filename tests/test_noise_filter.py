"""noise_filter: per-category and combined predicate contract.

Closes the v1.0.1 onboarding-noise gap. Atomic short tests; one
property each.
"""
from __future__ import annotations

import pytest

from aelfrice.noise_filter import (
    is_checklist_block,
    is_heading_block,
    is_license_boilerplate,
    is_noise,
    is_three_word_fragment,
    similarity_to_reference,
)


# --- Empty / whitespace -------------------------------------------------


def test_empty_string_is_noise() -> None:
    assert is_noise("") is True


def test_whitespace_only_is_noise() -> None:
    assert is_noise("   \n\n\t  ") is True


# --- Three-word fragments -----------------------------------------------


def test_one_word_is_fragment() -> None:
    assert is_three_word_fragment("INSTRUCTIONS") is True


def test_three_words_is_fragment() -> None:
    assert is_three_word_fragment("aelfrice is great") is True


def test_four_words_is_not_fragment() -> None:
    assert is_three_word_fragment("aelfrice is a memory") is False


def test_long_label_24_chars_but_two_words_is_fragment() -> None:
    """The existing _MIN_PARAGRAPH_CHARS = 24 lets long-but-empty labels
    through. The word check catches them."""
    assert is_three_word_fragment("INSTRUCTIONS_FOR_LATER:_BEGIN_AT_TOP") is True


# --- Heading blocks -----------------------------------------------------


def test_single_h1_is_heading_block() -> None:
    assert is_heading_block("# Project README") is True


def test_h2_with_content_is_heading_block() -> None:
    assert is_heading_block("## What this does") is True


def test_h6_is_heading_block() -> None:
    assert is_heading_block("###### Subsection") is True


def test_seven_hashes_is_not_heading() -> None:
    """Markdown spec: max 6 levels."""
    assert is_heading_block("####### Too deep") is False


def test_no_space_after_hash_is_not_heading() -> None:
    """`#tag` is not a heading."""
    assert is_heading_block("#projectname") is False


def test_multi_line_heading_run_is_heading_block() -> None:
    text = "# Section\n## Subsection\n### Detail"
    assert is_heading_block(text) is True


def test_heading_with_blank_line_between_is_heading_block() -> None:
    """Blank lines do not break the all-headings invariant."""
    text = "# Section\n\n## Subsection"
    assert is_heading_block(text) is True


def test_heading_followed_by_prose_is_not_heading_block() -> None:
    """A paragraph that *mixes* heading and prose is real content."""
    text = "# Section\nThis section explains the design rationale."
    assert is_heading_block(text) is False


def test_plain_prose_is_not_heading_block() -> None:
    assert is_heading_block("regular paragraph of text") is False


# --- Checklist blocks ---------------------------------------------------


def test_single_unchecked_item_is_checklist_block() -> None:
    assert is_checklist_block("- [ ] do thing") is True


def test_single_checked_item_is_checklist_block() -> None:
    assert is_checklist_block("- [x] thing done") is True


def test_capital_X_is_checklist_block() -> None:
    assert is_checklist_block("- [X] thing done") is True


def test_asterisk_marker_is_checklist_block() -> None:
    assert is_checklist_block("* [ ] alt marker") is True


def test_plus_marker_is_checklist_block() -> None:
    assert is_checklist_block("+ [ ] alt marker") is True


def test_indented_checklist_is_checklist_block() -> None:
    assert is_checklist_block("    - [ ] indented") is True


def test_multi_item_run_is_checklist_block() -> None:
    text = "- [ ] one\n- [x] two\n- [ ] three"
    assert is_checklist_block(text) is True


def test_checklist_mixed_with_prose_is_not_checklist_block() -> None:
    text = "Here are the tasks:\n- [ ] one\n- [ ] two"
    assert is_checklist_block(text) is False


def test_bare_dash_bullet_is_not_checklist() -> None:
    """`- item` is a regular bullet, not a task list."""
    assert is_checklist_block("- item one\n- item two") is False


def test_no_space_after_brackets_is_not_checklist() -> None:
    assert is_checklist_block("- [x]done") is False


# --- License boilerplate ------------------------------------------------


def test_copyright_c_is_license() -> None:
    assert is_license_boilerplate("Copyright (c) 2026 Example Author") is True


def test_copyright_unicode_c_is_license() -> None:
    assert is_license_boilerplate("Copyright © 2026 Example Author") is True


def test_mit_permission_clause_is_license() -> None:
    text = (
        "Permission is hereby granted, free of charge, to any person "
        "obtaining a copy of this software"
    )
    assert is_license_boilerplate(text) is True


def test_apache_clause_is_license() -> None:
    text = "Licensed under the Apache License, Version 2.0"
    assert is_license_boilerplate(text) is True


def test_bsd_redistribution_clause_is_license() -> None:
    text = "Redistribution and use in source and binary forms"
    assert is_license_boilerplate(text) is True


def test_gpl_banner_is_license() -> None:
    text = "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007"
    assert is_license_boilerplate(text) is True


def test_lgpl_banner_is_license() -> None:
    text = "GNU LESSER GENERAL PUBLIC LICENSE"
    assert is_license_boilerplate(text) is True


def test_bare_mit_license_line_is_license() -> None:
    assert is_license_boilerplate("MIT License") is True


def test_all_rights_reserved_is_license() -> None:
    text = "© 2026 ExampleCorp. All rights reserved."
    assert is_license_boilerplate(text) is True


def test_casual_mention_of_copyright_is_not_license() -> None:
    """Conservative: real prose mentioning copyright in passing should
    not match. None of the seven patterns picks up a bare word."""
    text = (
        "Copyright disputes are a frequent issue in this domain "
        "and need careful handling."
    )
    assert is_license_boilerplate(text) is False


def test_word_apache_in_prose_is_not_license() -> None:
    text = "We deploy on Apache HTTPD behind a load balancer."
    assert is_license_boilerplate(text) is False


# --- Combined is_noise --------------------------------------------------


def test_is_noise_true_on_short_fragment() -> None:
    assert is_noise("two words") is True


def test_is_noise_true_on_heading() -> None:
    assert is_noise("## Architecture overview") is True


def test_is_noise_true_on_checklist() -> None:
    assert is_noise("- [ ] task one\n- [ ] task two") is True


def test_is_noise_true_on_license() -> None:
    assert is_noise("Copyright (c) 2026 ExampleCorp. All rights reserved.") is True


def test_is_noise_false_on_real_belief() -> None:
    text = (
        "The default DB path is keyed by SHA256 of the working "
        "directory, which causes orphan stores on directory rename."
    )
    assert is_noise(text) is False


def test_is_noise_false_on_technical_paragraph() -> None:
    text = (
        "Beta-Bernoulli posterior mean is alpha over alpha plus beta. "
        "Apply feedback events update one of those two counters."
    )
    assert is_noise(text) is False


def test_is_noise_false_on_long_paragraph_with_copyright_word_in_middle() -> None:
    """Copyright-word in passing prose; only the patterns match."""
    text = (
        "We chose this path because copyright considerations made the "
        "alternative untenable for our deployment situation."
    )
    assert is_noise(text) is False


# --- similarity_to_reference: property tests ----------------------------


def test_similarity_identity(tmp_path: "pytest.TempPathFactory") -> None:  # type: ignore[type-arg]
    """sim(a, a) == 1.0 — a document compared to itself is fully similar."""
    ref = tmp_path / "ref.txt"
    text = "the belief graph stores weighted edges between project nodes"
    ref.write_text(text, encoding="utf-8")
    over, score, excerpt = similarity_to_reference(text, ref)
    assert score == pytest.approx(1.0)
    assert over is True


def test_similarity_symmetry(tmp_path: "pytest.TempPathFactory") -> None:  # type: ignore[type-arg]
    """sim(a, b) == sim(b, a) — Jaccard is symmetric."""
    a_text = "the belief graph stores weighted edges between project nodes"
    b_text = "weighted edges store belief scores between graph nodes here"
    ref_a = tmp_path / "a.txt"
    ref_b = tmp_path / "b.txt"
    ref_a.write_text(a_text, encoding="utf-8")
    ref_b.write_text(b_text, encoding="utf-8")
    _, score_ab, _ = similarity_to_reference(a_text, ref_b)
    _, score_ba, _ = similarity_to_reference(b_text, ref_a)
    assert score_ab == pytest.approx(score_ba)


def test_similarity_disjoint(tmp_path: "pytest.TempPathFactory") -> None:  # type: ignore[type-arg]
    """sim over no shared N-grams == 0.0."""
    ref = tmp_path / "ref.txt"
    ref.write_text(
        "alpha bravo charlie delta echo foxtrot golf hotel india",
        encoding="utf-8",
    )
    over, score, excerpt = similarity_to_reference(
        "one two three four five six seven eight nine", ref
    )
    assert score == pytest.approx(0.0)
    assert over is False
    assert excerpt is None


def test_similarity_threshold_fires(tmp_path: "pytest.TempPathFactory") -> None:  # type: ignore[type-arg]
    """A text above the threshold is flagged and an excerpt is returned."""
    ref = tmp_path / "ref.txt"
    body = "the belief graph stores weighted edges between nodes for facts"
    ref.write_text(body, encoding="utf-8")
    over, score, excerpt = similarity_to_reference(body, ref, threshold=0.5)
    assert over is True
    assert score >= 0.5
    assert excerpt is not None
    assert len(excerpt) > 0


def test_similarity_threshold_not_fired(tmp_path: "pytest.TempPathFactory") -> None:  # type: ignore[type-arg]
    """A text below the threshold is not flagged."""
    ref = tmp_path / "ref.txt"
    ref.write_text(
        "the belief graph stores weighted edges between nodes",
        encoding="utf-8",
    )
    # Completely different vocabulary — should score near 0.
    over, score, _ = similarity_to_reference(
        "retrieval subsystem uses a two lane ranked list to surface context",
        ref,
        threshold=0.6,
    )
    assert over is False
    assert score < 0.6


def test_similarity_excerpt_is_none_when_clean(
    tmp_path: "pytest.TempPathFactory",  # type: ignore[type-arg]
) -> None:
    """excerpt is None when the input is below threshold."""
    ref = tmp_path / "ref.txt"
    ref.write_text("alpha bravo charlie delta echo", encoding="utf-8")
    over, _, excerpt = similarity_to_reference(
        "one two three four five", ref, threshold=0.6
    )
    assert over is False
    assert excerpt is None
