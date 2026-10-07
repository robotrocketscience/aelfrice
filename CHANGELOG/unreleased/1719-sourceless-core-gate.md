### Fixed

- **A missing classifier source no longer empties the first-prompt session-start block ([#1719](https://github.com/robotrocketscience/aelfrice/issues/1719)).** `aelfrice.core_gate` derives `CLASSIFIER_VERSION` from the source of its prompt builder and reply parser. On an install that ships only bytecode, `inspect.getsource` raises `OSError`, so the import failed; a `.pyc` compiled from a different `.py` can raise `SyntaxError` or `tokenize.TokenError` the same way. The UserPromptSubmit hook imports the module while it builds the first prompt's `<session-start>` block whenever there are core candidates, and its fail-soft wrapper then returned the block empty, so its `<locked>` and `<core>` sections were missing. Locks still reached the model through ordinary retrieval. Now `CLASSIFIER_VERSION` is `None` when the source can't be read, and the failure stays in the core lane. The gate admits no unlocked belief to `<core>`, no stored label matches a `None` version, and nothing writes a batch or a label. Each command that would write refuses with a message and exits 1:

  - `aelf core-gate accept` refuses before it reads stdin.
  - `aelf doctor core-gate --emit` and `--rerun` refuse before they create `--out`.
  - `aelf doctor --gc-filesystem-corroboration` refuses before it opens the store, in both the dry run and `--apply`, so it prints no report and deletes no rows. With every unlocked belief out of core, it would otherwise report nothing leaving core and delete rows that keep beliefs there once the classifier is back.

  The Stop hook emits no session-end batch, and `aelf doctor core-gate` reports that no candidate is admitted. When the source is present, the version is unchanged.
