### Fixed

- **Pasted terminal and tool output no longer becomes beliefs ([#1691](https://github.com/robotrocketscience/aelfrice/issues/1691)).** Transcript ingest kept a `<pasted_content>` block whole, so each line of a pasted `git status` was stored and injected back on later turns. Pasted prose is still kept. Output is now dropped in two ways:
  - **Terminal transcripts:** a pasted block that opens with a shell prompt in fish, bash, or zsh form is dropped whole. A prompt shows a short host name with no dot, so an email address at the start of a line isn't treated as one.
  - **Output lines:** in any other pasted block, a line is dropped only when the whole line has a fixed output form. These forms include:
    - git status, hint, commit, merge, and diffstat lines;
    - the host's tool-count and timing lines;
    - a shell's own error lines;
    - log lines with an ISO timestamp, or a clock time followed by a column gap or a level word;
    - lines with two or more column gaps (a tab, or two or more spaces after a character that doesn't end a sentence) that don't end in `.`, `?`, or `!`;
    - a URL, a file path, a file name whose extension starts with a letter, or a `+`-joined slug, alone on its line. A path needs a leading `~`, `.`, or `/`, a trailing `/`, or a file name at its end.

  A sentence that only begins like one of these forms is kept, for example "Your branch is behind on reviews". So are words like "and/or", "read/write/execute", "e.g.", "3.14", "C++", and "x=1". Some prose is still dropped: a single word that looks like a file name, such as "Node.js", and a line typed with two or more spaces between its words and no closing punctuation. The paste's wrapper tags are host markup and are always removed. An unclosed paste is filtered to the end of the record. The tags are found in one pass and each line pattern runs in linear time, so neither a 200,000-character line nor 20,000 unclosed tags takes more than milliseconds.

  The change was measured on 12 private pasted blocks. They made 390 beliefs before the change and 42 after. Two blind graders rated 99.4% of the removed beliefs as output, and the change removed 94.8% of the beliefs they rated as output. One 313-line terminal paste accounts for most of the removals. The method and the per-block figures are in the #1691 comments.
