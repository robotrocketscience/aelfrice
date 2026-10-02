### Fixed

- **Pasted terminal and tool output no longer becomes beliefs ([#1691](https://github.com/robotrocketscience/aelfrice/issues/1691)).** Transcript ingest kept a `<pasted_content>` block whole, so each line of a pasted `git status` was stored and injected back on later turns. Pasted prose is still kept. Output is now dropped in two ways:
  - **Terminal transcripts:** a pasted block that opens with a shell prompt in fish, bash, or zsh form is dropped whole. An email address at the start of a line isn't treated as a prompt.
  - **Output lines:** in any other pasted block, a line is dropped only when the whole line has a fixed output form. These forms include:
    - git status, hint, commit, merge, and diffstat lines;
    - the host's tool-count and timing lines;
    - a shell's own error lines;
    - log lines with an ISO timestamp, or a clock time followed by a column gap or a level word;
    - lines with two or more column gaps;
    - a URL, a file path, a file name with an extension, or a `+`-joined slug, alone on its line.

  A sentence that only begins like one of these forms is kept, for example "Your branch is behind on reviews". So are words like "and/or", "e.g.", "C++", and "x=1". The paste's wrapper tags are host markup and are always removed. An unclosed paste is filtered to the end of the record. Each pattern runs in linear time, and a 200,000-character line takes milliseconds.

  The change was measured on 12 private pasted blocks. They made 390 beliefs before the change and 42 after. Two blind graders rated 99.4% of the removed beliefs as output, and the change removed 94.8% of the beliefs they rated as output. One 313-line terminal paste accounts for most of the removals. The method and the per-block figures are in the #1691 comments.
