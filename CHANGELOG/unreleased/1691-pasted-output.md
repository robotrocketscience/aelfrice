### Fixed

- **Pasted terminal and tool output no longer becomes beliefs ([#1691](https://github.com/robotrocketscience/aelfrice/issues/1691)).** Transcript ingest kept a `<pasted_content>` block whole, so each line of a pasted `git status` was stored and injected back on later turns. Pasted prose is still kept. Output is now dropped in two ways:
  - **Terminal transcripts:** a pasted block that opens with a shell prompt is dropped whole.
  - **Output lines:** in any other pasted block, lines in a fixed output format are dropped. These include git status, hint, commit, and diffstat lines; the host's tool and timing lines; shell error prefixes; timestamped log lines; column listings; and bare paths, URLs, and slugs.

  On 12 real pasted blocks (390 beliefs before the change, 43 after), two blind graders rated 99.4% of the removed beliefs as output, and the change removed 94.5% of the beliefs they rated as output. One 313-line terminal paste accounts for 310 of the 347 removals. On the other 11 blocks, 37 of 80 beliefs are removed.
