### Fixed

- **Transcript ingest no longer stores a background task's report, another session's message, or other machine output as your beliefs ([#1649](https://github.com/robotrocketscience/aelfrice/issues/1649)).** A background task's completion notice reaches the transcript as a user turn, and its `<result>` body is a sub-agent's report. aelfrice's own transcript logger has dropped such a prompt whole since #747. But `aelf ingest-transcript` also reads logs other loggers wrote and the host's own session files. There it applied only the per-sentence filter, which drops tag lines and passes the report's plain sentences.

  Measured on one store on 2026-09-30, 602 ingested rows carried a notice's `<status>` tag, its closing `</result>` tag, or its wrapper, and each became a belief. 169 of those carried `</result>`. Those beliefs were retrieved into later prompts. The counts are a lower bound, because a report's other sentences carry no tag.

  Ingest now keeps only what a person typed:
  - **Records skipped whole:** the host's session files mark records that aren't a person typing. Ingest skips a sidechain record (a sub-agent's prompt or reply), an `isMeta` record (harness text, which the logger already skips), and a compaction summary (the model's digest of the session).
  - **Marker lines cut:** from the rest, it cuts the host's own marker lines: the four-line `[SYSTEM NOTIFICATION]` banner and `[Request interrupted by user]`. It cuts the lines themselves, so a paragraph next to one is kept.
  - **Slash commands dropped whole:** a record that then opens with a slash command's `<command-*>` wrapper is dropped whole, because the wrapper is followed by the command's expanded body.
  - **Harness blocks cut:** every other harness block that opens on its own line is cut, closed or not:
    - `<task-notification>` and its parts;
    - `<system-reminder>`;
    - `<cross-session-message>`;
    - `<aelfrice-worker-context>`;
    - `<tool-result>`;
    - the `<bash-*>` and `<local-command-*>` wrappers;
    - `<user-prompt-submit-hook>`.

  A tag you mention mid-sentence stays, because it's your text. An unclosed block is cut to the end of its record, because nothing marks where it would have ended. What remains goes through the per-sentence filter as before. One real session record that produced 120 beliefs produces none.

  On one machine's session files and a second logger's archive (2026-09-30):
  - the three record flags skip 241,847 records, including 2,209 sub-agent prompts and 15 compaction summaries;
  - of the 3,147 user records left, ingest drops 2,489, and every one is harness text: 2,391 notices, 37 cross-session messages, 10 interrupt markers, and slash-command, bash, and local-command wrappers;
  - no record of plain prose is dropped, and none is trimmed.

  Beliefs already stored from these records aren't removed by this change.
