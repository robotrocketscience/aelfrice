### Removed

- **Contradiction precedence no longer reads the belief type ([#2011](https://github.com/robotrocketscience/aelfrice/issues/2011)).** When `aelf resolve` picked a winner, a belief of type `correction` whose origin had no precedence class of its own, such as `unknown`, took the `user_corrected` class. It now takes `document_recent`, like a belief of any other type. Every origin that names its own class keeps it, so a belief with origin `user_corrected` still takes the `user_corrected` class, and a locked belief still takes the highest class. Resolutions that are already recorded don't change.

  On by default: yes.

  Reached from the hook path: no. The precedence lookup runs only when `aelf resolve` or a library caller resolves a contradiction.

  Measured on: nothing measured.
