### Internal

- **The Stop hook's comments no longer say that Stop has no `additionalContext` channel ([#1651](https://github.com/robotrocketscience/aelfrice/issues/1651)).** It does have one. The lock prompt still goes to stderr, and the comment now gives the real reason: the listing is for the user, and `additionalContext` would continue the conversation and cost the user a turn. A test comment made the same claim and is corrected too. No behavior changed.
