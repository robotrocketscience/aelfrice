### Documentation

- **ARCHITECTURE.md no longer calls the onboard classifier the only outbound path ([#1926](https://github.com/robotrocketscience/aelfrice/issues/1926)).** Background labeling sends the session's new core candidates from the `Stop` hook, and the pre-issue guard sends words from an issue title to `gh issue list --search`; both are on by default. The sentence now names both and links to the table of outbound paths in PRIVACY.md.
