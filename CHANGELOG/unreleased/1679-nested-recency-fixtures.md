### Internal

- **The recency-keying tests use nested files that share a basename ([#1679](https://github.com/robotrocketscience/aelfrice/issues/1679)).** The #1619 fixtures put every file at the repository root, where a file's basename equals its relative path, so a lookup keyed on the basename passed them. The fixtures now use `p/m` and `q/m` with distinct dates, and the onboard handshake gives each of its four files its own date. A basename-keyed or first-entry lookup now fails two tests at each lookup site. No production code changed.
