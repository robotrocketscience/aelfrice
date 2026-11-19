### Fixed

- **The source distribution no longer ships the Hypothesis test cache ([#1940](https://github.com/robotrocketscience/aelfrice/issues/1940)).** The release job ran the tests and then built the packages in the same checkout, so the 5.3.0 and 5.4.0 sdists on PyPI carried the `.hypothesis/` folder the tests wrote: 239 cache files in 5.4.0. The release job now builds before it runs the tests, `.gitignore` lists `.hypothesis/`, and the sdist configuration in `pyproject.toml` excludes it. The sdist now holds the tracked files plus `PKG-INFO`. The wheel was never affected.
