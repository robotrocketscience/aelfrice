### Documentation

- **The phantom trigger flags an opportunity and generates no phantom ([#980](https://github.com/robotrocketscience/aelfrice/issues/980)).** The 3.7.0 entries called the #980 detector "trigger-driven generation". The detector only adds an `<aelfrice-phantom-opportunity>` note that suggests `/aelf:wonder`, and aelfrice generates no phantom on that path (`src/aelfrice/phantom_trigger.py`). `docs/user/CONFIG.md` and the design note now say so.
