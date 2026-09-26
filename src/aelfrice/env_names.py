"""Environment-variable names shared across modules that must not import
each other.

This module exists to hold one end of a dependency, not to be a home for
every env-var name in the product. Most names belong in the module that
reads them; a name lands here only when two modules need it and importing
one from the other would form a cycle.

Nothing here may import from `aelfrice` — that is the whole property. A
module with no outgoing edges cannot be part of a cycle, so anything it
holds is safe for any module to import at any scope.

#1626: `hook` gained a function-local `import cli`, because a typed
`/aelf:` command has to run through the CLI. `cli` already had a
function-local `from aelfrice.hook import ENV_SESSIONSTART_RECAP`, used
only to interpolate the name into a help string. Together those two
edges closed a cycle. Both imports were lazy, so nothing broke at
interpreter start, but a cycle that is only survivable because every
edge is deferred is a cycle waiting for the first module-scope import to
turn it into an ImportError. Moving the constant here deletes the
`cli -> hook` edge outright instead of documenting around it.
"""

from __future__ import annotations

from typing import Final

ENV_SESSIONSTART_RECAP: Final[str] = "AELFRICE_SESSIONSTART_RECAP"
"""Set to '0' to suppress the SessionStart belief-write recap line."""
