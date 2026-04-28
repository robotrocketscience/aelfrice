"""Monotonic ULID generator (Crockford base32, 26 chars).

Stdlib-only, no external deps. Hand-rolled per #205 design memo D1
choice "hand-rolled monotonic ULID":

- 48-bit big-endian millisecond timestamp (years 1970–10889).
- 80-bit randomness within the same millisecond.
- Monotone within a process: if `now()` returns the same ms as a
  prior call, the random-portion is incremented by 1 (with carry)
  rather than re-rolled. This guarantees lexicographic sort = time
  sort even under burst writes.
- Cross-process drift is possible but tolerated: the v2.0 ingest_log
  primary key only requires uniqueness, which a re-rolled random
  trivially provides; the monotone property is a sort convenience.

`make_generator()` returns a callable that closes over a deterministic
seed for tests; the module-level `ulid()` uses `os.urandom`.
"""
from __future__ import annotations

import os
import time
from typing import Callable

# Crockford base32: lowercase i/l/o/u removed to avoid ambiguity.
_ALPHABET: str = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

_TIME_BITS: int = 48
_RAND_BITS: int = 80
_TOTAL_BITS: int = _TIME_BITS + _RAND_BITS  # 128
_ULID_LEN: int = 26


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def make_generator(
    rand_source: Callable[[int], bytes] = os.urandom,
    time_source: Callable[[], float] = time.time,
) -> Callable[[], str]:
    """Build a ULID generator with the given entropy + clock sources.

    Defaults to `os.urandom` and `time.time`. Tests can pass a seeded
    rand_source to make IDs deterministic.
    """
    last_ms: list[int] = [-1]
    last_rand: list[int] = [0]

    def gen() -> str:
        ms = int(time_source() * 1000)
        if ms == last_ms[0]:
            # Same ms: increment last_rand by 1 with carry.
            new_rand = (last_rand[0] + 1) & ((1 << _RAND_BITS) - 1)
            if new_rand == 0:
                # Overflow within one ms — burn into next ms to keep
                # monotone. Practically unreachable (2^80 ids/ms).
                ms += 1
                new_rand = int.from_bytes(rand_source(10), "big")
        else:
            new_rand = int.from_bytes(rand_source(10), "big")
        last_ms[0] = ms
        last_rand[0] = new_rand
        value = (ms << _RAND_BITS) | new_rand
        return _encode(value, _ULID_LEN)

    return gen


_default_gen: Callable[[], str] = make_generator()


def ulid() -> str:
    """Return one ULID string. Process-monotone, sortable."""
    return _default_gen()
