"""Where the ``imsg`` bridge binary is allowed to come from.

The channel spawns a child process, so *who chooses that executable* is a
security boundary rather than a preference. It deliberately cannot be chosen by
configuration: ``config.json`` is agent-writable -- ``security.py`` says so in
as many words, and records that this is why the computer-use enable and the
denied-command opt-out live on the KEYSTONE floor instead -- so a settable
``cli_path`` would let any auto-approved agent shell write a payload path into
``config.json`` and have the gateway execute it, outside the agent sandbox and
with the gateway's privileges, on the next restart.

No other channel takes an executable path from configuration, so there is no
existing pattern to follow here: Slack's ``command`` is a slash-command trigger
word, not a binary. Resolution is therefore fixed in code.

The candidate list exists because ``PATH`` alone is not enough in the
deployment that matters most: under a launch agent the gateway inherits a
minimal ``PATH`` with no Homebrew prefix, which is exactly the case an operator
would otherwise reach for an absolute-path override to solve. Both Homebrew
prefixes are covered -- ``/opt/homebrew`` on Apple Silicon, ``/usr/local`` on
Intel -- which is where ``brew install steipete/tap/imsg`` puts it.

Relying on ``PATH`` for the first lookup is the same trust the repo already
places in it for ``git``, ``gh`` and ``npm``; it is the existing baseline, not a
surface this channel adds.
"""

from __future__ import annotations

import shutil
from pathlib import Path

BRIDGE_BINARY = "imsg"

# Standard install locations, tried in order after PATH. Absolute and fixed at
# source level: an entry here is a decision made by a reviewer, not by whatever
# happens to be in a config file at runtime.
TRUSTED_BRIDGE_PATHS: tuple[str, ...] = (
    "/opt/homebrew/bin/imsg",
    "/usr/local/bin/imsg",
)


def resolve_bridge_path() -> str:
    """Return the bridge executable to spawn, or ``""`` when none is installed.

    An empty return is the "not installed" signal the readiness surface renders
    as *needs setup*; it is never a reason to fall back to a caller-supplied
    string, because there is no caller-supplied string to fall back to.
    """
    found = shutil.which(BRIDGE_BINARY)
    if found:
        return found
    for candidate in TRUSTED_BRIDGE_PATHS:
        path = Path(candidate)
        try:
            if path.is_file():
                return str(path)
        except OSError:
            # An unreadable or unstattable candidate is simply not a hit; the
            # next one, or the "not installed" answer, is still correct.
            continue
    return ""
