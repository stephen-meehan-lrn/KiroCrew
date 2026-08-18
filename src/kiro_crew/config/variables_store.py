"""Storage for user-defined ``{{name}}`` variables, in their own file.

WHY THIS IS NOT IN ``config.json``
==================================

It was, and that placement was the root cause of most of this feature's review
history. ``KiroCrewConfig.save()`` serializes the MERGED config and replaces the
whole file, and ``to_dict()`` builds an explicit dict, so the file is a lossy
whole-document rewrite of exactly the keys the dataclass models. For a map whose
only legitimate writer is a dedicated endpoint, that produced a genuine trilemma —
every possible behaviour for the variables slot during an unrelated ``save()`` is
wrong in a different way:

* serialize the merged value  -> overwrites a base value the overlay shadowed, and
  the shadowed value is not in the merged view at all, so it is unrecoverable;
* preserve it while holding the config lock -> ``save()`` is a sync method called
  from 13 async call sites, so a contended POSIX flock stalls the event loop;
* preserve it with an unlocked read -> the read-then-write window silently drops a
  variables write that already returned 200 to its caller.

Moving the data out deletes the trilemma rather than choosing among its three
positions. ``save()`` no longer serializes variables at all, so there is nothing to
preserve, no lock to interact with, and no window. It also removes the overlay
subtraction problem, the overlay-owned-key refusal, and the deleted-workspace
resurrection window — all of which existed only because this map lived inside a
document with a second overlay layer and a whole-file writer.

The cost, stated plainly: variables are no longer part of ``config.json``, so they
are not covered by whatever backs that file up, and a hand-edit goes here instead.
There is no migration path because no released version stored them anywhere.

SHAPE
=====

One flat document, one writer, three scopes::

    {
      "global":     {"NAME": "value"},
      "workspaces": {"ops": {"NAME": "value"}},
      "crews":      {"reviewer": {"NAME": "value"}}
    }

Session scope is deliberately absent: it is per-turn state, never persisted.

READ is tolerant, WRITE is strict. An unreadable or malformed store resolves to no
variables rather than raising, because a broken store must not take the gateway down
over an optional feature. A WRITE refuses a malformed container instead of replacing
it, because the operator's hand-written value is the only copy there is.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCOPE_GLOBAL = "global"
SCOPE_WORKSPACE = "workspace"
SCOPE_CREW = "crew"

# The scope's key in the stored document. Global is a flat map; the other two are
# maps of name -> map, so they need a container key.
_CONTAINER = {SCOPE_WORKSPACE: "workspaces", SCOPE_CREW: "crews"}

_STORE_NAME = "variables.json"


class MalformedStore(Exception):
    """A container the write would have to replace holds a non-mapping.

    Refused rather than coerced: replacing it would discard whatever the operator
    hand-wrote, and there is no second copy to restore from. Carries the dotted path
    so the caller can tell the operator what to repair.
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = path


def store_path() -> Path:
    """Location of the variables store, beside ``config.json``.

    Derived from ``config_path()`` rather than hardcoded so a relocated or
    test-redirected config root carries the store with it. Imported lazily because
    this module is a leaf and ``loader`` imports it.
    """
    from kiro_crew.config.loader import config_path

    return config_path().parent / _STORE_NAME


def read_store() -> dict[str, Any]:
    """Read the raw store document. Never raises.

    Every failure resolves to an empty document, which resolves to no variables. A
    malformed store must not break a gateway boot over an optional feature; the
    write path is where a malformed value is reported, because that is where it can
    be acted on and where silence would destroy data.
    """
    path = store_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "variables store at %s is unreadable (%s); resolving no variables. "
            "Repair or remove that file to restore them.",
            path.name,
            exc.__class__.__name__,
        )
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "variables store at %s is not a JSON object; resolving no variables.",
            path.name,
        )
        return {}
    return raw


def _clean_pairs(raw: object, where: str) -> dict[str, str]:
    """Coerce one scope's map to validated str->str pairs.

    Delegates to the loader's ``coerce_variables`` so validation lives in exactly one
    place — the same name grammar and value-length cap the write path enforces, and
    the same drop-one-pair-not-the-scope tolerance. Imported lazily: the loader
    imports this module, so a module-level import here would close a cycle.
    """
    from kiro_crew.config.loader import coerce_variables

    return coerce_variables(raw, where)


def global_values(doc: dict[str, Any] | None = None) -> dict[str, str]:
    """Global-scope pairs."""
    doc = read_store() if doc is None else doc
    return _clean_pairs(doc.get(SCOPE_GLOBAL), SCOPE_GLOBAL)


def scoped_values(scope: str, doc: dict[str, Any] | None = None) -> dict[str, dict[str, str]]:
    """All named maps for ``workspace`` or ``crew`` scope."""
    container = _CONTAINER[scope]
    doc = read_store() if doc is None else doc
    raw = doc.get(container)
    if not isinstance(raw, dict):
        if raw is not None:
            logger.warning("variables store: %s is not an object; ignoring it", container)
        return {}
    return {
        name: _clean_pairs(pairs, f"{container}.{name}")
        for name, pairs in raw.items()
        if isinstance(name, str)
    }


def _mutate(
    doc: dict[str, Any],
    *,
    scope: str,
    name: str,
    values: dict[str, str],
    removals: list[str],
) -> dict[str, Any]:
    """Apply a per-KEY patch to the document read under the lock.

    Named keys only: a key nobody mentioned is never read and never rewritten, so
    two concurrent writers touching different keys cannot lose each other's edits,
    and there is no whole-scope echo to go stale.

    A container that is ABSENT is created — that is the legitimate first write. A
    container that is PRESENT but not a mapping is refused, because replacing it
    would destroy the only copy of what the operator wrote.
    """
    if scope == SCOPE_GLOBAL:
        target = doc.get(SCOPE_GLOBAL)
        if target is None:
            target = {}
            doc[SCOPE_GLOBAL] = target
        elif not isinstance(target, dict):
            raise MalformedStore(SCOPE_GLOBAL)
    else:
        container = _CONTAINER[scope]
        holder = doc.get(container)
        if holder is None:
            holder = {}
            doc[container] = holder
        elif not isinstance(holder, dict):
            raise MalformedStore(container)
        target = holder.get(name)
        if target is None:
            target = {}
            holder[name] = target
        elif not isinstance(target, dict):
            raise MalformedStore(f"{container}.{name}")

    for key, value in values.items():
        target[key] = value
    for key in removals:
        target.pop(key, None)
    return doc


def patch_store(
    *,
    scope: str,
    name: str = "",
    values: dict[str, str] | None = None,
    removals: list[str] | None = None,
) -> dict[str, Any]:
    """Apply a per-key patch under the store's own lock. Blocking; call off-loop.

    Routed through ``update_config_locked`` so the read and the write are one
    transaction against the store's advisory lock. That helper is reused rather than
    re-implemented so this file inherits its atomic replace, its mode preservation,
    and its symlink handling.

    This is the ONLY writer. ``KiroCrewConfig.save()`` does not touch this file,
    which is the entire point of the file existing.
    """
    if scope not in (SCOPE_GLOBAL, SCOPE_WORKSPACE, SCOPE_CREW):
        raise ValueError(f"unknown variables scope: {scope!r}")
    if scope != SCOPE_GLOBAL and not name:
        raise ValueError(f"{scope} scope requires a name")

    from kiro_crew.config.loader import update_config_locked

    vals = dict(values or {})
    dels = list(removals or [])

    def _apply(current: dict) -> dict:
        return _mutate(current, scope=scope, name=name, values=vals, removals=dels)

    # on_corrupt="fail": a corrupt store must NOT be reset to {} by a write, which
    # would delete every variable at every scope to service one patch. read_store()
    # is the tolerant path; this one refuses and the caller reports it.
    #
    # stamp_meta=False: this is not a config document and must not grow config's
    # bookkeeping keys — the shape here is exactly the three scope containers.
    result = update_config_locked(store_path(), mutate=_apply, stamp_meta=False, on_corrupt="fail")
    try:
        os.chmod(store_path(), 0o600)
    except OSError:
        # Mode is defence in depth, not the security boundary — values are declared
        # non-secret. A filesystem that refuses chmod must not fail the write.
        logger.debug("could not tighten mode on the variables store", exc_info=True)
    return result if isinstance(result, dict) else {}
