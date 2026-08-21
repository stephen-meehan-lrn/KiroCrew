"""Whether a NAME-based approval may be honoured for a shell command line.

Every auto-approve tier that decides from a program NAME -- a session-trusted
pattern (``head *``), a configured ``auto_approve_tools`` glob, the read-only
allowlist -- is making a statement about a PROGRAM. The shell then performs its
own ``PATH`` lookup, and a gateway's ``PATH`` legitimately leads with
directories the agent itself can write (a worktree venv's ``bin``, mise shims,
``~/.local/bin``). A file the agent planted at ``~/.local/bin/head`` therefore
wins the lookup over ``/usr/bin/head``, and a grant made because the command
"is just ``head``" runs it.

:func:`name_grant_refusal` answers the one question those tiers need: does the
name still identify the program it appears to name? A refusal does NOT block the
command and does NOT rewrite it -- the request falls through to the ordinary
interactive approval card, where the human decides on this specific command.
That is the whole point: the tier's job is to skip a prompt the user has already
answered in general, and a shadowed name is a case the user has not answered.

Three shapes are refused, and nothing else:

* **A shadowing resolution.** The name resolves somewhere other than the
  same-named program in the trusted system directories
  (:func:`platform_compat.trusted_system_bin`). ``head`` found at
  ``~/.local/bin/head`` while ``/usr/bin/head`` exists is the reported attack.
* **A resolution inside a tree the agent writes** -- the project checkout, the
  LLM workspace root (:func:`github_runner.agent_writable_roots`), or a
  project-local tool directory (``.venv/bin``, ``node_modules/.bin``). No
  shadowing is needed for this one to be suspicious: it is the same class
  ``github_runner.validate_provider_executable`` and the terminal panel's
  command probe already refuse.
* **A name no approval has identified.** The two rules above cannot help a name
  the system directories do not carry -- ``gh``, ``node``, ``kirocrew``, a
  version manager's ``python`` -- because such a program legitimately lives
  where the user installed it, which is also where the agent can write. For
  those the tiers require a WITNESS: a human answering an approval card has seen
  the command and said yes, and that moment records the file's identity
  (:func:`pin_human_approval`). A grant naming the program is then honoured only
  while the same file answers to the name. No pin means refuse -- pinning on
  first SIGHT would bless whatever is there the first time a tier looks, and a
  tier looks precisely when it is about to auto-approve without asking anyone.

A command carrying a construct whose programs cannot be enumerated -- a
substitution inside quotes, a backtick, a process substitution -- is refused
whole, and so is a program token the shell expands (``$CMD``). Seeing part of a
command's program set is not a basis for vouching for the command.

What this deliberately does NOT do, stated plainly so the boundary is not
mistaken for a stronger one:

* It does not make a decision BINDING on the exec. The check runs when the
  approval is decided and the shell resolves again when it runs, so a second
  agent writing the shim in that window still wins. Closing that needs the
  child's ``PATH`` to stop leading with agent-writable directories, which
  changes the execution environment of every command the agent runs and is a
  separate change with its own compatibility surface (upstream issue #4438
  names it).
* It does not decide that a user-owned directory is untrustworthy. A program
  the user installed into ``~/.local/bin`` is theirs, and refusing it outright
  would leave the auto-approve tiers dead on the most common developer host --
  an unused code path, not a security win. It is admitted on a human's say-so
  and only while it stays the same file, which costs one approval card per
  program (and one more after an upgrade) rather than the whole tier.
* It says nothing about full-trust or YOLO mode, which approve everything by
  construction and are not name-based grants.
* The witness is recorded on the dashboard's approval card. Another surface's
  approval does not pin, so a non-system program there keeps prompting -- more
  prompts, never fewer, which is the safe direction to be incomplete in.

Resolution runs against the same ``PATH`` value the spawn code hands the
child (:func:`env.augmented_path`), not this process's own ``PATH``: the child's
is a superset with the version-manager directories PREPENDED, so resolving
against ours would answer for a search order the command will not use.

Cost is a ``which`` walk plus a handful of ``stat`` calls per decision, on the
same order as ``trusted_system_bin``'s own lookup, and it runs on the event loop
where the approval is decided. Building the search path is safe there because
:func:`env.augmented_path` is string work over a glob that
``env._node_all_bin_dirs`` caches for the process lifetime, and that cache is
already warm: the same call builds the ``PATH`` handed to the agent process at
session start, long before any tool approval.

The verdict itself is deliberately uncached: a cached "trusted" answer is a
substitution window, and this must reflect the filesystem as it is when the tier
decides.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shlex
import shutil
from collections import OrderedDict
from dataclasses import dataclass

from kiro_crew import platform_compat
from kiro_crew.env import augmented_path

logger = logging.getLogger(__name__)

#: Path segments that mark a directory as PROJECT-LOCAL tooling rather than an
#: installed program. A binary under one of these is writable by whatever can
#: write the project -- which includes the agent -- so a name that resolves into
#: one is not a name a grant can vouch for.
PROJECT_LOCAL_SEGMENTS = frozenset(
    {
        ".venv",
        "venv",
        ".virtualenv",
        "virtualenv",
        "node_modules",
        ".tox",
        ".nox",
        "vendor",
        ".direnv",
        "target",
        "build",
        "dist",
        ".git",
    }
)

#: Tokens after which the NEXT word is a program name rather than an operand.
#: Redirects (``>``, ``2>&1``) are deliberately absent: what follows them is a
#: file, and treating it as a program would resolve an operand.
_COMMAND_STARTERS = frozenset({"|", "||", "&&", ";", ";;", "&", "|&", "(", ")", "\n"})

#: Every character ``shlex`` may hand back as punctuation with
#: ``punctuation_chars=True``. A token made only of these is an operator, never
#: a program name.
_PUNCTUATION = "();<>|&"


#: Constructs that RUN a program in a position this tokenizer cannot enumerate.
#: ``shlex`` in POSIX mode consumes quotes, so a substitution inside double
#: quotes (``echo "$(head x)"``) collapses into one ordinary token and its inner
#: program disappears from the walk entirely. Refusing the whole command line is
#: the only honest answer: the tier cannot vouch for a program it cannot see.
_UNENUMERABLE = ("$(", "`", "<(", ">(")

#: Characters that make a PROGRAM token something other than a literal name --
#: the shell expands them, so what runs is decided after this check reads it.
_EXPANDING_CHARS = ("$", "`", "*", "?", "[")

#: ``(program name, directory) -> identity`` for the first file each name
#: resolved to. See :func:`_pin_refusal`; bounded so a long-lived gateway cannot
#: accumulate an entry per name it has ever seen.
_PINS: "OrderedDict[tuple[str, str], tuple]" = OrderedDict()
_PIN_LIMIT = 512

# ── Refusal reasons ──
#
# Each refusal carries a CODE as well as its human detail, because the detail is
# built from the command line and from resolved paths: logging it is a dataflow
# from tool input into a log sink, which CodeQL's
# `py/clear-text-logging-sensitive-data` query reports at high severity (and it
# is right to -- a resolved path discloses more than the user typed). Callers log
# ``Refusal.log_text``, which reads a constant OUT of a table below; that severs
# the flow in a way the analysis can verify, where returning the caller's own
# string after checking it would not.

UNENUMERABLE = "unenumerable_construct"
UNTOKENIZABLE = "untokenizable"
EXPANDED = "expanded_program_token"
RELATIVE_PATH = "relative_path_program"
AGENT_TREE = "agent_writable_tree"
SHADOWED = "shadows_system_program"
IDENTITY_CHANGED = "identity_changed"
UNWITNESSED = "no_approval_identified_this_file"
UNINSPECTABLE = "uninspectable"

_REFUSAL_LOG_TEXT = {
    UNENUMERABLE: "the command carries a construct whose programs cannot be enumerated",
    UNTOKENIZABLE: "the command line could not be tokenized, or uses shell grammar "
    "this check does not model",
    EXPANDED: "a program token is expanded by the shell",
    RELATIVE_PATH: "a program is named by relative path",
    AGENT_TREE: "a program resolves inside a tree the agent can write",
    SHADOWED: "a program name shadows the system program of that name",
    IDENTITY_CHANGED: "a program name resolves to a different file than an approval identified",
    UNWITNESSED: "a non-system program has no file identified by an approval",
    UNINSPECTABLE: "a program could not be inspected",
}


@dataclass(frozen=True)
class Refusal:
    """Why a name-based auto-approve must not be honoured.

    ``detail`` names the program and the paths involved and is meant for the
    person deciding at the approval card. ``log_text`` is the constant to log --
    see the note above on why the two are separate.
    """

    code: str
    detail: str

    @property
    def log_text(self) -> str:
        return _REFUSAL_LOG_TEXT.get(self.code, "a program name could not be vouched for")


#: Shell RESERVED WORDS and grouping tokens. This walk models one grammar --
#: simple commands joined by pipes, ``&&``/``||``/``;`` and subshells -- and a
#: reserved word means the command is using grammar it does NOT model, where the
#: real program hides behind a syntax word: in ``head x | { evil; }`` a walk that
#: reads ``{`` as the program never sees ``evil``. Meeting one in a command
#: position refuses the whole line rather than vouching for what it could see.
#:
#: ``test`` and ``[`` are absent on purpose: those are real programs, not
#: grammar. ``time`` and ``!`` are here because they PREFIX a command, which is
#: the same hiding shape.
_RESERVED_WORDS = frozenset(
    {
        "{",
        "}",
        "!",
        "time",
        "if",
        "then",
        "elif",
        "else",
        "fi",
        "for",
        "while",
        "until",
        "do",
        "done",
        "case",
        "esac",
        "select",
        "function",
        "coproc",
        "[[",
        "]]",
    }
)


def _is_redirect(token: str) -> bool:
    """Whether a token is a redirection operator (``>``, ``>>``, ``2>&``, ``<``)."""

    return (
        bool(token) and all(ch in _PUNCTUATION for ch in token) and ("<" in token or ">" in token)
    )


def is_project_local(entry: str) -> bool:
    """Whether a path belongs to a project tree rather than an install.

    Segment-wise, not substring: ``/opt/venv-tools/bin`` is an installed prefix
    that merely CONTAINS the text, while ``/home/u/proj/.venv/bin`` genuinely is
    project-local.

    Both separators are honoured regardless of host. ``os.sep`` alone would make
    this silently useless for POSIX-shaped input on Windows (and vice versa),
    and a security filter that quietly stops matching is worse than one that is
    absent, because the tests covering it keep passing on the host that wrote
    them.
    """

    parts = entry.replace("\\", "/").split("/")
    return any(part in PROJECT_LOCAL_SEGMENTS for part in parts)


def _agent_search_path() -> str:
    """The ``PATH`` a spawned agent command actually searches."""

    return augmented_path(os.environ.get("PATH", ""))


def _agent_writable_roots() -> tuple[str, ...] | None:
    """Trees the agent itself writes, or ``None`` when that cannot be decided.

    ``None`` is fail-closed at every caller ("assume the path IS inside one"):
    a filter that silently dropped a root it could not resolve would admit
    exactly the trees it exists to refuse.

    Read live rather than cached, because a session can retarget its project
    directory between two tool calls.
    """

    try:
        from kiro_crew.github_runner import agent_writable_roots

        return tuple(os.path.normcase(str(root)) for root in agent_writable_roots())
    except Exception:
        logger.warning(
            "agent-writable roots unavailable; refusing to honour a name-based "
            "grant until they can be resolved",
            exc_info=True,
        )
        return None


def _within(path: str, roots: tuple[str, ...] | None) -> bool:
    """Whether *path* sits inside one of *roots*, refusing when *roots* is None.

    Compared against ``root + os.sep`` rather than by bare prefix, so a sibling
    that merely starts with the same characters (``…/workspace-other`` next to
    ``…/workspace``) is outside.
    """

    if roots is None:
        return True
    real = os.path.normcase(path)
    return any(real == root or real.startswith(root + os.sep) for root in roots)


def program_names(command: str) -> list[str] | None:
    """Program tokens of *command*, or ``None`` when it cannot be tokenized.

    Every command position is collected -- each stage of a pipeline, each side
    of ``&&``/``||``/``;``, and the inside of a substitution or subshell -- so a
    grant cannot be honoured on the strength of its first word alone.

    A ``VAR=value`` prefix keeps the position open: it assigns into the
    command's environment, and the program is the token after it.

    ``None`` (unbalanced quotes, an unterminated construct) means argv could not
    be established, which callers treat as a refusal rather than as "no
    programs found".
    """

    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return None
    names: list[str] = []
    expect_program = True
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        if token in _COMMAND_STARTERS:
            expect_program = True
            continue
        # A REDIRECT may appear anywhere in a simple command, INCLUDING BEFORE
        # the program: `2>/dev/null head x` runs `head`. So consume the operator
        # and the file it names, and leave the command position as it was.
        # Closing the position here would make `head` invisible; opening it would
        # judge the FILE in `head x > out`. `2>` and `2>&1` arrive as a digit
        # token followed by the operator, so that fd prefix is consumed too.
        if _is_redirect(token):
            index += 1  # its target, if any
            continue
        if token.isdigit() and index < len(tokens) and _is_redirect(tokens[index]):
            index += 2  # the operator and its target
            continue
        if all(ch in _PUNCTUATION for ch in token):
            continue
        if not expect_program:
            continue
        if token in _RESERVED_WORDS:
            # Grammar this walk does not model. The program is elsewhere in a
            # shape it cannot follow, so report "unknown" rather than the subset
            # it managed to see.
            return None
        # `VAR=value cmd` assigns into the environment; the program follows it.
        # Only a leading bare-word assignment counts -- `--flag=value` is an
        # operand and cannot appear in a command position anyway.
        if "=" in token and not token.startswith("="):
            head = token.split("=", 1)[0]
            if head and head.replace("_", "").isalnum() and not head[0].isdigit():
                continue
        names.append(token)
        expect_program = False
    return names


#: How much of a program file the identity digest covers. Reached on an
#: auto-approve decision, so it is a page-cached read of at most this much, not a
#: hash of a 100 MB interpreter.
_DIGEST_CAP = 1 << 20


def _content_digest(real: str, size: int) -> str | None:
    """A digest of *real*'s bytes, capped at :data:`_DIGEST_CAP`.

    Metadata alone cannot answer "is this the same program": ``mtime`` and
    ``size`` are both under the writer's control (a same-size rewrite followed by
    ``os.utime`` restores the pair exactly), and while ``st_ctime_ns`` is
    kernel-set and unrestorable, its clock has a tick -- a rewrite inside the
    same tick as the pin leaves it equal, measured on both tmpfs and xfs. So the
    digest is what actually decides, and the metadata rides along to catch the
    cheap cases first.

    Files at or under the cap are covered completely, which is the realistic
    plant: a shim is a small script. Above it, the head and tail are digested
    with the size, so a substitution has to preserve both ends and the exact
    length -- a far narrower capability than writing a file, and stated here
    rather than implied.
    """

    try:
        digest = hashlib.sha256()
        digest.update(str(size).encode())
        with open(real, "rb") as handle:
            if size <= _DIGEST_CAP:
                digest.update(handle.read(_DIGEST_CAP))
            else:
                digest.update(handle.read(_DIGEST_CAP // 2))
                handle.seek(-(_DIGEST_CAP // 2), os.SEEK_END)
                digest.update(handle.read(_DIGEST_CAP // 2))
        return digest.hexdigest()
    except OSError:
        return None


def _identity(real: str) -> tuple | None:
    """A value that changes whenever the file behind a name changes.

    Content first (:func:`_content_digest`), with the metadata that a writer
    cannot restore -- inode, device, and the kernel-set ``st_ctime_ns`` -- as
    corroboration. ``mtime`` and ``size`` are included for completeness but are
    NOT what the guarantee rests on: both are forgeable by the same-uid process
    this pin exists to catch.
    """

    try:
        st = os.stat(real)
    except OSError:
        return None
    digest = _content_digest(real, st.st_size)
    if digest is None:
        return None
    return (
        real,
        digest,
        st.st_mtime_ns,
        st.st_ctime_ns,
        st.st_size,
        st.st_ino,
        st.st_dev,
    )


def _pin_refusal(name: str, found: str, real: str, witness: bool) -> Refusal | None:
    """Vouch for a non-system program only on the file a HUMAN approved.

    The shadowing rule cannot help a name the trusted system directories do not
    carry -- ``gh``, ``node``, ``kirocrew``, a version manager's ``python``.
    Those live where the user installed them, which is also where the agent can
    write, so the name alone says nothing about the file.

    Refusing them outright is not a trade worth taking: it would leave the trust
    tiers dead for most of what a developer actually grants, and a "Trust all gh
    commands" button that never takes effect pushes people to blanket trust. What
    the tiers can require instead is a WITNESS. A human answering an approval
    card has seen the command and said yes to it, so that moment records the
    file's identity (:func:`pin_human_approval`); afterwards a grant naming it is
    honoured only while the same file answers to the name.

    That ordering is the point. Pinning on first SIGHT would bless whatever is
    there the first time a tier looks -- and a tier looks precisely when it is
    about to auto-approve without asking anyone, so a file planted before that
    moment would pin itself. No pin therefore means refuse, not adopt.

    Keyed by ``(name, directory)``, so two projects that ship a same-named tool
    do not invalidate each other; only a swap in place is a mismatch.

    A mismatch does NOT re-pin from a check -- re-pinning there would mean "one
    prompt, then trusted", and this code cannot see whether the human answered
    that prompt with yes. The next human approval re-pins, which is how an
    upgraded tool becomes auto-approvable again.

    Bounded LRU: a long-lived gateway must not accumulate an entry per name it
    has ever seen.
    """

    identity = _identity(real)
    if identity is None:
        return Refusal(UNINSPECTABLE, f"{name} could not be inspected")
    key = (name, os.path.normcase(os.path.dirname(found)))
    pinned = _PINS.get(key)
    if witness:
        # A human just approved this command: record what they approved, and
        # replace a stale entry (an upgraded tool) with it.
        _PINS[key] = identity
        _PINS.move_to_end(key)
        if len(_PINS) > _PIN_LIMIT:
            _PINS.popitem(last=False)
        return None
    if pinned is None:
        return Refusal(
            UNWITNESSED,
            f"{name} at {found} is not a system program and no approval has "
            "identified this file, so a grant naming it cannot be honoured yet",
        )
    if pinned != identity:
        return Refusal(
            IDENTITY_CHANGED,
            f"{name} at {found} is not the file an approval identified earlier, "
            "so a grant made about it no longer identifies it",
        )
    _PINS.move_to_end(key)
    return None


def _program_refusal(name: str, witness: bool = False) -> Refusal | None:
    """Why a name-based grant must not be honoured for *name*, else ``None``."""

    if any(ch in name for ch in _EXPANDING_CHARS):
        # `$CMD arg`, `./*.sh`: the shell decides what this names after this
        # check has read it, so no grant can identify the program.
        return Refusal(EXPANDED, f"{name} is expanded by the shell rather than naming a program")
    if "/" in name or (os.sep != "/" and os.sep in name) or (os.altsep and os.altsep in name):
        if not os.path.isabs(name):
            # A relative program is resolved against the command's working
            # directory, which the approval never saw, so no name-based grant
            # can identify what it will run.
            return Refusal(
                RELATIVE_PATH,
                f"{name} names a program by relative path, which the grant cannot identify",
            )
        try:
            real = os.path.realpath(name)
        except OSError:
            return Refusal(UNINSPECTABLE, f"{name} could not be resolved")
        roots = _agent_writable_roots()
        if is_project_local(name) or _within(name, roots) or _within(real, roots):
            return Refusal(AGENT_TREE, f"{name} resolves inside a tree the agent can write")
        if _is_trusted_system_file(os.path.basename(name), real):
            # Spelling the system program's own path out is still the system
            # program; it needs no witness.
            return None
        return _pin_refusal(name, name, real, witness)

    found = shutil.which(name, path=_agent_search_path())
    if not found:
        # Nothing on the search path answers to this name, so there is no
        # shadowed program and nothing to vouch for: the shell will use a
        # builtin (`cd`, `echo`) or fail on its own.
        return None
    try:
        real = os.path.realpath(found)
    except OSError:
        return Refusal(UNINSPECTABLE, f"{name} could not be resolved")
    # `is_project_local` reads the location the name was FOUND in, never the
    # symlink target: a real system install can legitimately resolve THROUGH a
    # segment on that list (`/usr/bin/npm` -> `…/node_modules/npm/bin/npm-cli.js`),
    # and judging the target would refuse it. Where the target LEADS is covered
    # by the agent-writable roots below, which compare whole paths instead of
    # guessing from a segment name.
    roots = _agent_writable_roots()
    if is_project_local(found) or _within(found, roots) or _within(real, roots):
        return Refusal(AGENT_TREE, f"{name} resolves inside a tree the agent can write ({found})")
    system = platform_compat.trusted_system_bin(name)
    if system is not None:
        if not _is_trusted_system_file(name, real):
            return Refusal(
                SHADOWED,
                f"{name} resolves to {found}, which shadows the system program at {system}",
            )
        # The system program itself. The name identifies it by construction, so
        # no witness is needed -- this is what keeps coreutils and the read-only
        # allowlist working with no approval history at all.
        return None
    return _pin_refusal(name, found, real, witness)


def _is_trusted_system_file(name: str, real: str) -> bool:
    """Whether *real* IS the trusted system program called *name*."""

    system = platform_compat.trusted_system_bin(name)
    if system is None:
        return False
    try:
        return os.path.normcase(os.path.realpath(system)) == os.path.normcase(real)
    except OSError:
        return False


def pin_human_approval(command: str) -> None:
    """Record the programs in a command a HUMAN just approved.

    This is what makes a later name-based grant honourable for a program the
    trusted system directories do not carry: the person saw this command on the
    approval card and said yes, so the file behind each of its program names is
    the file their decision was about. :func:`_pin_refusal` refuses such a name
    until this has run, and refuses it again once a DIFFERENT file answers to it.

    Call it only on a genuine human answer -- never from an auto-approve path,
    which is the very thing the pin exists to constrain. Failures are swallowed:
    a missing pin costs one prompt, and an approval must not fail because a
    program could not be stat-ed.
    """

    try:
        for name in program_names(command) or []:
            _program_refusal(name, witness=True)
    except Exception:
        logger.debug("could not record approved program identities", exc_info=True)


def name_grant_refusal(command: str) -> Refusal | None:
    """Why *command* may not be auto-approved by NAME, or ``None`` when it may.

    The result is a diagnostic, not a denial: the caller falls through to
    interactive approval, so a refusal costs one prompt and never blocks the
    command. Log ``Refusal.log_text`` (a constant) and show ``Refusal.detail``
    to the person deciding.

    An empty command returns ``None`` -- there is no name to vouch for, and the
    tiers that call this have already established they have a command.
    """

    if not command.strip():
        return None
    for construct in _UNENUMERABLE:
        if construct in command:
            # A substitution runs a program in a position the tokenizer cannot
            # reach (POSIX quote handling swallows `"$(head x)"` whole), so the
            # command's program set is not knowable here. Refuse rather than
            # vouch for the part that happens to be visible.
            return Refusal(
                UNENUMERABLE,
                f"the command line contains {construct!r}, whose programs cannot be enumerated",
            )
    names = program_names(command)
    if names is None:
        return Refusal(
            UNTOKENIZABLE,
            "the command line could not be reduced to a known set of program names",
        )
    for name in names:
        refusal = _program_refusal(name)
        if refusal is not None:
            return refusal
    return None
