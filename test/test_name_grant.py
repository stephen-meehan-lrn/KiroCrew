"""A name-based auto-approve must not be honoured for a shadowed program name.

Upstream issue #4438: trust grants and the read-only allowlist authorize a
command by program NAME, while the shell resolves that name afterwards through a
``PATH`` that can lead with directories the agent itself writes. These tests pin
both halves -- the decision function and the tiers wired to it.

Every test that resolves a name builds its own ``PATH`` and its own stand-in for
the trusted system directories, so no assertion depends on what the host has
installed.
"""

from __future__ import annotations

import os

import pytest

from kiro_crew import name_grant, platform_compat
from kiro_crew.hooks import TOOL_ALLOW, TOOL_AUTO_APPROVE, HookManager, HooksConfig

pytestmark = pytest.mark.skipif(
    platform_compat.IS_WINDOWS,
    reason="resolution fixtures rely on the POSIX execute bit and a ':'-joined PATH",
)


def _program(directory, name: str) -> str:
    """Create an executable *name* in *directory* and return its path."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return str(path)


@pytest.fixture(autouse=True)
def _clear_pins():
    """Each test starts with no pinned identities.

    The pin store is process-wide by design (a file's identity is not a property
    of one session), so tests must not inherit each other's observations.
    """

    name_grant._PINS.clear()
    yield
    name_grant._PINS.clear()


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A hermetic search path plus a stand-in for the system bin directories.

    Returns the tuple ``(system_dir, user_dir)``: ``user_dir`` comes FIRST on the
    search path, which is the ordering the issue reports on a real host.
    """

    system_dir = tmp_path / "usr" / "bin"
    user_dir = tmp_path / "home" / ".local" / "bin"
    system_dir.mkdir(parents=True)
    user_dir.mkdir(parents=True)

    monkeypatch.setattr(
        name_grant,
        "_agent_search_path",
        lambda: os.pathsep.join([str(user_dir), str(system_dir)]),
    )

    def fake_system_bin(name: str) -> str | None:
        candidate = system_dir / name
        return str(candidate) if candidate.is_file() else None

    monkeypatch.setattr(platform_compat, "trusted_system_bin", fake_system_bin)
    # No project checkout / workspace root unless a test declares one.
    monkeypatch.setattr(name_grant, "_agent_writable_roots", lambda: ())
    return system_dir, user_dir


class TestProgramNames:
    """Every command position is collected, and only command positions."""

    def test_pipeline_and_chain_positions(self):
        assert name_grant.program_names("cat a | grep x | wc -l") == ["cat", "grep", "wc"]
        assert name_grant.program_names("cd /tmp && ls") == ["cd", "ls"]
        assert name_grant.program_names("a; b || c") == ["a", "b", "c"]

    def test_environment_prefix_keeps_the_position_open(self):
        assert name_grant.program_names("FOO=bar head x") == ["head"]
        assert name_grant.program_names("A=1 B=2 grep x") == ["grep"]

    def test_redirect_target_is_not_a_program(self):
        # `>` is punctuation but does NOT open a command position: what follows
        # is a file. Judging it would resolve an operand.
        assert name_grant.program_names("head x > /tmp/out") == ["head"]
        assert name_grant.program_names("head x 2>&1") == ["head"]

    def test_leading_redirect_does_not_hide_the_program(self):
        # GPT 5.6's round-2 finding: a redirect may PRECEDE the program, and the
        # fd prefix arrives as its own token, so a position-closing rule made the
        # real program invisible.
        assert name_grant.program_names("2>/dev/null head README.md") == ["head"]
        assert name_grant.program_names("2>&1 head x") == ["head"]
        assert name_grant.program_names(">out head x") == ["head"]
        assert name_grant.program_names("cat a | 2>/dev/null grep x") == ["cat", "grep"]

    def test_substitution_inner_program_is_collected(self):
        assert name_grant.program_names("echo $(head x)") == ["echo", "head"]

    def test_quoted_separator_is_not_a_position(self):
        assert name_grant.program_names("echo 'a && b'") == ["echo"]

    def test_untokenizable_is_none_not_empty(self):
        # None means "argv could not be established", which the caller refuses.
        # An empty list would read as "no programs found" and be honoured.
        assert name_grant.program_names("head 'unbalanced") is None

    @pytest.mark.parametrize(
        "command",
        [
            "head x | { evil; }",  # GPT 5.6 round-3: `{` read as the program
            "if true; then evil; fi",
            "for f in a b; do evil; done",
            "while true; do evil; done",
            "! evil",
            "time evil",
            "case x in a) evil;; esac",
        ],
    )
    def test_unmodelled_grammar_is_none(self, command):
        # This walk models simple commands joined by pipes, `&&`/`||`/`;` and
        # subshells. A reserved word means the real program hides behind a syntax
        # word, so the answer is "unknown", never the subset it could see.
        assert name_grant.program_names(command) is None

    def test_subshell_is_still_walked(self):
        # `(` opens a command position, so a subshell needs no refusal.
        assert name_grant.program_names("head x && (grep y)") == ["head", "grep"]


class TestShadowedResolution:
    """The reported attack and the cases that must stay auto-approvable."""

    def test_system_program_is_honoured(self, world):
        system_dir, _ = world
        _program(system_dir, "head")
        assert name_grant.name_grant_refusal("head -5 /etc/hosts") is None

    def test_planted_shim_that_shadows_a_system_program_is_refused(self, world):
        system_dir, user_dir = world
        _program(system_dir, "head")
        shim = _program(user_dir, "head")
        refusal = name_grant.name_grant_refusal("head -5 /etc/hosts")
        assert refusal is not None
        assert refusal.code == name_grant.SHADOWED
        assert shim in refusal.detail
        assert str(system_dir / "head") in refusal.detail

    def test_shim_in_a_later_pipeline_stage_is_refused(self, world):
        system_dir, user_dir = world
        _program(system_dir, "cat")
        _program(system_dir, "grep")
        _program(user_dir, "grep")
        assert name_grant.name_grant_refusal("cat a | grep x") is not None

    def test_user_installed_program_with_no_system_twin_shadows_nothing(self, world):
        # `gh`, `node`, `kirocrew`: nothing in the system directories answers to
        # the name, so the shadowing rule has no opinion. Such a name is gated by
        # the witnessed pin instead (see TestIdentityPin) -- here the point is
        # only that it is not refused AS A SHADOW.
        _, user_dir = world
        _program(user_dir, "gh")
        refusal = name_grant.name_grant_refusal("gh pr view 1")
        assert refusal is not None
        assert refusal.code != name_grant.SHADOWED

    def test_unresolvable_name_is_honoured(self, world):
        # A shell builtin (`cd`) resolves nowhere. There is no shadowed program,
        # and refusing every builtin would break `cd /tmp && ls`.
        system_dir, _ = world
        _program(system_dir, "ls")
        assert name_grant.name_grant_refusal("cd /tmp && ls") is None

    def test_symlink_to_the_same_system_file_is_honoured(self, world):
        # Distros ship `/usr/bin/head` -> a multi-call binary, and a second
        # spelling of the SAME file is not a substitution.
        system_dir, user_dir = world
        real = _program(system_dir, "head")
        (user_dir / "head").symlink_to(real)
        assert name_grant.name_grant_refusal("head x") is None

    def test_untokenizable_command_is_refused(self, world):
        assert name_grant.name_grant_refusal("head 'unbalanced") is not None

    def test_empty_command_is_not_refused(self, world):
        assert name_grant.name_grant_refusal("   ") is None


class TestAgentWritableTrees:
    """A resolution inside a tree the agent writes needs no shadowing."""

    def test_resolution_inside_the_project_checkout_is_refused(self, world, tmp_path, monkeypatch):
        system_dir, _ = world
        _program(system_dir, "head")
        checkout = tmp_path / "checkout"
        planted = _program(checkout / "bin", "tool")
        monkeypatch.setattr(
            name_grant,
            "_agent_search_path",
            lambda: os.pathsep.join([str(checkout / "bin"), str(system_dir)]),
        )
        monkeypatch.setattr(
            name_grant, "_agent_writable_roots", lambda: (os.path.normcase(str(checkout)),)
        )
        refusal = name_grant.name_grant_refusal("tool --list")
        assert refusal is not None
        assert refusal.code == name_grant.AGENT_TREE
        assert planted in refusal.detail

    def test_project_local_tool_directory_is_refused(self, world, tmp_path, monkeypatch):
        system_dir, _ = world
        venv_bin = tmp_path / "proj" / ".venv" / "bin"
        _program(venv_bin, "tool")
        monkeypatch.setattr(
            name_grant,
            "_agent_search_path",
            lambda: os.pathsep.join([str(venv_bin), str(system_dir)]),
        )
        assert name_grant.name_grant_refusal("tool --list") is not None

    def test_system_install_resolving_through_a_project_segment_is_honoured(
        self, world, tmp_path, monkeypatch
    ):
        # `/usr/bin/npm` -> `…/node_modules/npm/bin/npm-cli.js`. The segment list
        # describes where a name was FOUND, so judging the symlink TARGET would
        # refuse a stock install. Regression guard for that false positive.
        system_dir, _ = world
        target = _program(tmp_path / "usr" / "lib" / "node_modules" / "npm" / "bin", "npm-cli.js")
        (system_dir / "npm").symlink_to(target)
        assert name_grant.name_grant_refusal("npm run build") is None

    def test_roots_lookup_failure_fails_closed(self, world, monkeypatch):
        system_dir, _ = world
        _program(system_dir, "head")
        monkeypatch.setattr(name_grant, "_agent_writable_roots", lambda: None)
        assert name_grant.name_grant_refusal("head x") is not None


class TestPathFormPrograms:
    """A program named by path, not by name."""

    def test_relative_path_is_refused(self, world):
        assert name_grant.name_grant_refusal("./gradlew build") is not None

    def test_absolute_path_outside_the_agent_trees_is_honoured(self, world):
        system_dir, _ = world
        head = _program(system_dir, "head")
        assert name_grant.name_grant_refusal(f"{head} -5 /etc/hosts") is None

    def test_absolute_path_inside_an_agent_tree_is_refused(self, world, tmp_path, monkeypatch):
        checkout = tmp_path / "checkout"
        planted = _program(checkout / "bin", "payload")
        monkeypatch.setattr(
            name_grant, "_agent_writable_roots", lambda: (os.path.normcase(str(checkout)),)
        )
        assert name_grant.name_grant_refusal(f"{planted} --help") is not None


class TestUnenumerableConstructs:
    """Seeing part of a command's program set is not a basis for vouching."""

    @pytest.mark.parametrize(
        "command",
        [
            'echo "$(head x)"',  # POSIX quote handling swallows the substitution
            "echo $(head x)",
            'echo "`head x`"',
            "echo `head x`",
            "cat <(head x)",
            "diff <(head a) <(head b)",
        ],
    )
    def test_substitutions_are_refused(self, world, command):
        system_dir, _ = world
        _program(system_dir, "echo")
        _program(system_dir, "cat")
        _program(system_dir, "diff")
        assert name_grant.name_grant_refusal(command) is not None

    def test_expanded_program_token_is_refused(self, world):
        assert name_grant.name_grant_refusal("$CMD --version") is not None
        assert name_grant.name_grant_refusal("./*.sh") is not None


class TestIdentityPin:
    """A non-system program is vouched for only on the file a human approved."""

    def test_unwitnessed_program_is_refused(self, world):
        # GPT 5.6's round-2 finding: pinning on first SIGHT would bless whatever
        # is there when a tier first looks -- and a tier looks precisely when it
        # is about to auto-approve without asking anyone.
        _, user_dir = world
        _program(user_dir, "gh")
        refusal = name_grant.name_grant_refusal("gh pr view 1")
        assert refusal is not None
        assert refusal.code == name_grant.UNWITNESSED

    def test_human_approval_makes_it_auto_approvable(self, world):
        _, user_dir = world
        _program(user_dir, "gh")
        name_grant.pin_human_approval("gh pr view 1")
        assert name_grant.name_grant_refusal("gh pr view 1") is None
        assert name_grant.name_grant_refusal("gh pr view 2") is None

    def test_replaced_binary_with_no_system_twin_is_refused(self, world):
        # `gh` has no `/usr/bin/gh`, so the shadowing rule cannot see a
        # substitution. The witnessed pin can.
        _, user_dir = world
        _program(user_dir, "gh")
        name_grant.pin_human_approval("gh pr view 1")
        assert name_grant.name_grant_refusal("gh pr view 1") is None
        (user_dir / "gh").write_text("#!/bin/sh\necho pwned\n")
        (user_dir / "gh").chmod(0o755)
        refusal = name_grant.name_grant_refusal("gh pr view 1")
        assert refusal is not None
        assert refusal.code == name_grant.IDENTITY_CHANGED
        assert "gh" in refusal.detail

    def test_mismatch_does_not_re_pin_from_a_check(self, world):
        # Re-pinning on a check would mean "one prompt, then trusted" -- and this
        # code cannot see whether the human said yes to that prompt. Only a real
        # approval re-pins.
        _, user_dir = world
        _program(user_dir, "gh")
        name_grant.pin_human_approval("gh pr view 1")
        (user_dir / "gh").write_text("#!/bin/sh\necho pwned\n")
        (user_dir / "gh").chmod(0o755)
        assert name_grant.name_grant_refusal("gh pr view 1") is not None
        assert name_grant.name_grant_refusal("gh pr view 1") is not None
        name_grant.pin_human_approval("gh pr view 1")
        assert name_grant.name_grant_refusal("gh pr view 1") is None

    def test_system_program_needs_no_witness(self, world):
        # What keeps coreutils and the read-only allowlist working with no
        # approval history at all.
        system_dir, _ = world
        _program(system_dir, "head")
        assert name_grant.name_grant_refusal("head x") is None

    def test_absolute_system_path_needs_no_witness(self, world):
        system_dir, _ = world
        head = _program(system_dir, "head")
        assert name_grant.name_grant_refusal(f"{head} x") is None

    def test_same_name_in_two_directories_is_independent(self, world, tmp_path, monkeypatch):
        # Two projects shipping a same-named tool must not invalidate each other:
        # only a swap IN PLACE is a mismatch.
        system_dir, user_dir = world
        _program(user_dir, "toolx")
        name_grant.pin_human_approval("toolx run")
        assert name_grant.name_grant_refusal("toolx run") is None
        other = tmp_path / "other" / "bin"
        _program(other, "toolx")
        monkeypatch.setattr(
            name_grant,
            "_agent_search_path",
            lambda: os.pathsep.join([str(other), str(system_dir)]),
        )
        # A different directory is a different pin, so it starts unwitnessed
        # rather than inheriting the other one's approval.
        refusal = name_grant.name_grant_refusal("toolx run")
        assert refusal is not None and refusal.code == name_grant.UNWITNESSED
        name_grant.pin_human_approval("toolx run")
        assert name_grant.name_grant_refusal("toolx run") is None

    def test_same_size_rewrite_with_restored_mtime_is_refused(self, world):
        # GPT 5.6 round-3: mtime and size are both under the writer's control, so
        # a same-size in-place rewrite plus os.utime restores them exactly. No
        # sleep here on purpose -- the rewrite lands inside the same ctime tick as
        # the pin, so the metadata alone (including the kernel-set ctime) is
        # identical and only the content digest can tell them apart.
        _, user_dir = world
        gh = user_dir / "gh"
        gh.write_text("#!/bin/sh\nexit 0\n")
        gh.chmod(0o755)
        before = gh.stat()
        name_grant.pin_human_approval("gh pr view 1")
        assert name_grant.name_grant_refusal("gh pr view 1") is None
        gh.write_text("#!/bin/sh\nexit 9\n")  # byte-for-byte same length
        os.utime(gh, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = gh.stat()
        assert after.st_size == before.st_size
        assert after.st_mtime_ns == before.st_mtime_ns
        assert after.st_ino == before.st_ino
        refusal = name_grant.name_grant_refusal("gh pr view 1")
        assert refusal is not None
        assert refusal.code == name_grant.IDENTITY_CHANGED

    def test_above_cap_file_is_digested_at_its_ends(self, world):
        # A file larger than the digest cap is covered at head and tail plus its
        # size, so a substitution has to preserve both ends and the exact length.
        _, user_dir = world
        big = user_dir / "toolbig"
        payload = bytearray(b"A" * (name_grant._DIGEST_CAP + 4096))
        big.write_bytes(bytes(payload))
        big.chmod(0o755)
        before = big.stat()
        name_grant.pin_human_approval("toolbig run")
        assert name_grant.name_grant_refusal("toolbig run") is None
        payload[0:8] = b"BBBBBBBB"  # same size, changed head
        big.write_bytes(bytes(payload))
        os.utime(big, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert big.stat().st_size == before.st_size
        refusal = name_grant.name_grant_refusal("toolbig run")
        assert refusal is not None
        assert refusal.code == name_grant.IDENTITY_CHANGED

    def test_pin_store_is_bounded(self, world):
        _, user_dir = world
        _program(user_dir, "gh")
        for index in range(name_grant._PIN_LIMIT + 20):
            name_grant.pin_human_approval(f"synthetic{index}")
        name_grant.pin_human_approval("gh pr view 1")
        assert len(name_grant._PINS) <= name_grant._PIN_LIMIT
        assert name_grant.name_grant_refusal("gh pr view 1") is None


class TestLogSafety:
    """A refusal's log text must not carry the command line or a resolved path."""

    def test_log_text_is_a_module_constant(self, world):
        system_dir, user_dir = world
        _program(system_dir, "head")
        shim = _program(user_dir, "head")
        refusal = name_grant.name_grant_refusal("head /etc/hosts")
        assert refusal is not None
        assert refusal.log_text in name_grant._REFUSAL_LOG_TEXT.values()
        # The path IS in the detail (the person deciding needs it) and must NOT
        # be in the text that reaches a log sink.
        assert shim in refusal.detail
        assert shim not in refusal.log_text
        assert "/etc/hosts" not in refusal.log_text

    def test_every_code_has_log_text(self):
        codes = {
            name_grant.UNENUMERABLE,
            name_grant.UNTOKENIZABLE,
            name_grant.EXPANDED,
            name_grant.RELATIVE_PATH,
            name_grant.AGENT_TREE,
            name_grant.SHADOWED,
            name_grant.IDENTITY_CHANGED,
            name_grant.UNWITNESSED,
            name_grant.UNINSPECTABLE,
        }
        assert codes == set(name_grant._REFUSAL_LOG_TEXT)


class TestHookTierWiring:
    """The shell auto-approve tiers consult the check, and fall through on a refusal."""

    def test_read_only_tier_auto_approves_a_clean_name(self, world):
        system_dir, _ = world
        _program(system_dir, "ls")
        result = HookManager().on_tool_call("list", command="ls -la", is_shell=True)
        assert result.action == TOOL_AUTO_APPROVE

    def test_read_only_tier_falls_through_on_a_shadowed_name(self, world):
        system_dir, user_dir = world
        _program(system_dir, "ls")
        _program(user_dir, "ls")
        result = HookManager().on_tool_call("list", command="ls -la", is_shell=True)
        # ALLOW, not DENY: the command is not blocked, it just no longer skips
        # the interactive approval card.
        assert result.action == TOOL_ALLOW

    def test_configured_glob_tier_falls_through_on_a_shadowed_name(self, world):
        system_dir, user_dir = world
        _program(system_dir, "head")
        _program(user_dir, "head")
        cfg = HooksConfig(auto_approve_tools=["Running: head *"])
        mgr = HookManager(cfg)
        assert (
            mgr.on_tool_call("Running: head x", command="head x", is_shell=True).action
            == TOOL_ALLOW
        )

    def test_configured_glob_tier_still_auto_approves_a_clean_name(self, world):
        system_dir, _ = world
        _program(system_dir, "head")
        cfg = HooksConfig(auto_approve_tools=["Running: head *"])
        mgr = HookManager(cfg)
        assert (
            mgr.on_tool_call("Running: head x", command="head x", is_shell=True).action
            == TOOL_AUTO_APPROVE
        )

    def test_non_shell_tools_are_untouched(self, world):
        # The check reads a shell command line; an MCP tool has none, and its
        # tiers must keep working exactly as before.
        cfg = HooksConfig(auto_approve_tools=["ReadFile"])
        assert HookManager(cfg).on_tool_call("ReadFile").action == TOOL_AUTO_APPROVE
