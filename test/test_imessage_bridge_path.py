"""Where the bridge executable may come from.

The point of these is not that resolution works -- it is that resolution is NOT
configurable. ``config.json`` is agent-writable, so a settable path would let an
auto-approved agent shell choose which binary the gateway executes.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from kiro_crew.config.loader import IMessageConfig
from kiro_crew.imessage import bridge_path as bp


class TestResolution:
    def test_a_binary_on_path_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bp.shutil, "which", lambda _n: "/somewhere/bin/imsg")
        assert bp.resolve_bridge_path() == "/somewhere/bin/imsg"

    def test_a_trusted_location_is_used_when_path_misses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The launch-agent case: a minimal PATH with no Homebrew prefix, which is
        # exactly what an absolute-path override used to be needed for.
        monkeypatch.setattr(bp.shutil, "which", lambda _n: None)
        installed = tmp_path / "imsg"
        installed.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setattr(bp, "TRUSTED_BRIDGE_PATHS", (str(installed),))
        assert bp.resolve_bridge_path() == str(installed)

    def test_not_installed_resolves_to_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(bp.shutil, "which", lambda _n: None)
        monkeypatch.setattr(bp, "TRUSTED_BRIDGE_PATHS", (str(tmp_path / "absent"),))
        assert bp.resolve_bridge_path() == ""

    def test_an_unstattable_candidate_is_skipped_not_fatal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(bp.shutil, "which", lambda _n: None)
        good = tmp_path / "imsg"
        good.write_text("#!/bin/sh\n", encoding="utf-8")

        real_is_file = Path.is_file

        def flaky(self: Path) -> Any:
            if self.name == "boom":
                raise OSError("permission denied")
            return real_is_file(self)

        monkeypatch.setattr(Path, "is_file", flaky)
        monkeypatch.setattr(bp, "TRUSTED_BRIDGE_PATHS", (str(tmp_path / "boom"), str(good)))
        assert bp.resolve_bridge_path() == str(good)

    def test_the_trusted_list_is_absolute(self) -> None:
        # A relative entry would resolve against the gateway's working directory,
        # which is not a location anybody audited.
        #
        # Asserted through PurePosixPath, not Path: these are POSIX paths by
        # definition (the channel is macOS-only), and on a Windows test runner
        # `WindowsPath("/opt/...").is_absolute()` is False for want of a drive
        # letter -- which would fail the check for the wrong reason.
        for candidate in bp.TRUSTED_BRIDGE_PATHS:
            assert PurePosixPath(candidate).is_absolute(), candidate


class TestNotConfigurable:
    def test_the_config_section_carries_no_executable_path(self) -> None:
        """The whole point of the fix: no field selects the binary.

        A reachable arbitrary-exec path is the finding this closes, so this
        asserts the ABSENCE of the field rather than any behaviour -- an
        innocent-looking re-addition would otherwise pass every other test.
        """
        fields = set(vars(IMessageConfig()))
        assert "cli_path" not in fields
        for name in fields:
            assert "exec" not in name and "binary" not in name, name
