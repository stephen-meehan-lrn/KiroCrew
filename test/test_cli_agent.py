"""Unit tests for the ``kirocrew agent`` CLI subcommand group.

Tests cover list output format, create with defaults, create duplicate,
update non-existent, and delete default agent.
"""

from __future__ import annotations

import json
import unittest.mock
from pathlib import Path

import pytest

from kiro_crew.cli import main


def _write_config(tmp_path: Path, data: dict) -> Path:
    """Write a config.json to *tmp_path* and return the path."""
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def _base_config() -> dict:
    """Return a minimal valid config with a default agent."""
    return {
        "agents": {
            "default": {
                "kiro_agent": "kirocrew",
                "workspace": "default",
                "memory_store": "default",
            },
        },
        "default_agent": "default",
        "workspaces": {"default": {"dir": "workspace"}},
        "memory_stores": {"default": {}},
    }


class TestAgentList:
    """Test ``kirocrew agent list`` output format."""

    def test_list_output_format(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cfg_path = _write_config(tmp_path, _base_config())

        with (
            unittest.mock.patch("kiro_crew.config.loader.config_path", return_value=cfg_path),
            unittest.mock.patch("sys.argv", ["kirocrew", "agent", "list"]),
        ):
            main()

        out = capsys.readouterr().out
        # Header row
        assert "NAME" in out
        assert "KIRO_AGENT" in out
        assert "WORKSPACE" in out
        assert "MEMORY_STORE" in out
        # Default agent marked with *
        assert "default *" in out or "default*" in out

    def test_list_multiple_agents(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        data = _base_config()
        data["agents"]["oncall"] = {
            "kiro_agent": "oncall-agent",
            "workspace": "oncall-ws",
            "memory_store": "oncall-mem",
        }
        cfg_path = _write_config(tmp_path, data)

        with (
            unittest.mock.patch("kiro_crew.config.loader.config_path", return_value=cfg_path),
            unittest.mock.patch("sys.argv", ["kirocrew", "agent", "list"]),
        ):
            main()

        out = capsys.readouterr().out
        assert "oncall" in out
        assert "oncall-agent" in out


class TestAgentCreate:
    """Test ``kirocrew agent create``."""

    def test_create_with_defaults(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cfg_path = _write_config(tmp_path, _base_config())

        with (
            unittest.mock.patch("kiro_crew.config.loader.config_path", return_value=cfg_path),
            unittest.mock.patch(
                "sys.argv",
                ["kirocrew", "agent", "create", "--name", "research"],
            ),
        ):
            main()

        out = capsys.readouterr().out
        assert "Created agent: research" in out

        # Verify persisted to disk
        saved = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert "research" in saved["agents"]
        assert saved["agents"]["research"]["kiro_agent"] == "kirocrew"
        assert saved["agents"]["research"]["workspace"] == "default"
        assert saved["agents"]["research"]["memory_store"] == "default"

    def test_create_duplicate_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg_path = _write_config(tmp_path, _base_config())

        with (
            unittest.mock.patch("kiro_crew.config.loader.config_path", return_value=cfg_path),
            unittest.mock.patch(
                "sys.argv",
                ["kirocrew", "agent", "create", "--name", "default"],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code != 0
        err = capsys.readouterr().err
        assert "already exists" in err


class TestAgentUpdate:
    """Test ``kirocrew agent update``."""

    def test_update_nonexistent_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg_path = _write_config(tmp_path, _base_config())

        with (
            unittest.mock.patch("kiro_crew.config.loader.config_path", return_value=cfg_path),
            unittest.mock.patch(
                "sys.argv",
                ["kirocrew", "agent", "update", "nonexistent", "--kiro-agent", "x"],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code != 0
        err = capsys.readouterr().err
        assert "not found" in err


class TestAgentDelete:
    """Test ``kirocrew agent delete``."""

    def test_delete_default_agent_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg_path = _write_config(tmp_path, _base_config())

        with (
            unittest.mock.patch("kiro_crew.config.loader.config_path", return_value=cfg_path),
            unittest.mock.patch(
                "sys.argv",
                ["kirocrew", "agent", "delete", "default"],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code != 0
        err = capsys.readouterr().err
        assert "cannot delete default agent" in err


class TestAgentResetModel:
    """``kirocrew agent reset-model`` -- the explicit way back to the default.

    Goes through ``main()`` rather than calling the handler directly: the verb
    lives on the SAME ``agent`` subparser group as list/create/update/delete, and
    a second group registered under the same name raises
    ``conflicting subparser: agent`` at parser-build time, which breaks the whole
    CLI while a handler-level test still passes. Routing through main() is what
    pins the wiring.
    """

    def _isolated_spec(self, tmp_path: Path, monkeypatch, model: str | None) -> Path:
        """Install a kiro spec inside a throwaway KIRO_HOME.

        The suite pins KIROCREW_HOME per test but deliberately NOT KIRO_HOME, so
        ``kiro_agents_dir()`` resolves the operator's real ``~/.kiro`` unless a
        test isolates it. The containment assert makes a lapse fail loudly rather
        than overwrite a live agent spec.
        """
        from kiro_crew.config.paths import kiro_agents_dir

        monkeypatch.setenv("KIRO_HOME", str(tmp_path / "kiro-home"))
        agents_dir = kiro_agents_dir()
        assert agents_dir.is_relative_to(tmp_path), f"{agents_dir} escaped {tmp_path}"
        agents_dir.mkdir(parents=True, exist_ok=True)
        body: dict = {"name": "kirocrew", "tools": ["fs_read"]}
        if model is not None:
            body["model"] = model
        spec = agents_dir / "kirocrew.json"
        spec.write_text(json.dumps(body), encoding="utf-8")
        return spec

    def test_reset_clears_the_pin_and_keeps_the_rest_of_the_spec(
        self, tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from kiro_crew import agent_state

        spec = self._isolated_spec(tmp_path, monkeypatch, "claude-opus-4.8")

        with unittest.mock.patch("sys.argv", ["kirocrew", "agent", "reset-model"]):
            main()

        written = json.loads(spec.read_text(encoding="utf-8"))
        assert "model" not in written
        assert written["tools"] == ["fs_read"], "reset must not regenerate the whole spec"
        assert agent_state.get_model_managed("kirocrew") is True
        out = capsys.readouterr().out
        assert "claude-opus-4.8" in out, "the cleared value is reported so it can be re-pinned"

    def test_reset_reports_a_spec_that_had_no_pin(
        self, tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._isolated_spec(tmp_path, monkeypatch, None)

        with unittest.mock.patch("sys.argv", ["kirocrew", "agent", "reset-model"]):
            main()

        assert "had no pinned model" in capsys.readouterr().out

    def test_reset_exits_nonzero_for_an_unknown_agent(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        self._isolated_spec(tmp_path, monkeypatch, "claude-opus-4.8")

        with unittest.mock.patch(
            "sys.argv", ["kirocrew", "agent", "reset-model", "--agent", "nope"]
        ):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 1

    def test_the_existing_agent_verbs_still_route(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Ratchet for the conflicting-subparser class: adding reset-model must
        not shadow or displace the group's original verbs."""
        cfg_path = _write_config(tmp_path, _base_config())

        with (
            unittest.mock.patch("kiro_crew.config.loader.config_path", return_value=cfg_path),
            unittest.mock.patch("sys.argv", ["kirocrew", "agent", "list"]),
        ):
            main()

        assert "KIRO_AGENT" in capsys.readouterr().out
