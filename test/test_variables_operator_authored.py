"""Only the operator's own dashboard text may have ``{{name}}`` expanded.

``_run_chat`` is the dashboard turn engine, but it is reached from ~22 call sites,
including the Slack linked-thread route, which hands it a channel participant's raw
message. While the expansion gate keyed only on ``is_slash`` and ``_prompt_depth``
-- both satisfied by an ordinary inbound message -- a participant could send
``{{NAME}}`` and read operator config back off the thread.

The transport ratchet in ``test_variables_channels.py`` could not catch this: the
expansion is not IN a transport module, it is in the dashboard engine the transport
calls. So this file guards the OTHER axis -- who is allowed to ask for expansion.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import kiro_crew.dashboard.chat_runner as runner_mod

SRC = pathlib.Path(runner_mod.__file__).resolve().parents[1]

# The only call sites permitted to pass operator_authored=True. Each is text the
# operator typed into the dashboard themselves. Adding an entry here is a security
# claim and should be argued in review, which is the point of pinning the set.
ALLOWED_OPT_IN = {
    "dashboard/chat_handlers.py",  # the composer POST
    "dashboard/chat_regenerate.py",  # regenerate + edit-resend of the operator's row
    "dashboard/chat_rewind.py",  # rewind replay of the operator's row
}


def _call_sites() -> list[tuple[str, int, bool]]:
    """Every ``_run_chat(...)`` call in the package, with whether it opts in."""
    found: list[tuple[str, int, bool]] = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = str(path.relative_to(SRC))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name != "_run_chat":
                continue
            # cli_chat defines an unrelated sync _run_chat(message, model, agent).
            if rel.startswith("cli_chat.py"):
                continue
            opts_in = any(
                kw.arg == "operator_authored"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )
            found.append((rel, node.lineno, opts_in))
    return found


class TestExpansionIsOptIn:
    def test_the_parameter_defaults_to_false(self):
        """The default is the security property: a new caller does not expand."""
        param = inspect.signature(runner_mod._run_chat).parameters["operator_authored"]
        assert param.default is False, "expansion must be opt-in, never opt-out"
        assert (
            param.kind is inspect.Parameter.KEYWORD_ONLY
        ), "keyword-only so it can never be set by accident from position"

    def test_the_gate_requires_it(self):
        """Source guard: the expansion call must sit behind the new conjunct.

        Anchored on the conjunct rather than the whole expression, because the two
        older conjuncts have each been legitimately extended before.
        """
        src = inspect.getsource(runner_mod._run_chat)
        gate = [ln for ln in src.splitlines() if "_expand_message_variables(message" in ln]
        assert len(gate) == 1, f"expected one expansion site, found {len(gate)}"
        assert (
            "if operator_authored and" in src
        ), "the expansion site is no longer gated on operator_authored"

    def test_only_allowlisted_call_sites_opt_in(self):
        """The enumeration. A new opt-in outside the allowlist fails here."""
        sites = _call_sites()
        assert sites, "found no _run_chat call sites; the walker is broken"
        offenders = sorted(
            {rel for rel, _lineno, opts_in in sites if opts_in and rel not in ALLOWED_OPT_IN}
        )
        assert not offenders, (
            "these call sites claim operator-authored text without being allowlisted: "
            f"{offenders}. If the claim is genuine, add it to ALLOWED_OPT_IN with a "
            "reason; if the text can come from a channel participant, it must not expand."
        )

    def test_the_slack_linked_thread_route_does_not_opt_in(self):
        """The specific regression. This caller passes a participant's raw text."""
        sites = _call_sites()
        slack_sites = [(rel, ln, opt) for rel, ln, opt in sites if rel == "slack/handler.py"]
        assert slack_sites, (
            "slack/handler.py no longer calls _run_chat; if the linked-thread route "
            "moved, re-point this test at its new home rather than deleting it"
        )
        for rel, lineno, opts_in in slack_sites:
            assert not opts_in, (
                f"{rel}:{lineno} opts into variable expansion, but the linked-thread "
                "route forwards a channel participant's message verbatim"
            )

    def test_the_composer_does_opt_in(self):
        """Positive control. Without this the suite would pass on a feature that
        never expands anything at all -- which is exactly what an absence-only
        assertion cannot distinguish."""
        sites = _call_sites()
        composer = [
            (rel, ln) for rel, ln, opt in sites if rel == "dashboard/chat_handlers.py" and opt
        ]
        assert composer, (
            "the dashboard composer no longer opts in, so no operator text expands "
            "and the feature is inert"
        )
