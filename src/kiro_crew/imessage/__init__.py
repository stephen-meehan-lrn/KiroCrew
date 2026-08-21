"""iMessage channel — a local-only chat surface on the user's own Mac.

Unlike every other channel, iMessage needs no third-party bot registration and
no credential: the transport is the user's own Messages.app, and the identity
that talks to the agent is their own handle. That is the design constraint, not
a convenience — a hosted relay would put a third party in the message path of a
channel whose entire value is that it never leaves the user's machine.

The bridge to Messages.app is the external ``imsg`` CLI in its long-lived
``rpc`` mode: a child process spoken to over newline-framed JSON-RPC 2.0 on
stdio, the same shape as a language server. No daemon, no port, no webhook, and
therefore no new inbound network surface.

v1 is DM-only, text-only, and requires the gateway to run on the Messages host.
"""

from __future__ import annotations
