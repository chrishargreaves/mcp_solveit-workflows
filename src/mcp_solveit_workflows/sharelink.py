"""Share links for the Workflow Builder, and opening them in a browser.

The encoding matches the deployed builder (``src/state/urlState.ts``): the
compact JSON is compressed with raw deflate, base64url-encoded without padding
and prefixed with ``~``. Links are only ever built for workflows.hargs.co.uk.

Decoding treats a link as untrusted input: the hash length and the inflated
size are capped at the builder's own limits, and error messages never quote
the payload.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import shutil
import subprocess
import sys
import threading
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

BUILDER_ORIGIN = "https://workflows.hargs.co.uk/"
DEFLATE_SCHEME = "~"
SHARE_PREFIX = BUILDER_ORIGIN + "#" + DEFLATE_SCHEME

# The builder warns above this whole-URL length; email and chat clients start
# breaking links around here. A URL of exactly this length is allowed.
SHARE_LINK_MAX_CHARS = 8000
# The builder's caps on a hash it will decode, and on the inflated JSON.
MAX_HASH_LENGTH = 200_000
MAX_INFLATED_BYTES = 4_000_000

# How long the opener is given to report a failure before it counts as launched.
OPEN_WAIT_SECONDS = 3.0
_INFLATE_CHUNK = 64 * 1024
_B64URL = re.compile(r"[A-Za-z0-9_-]+")


class ShareLinkError(Exception):
    """A share link could not be decoded. The message never quotes the payload."""


@dataclass(frozen=True)
class LinkResult:
    url: str | None
    length: int
    reason: str | None


def encode_hash(compact_json: str) -> str:
    """Encode compact workflow JSON as a builder hash (without the leading ``#``)."""
    compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
    deflated = compressor.compress(compact_json.encode("utf-8")) + compressor.flush()
    return DEFLATE_SCHEME + base64.urlsafe_b64encode(deflated).decode("ascii").rstrip("=")


def build_url(compact_json: str) -> str:
    """The full builder URL, with no length cap (the opener uses this)."""
    return BUILDER_ORIGIN + "#" + encode_hash(compact_json)


def share_link(compact_json: str) -> LinkResult:
    """A share link, or a reason there is none when the URL is over the limit."""
    return link_for_url(build_url(compact_json))


def link_for_url(url: str) -> LinkResult:
    """`share_link` for a URL already made by `build_url`."""
    length = len(url)
    if length <= SHARE_LINK_MAX_CHARS:
        return LinkResult(url=url, length=length, reason=None)
    reason = (
        f"No share link: the link would be {length} characters, over the "
        f"{SHARE_LINK_MAX_CHARS}-character limit that email and chat reliably carry. "
        "The saved file still works (open it in the builder), and the open flag can "
        "still show it in a browser on this machine."
    )
    return LinkResult(url=None, length=length, reason=reason)


def _extract_hash(link: str) -> str:
    text = link.strip()
    if "#" in text:
        text = text.split("#", 1)[1]
    elif "://" in text:
        raise ShareLinkError("The link has no workflow in it (nothing after '#').")
    return text


def _too_big() -> ShareLinkError:
    return ShareLinkError(
        f"The link's workflow inflates to more than {MAX_INFLATED_BYTES:,} bytes, "
        "the builder's limit."
    )


def _inflate_capped(data: bytes) -> bytes:
    inflater = zlib.decompressobj(-15)
    out = bytearray()
    pending = data
    try:
        while not inflater.eof:
            # Ask for at most one byte past the cap so a deflate bomb stops early.
            room = MAX_INFLATED_BYTES + 1 - len(out)
            chunk = inflater.decompress(pending, min(_INFLATE_CHUNK, room))
            out += chunk
            if len(out) > MAX_INFLATED_BYTES:
                raise _too_big()
            pending = inflater.unconsumed_tail
            if not chunk and not pending:
                break
    except zlib.error:
        raise ShareLinkError("The link's data is not valid deflate data.") from None
    if not inflater.eof:
        raise ShareLinkError("The link's data is truncated or not valid deflate data.")
    return bytes(out)


def decode_link(link: str) -> dict[str, Any]:
    """Decode a builder share link (full URL or bare hash) to the workflow object.

    Version and structure checks are left to the model layer.
    """
    raw = _extract_hash(link)
    if not raw:
        raise ShareLinkError("The link has no workflow in it (nothing after '#').")
    if raw.startswith("fetch="):
        raise ShareLinkError(
            "Builder links that fetch a workflow from another address are not supported. "
            "Export the workflow file from the builder and use that instead."
        )
    if len(raw) > MAX_HASH_LENGTH:
        raise ShareLinkError(
            f"The link is {len(raw):,} characters, over the builder's "
            f"{MAX_HASH_LENGTH:,} character limit."
        )
    if not raw.startswith(DEFLATE_SCHEME):
        raise ShareLinkError(
            "This is an older-format builder link, which is not supported. Open it in the "
            "builder and copy a fresh share link, or export the workflow file."
        )
    body = raw[len(DEFLATE_SCHEME) :]
    if not body:
        raise ShareLinkError("The link has no workflow data after '~'.")
    if _B64URL.fullmatch(body) is None:
        raise ShareLinkError("The link's data is not valid base64url.")
    try:
        data = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
    except (binascii.Error, ValueError):
        raise ShareLinkError("The link's data is not valid base64url.") from None
    inflated = _inflate_capped(data)
    try:
        parsed = json.loads(inflated.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ShareLinkError("The link's data is not valid JSON.") from None
    if not isinstance(parsed, dict):
        raise ShareLinkError("The link's data is not a workflow (expected a JSON object).")
    return parsed


class _Process(Protocol):
    def wait(self, timeout: float | None = None) -> int: ...


Launcher = Callable[[list[str]], _Process]


def _launch_detached(argv: list[str]) -> _Process:
    """Start the opener in its own session, with no shell and no inherited stdio.

    The MCP server speaks over stdio, so the child must not hold or write to it;
    a new session keeps it alive (and out of our signals) if the server exits.
    """
    return subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def open_in_browser(
    url: str,
    *,
    launcher: Launcher = _launch_detached,
    platform: str = sys.platform,
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[bool, str | None]:
    """Open a server-built builder URL in the default browser, without a shell.

    The opener (``open`` on macOS, ``xdg-open`` on Linux) is given up to
    ``OPEN_WAIT_SECONDS`` to fail. Some ``xdg-open`` back ends stay running
    as long as the browser does, so a process still running then counts as
    launched and is left alone.

    Returns ``(True, None)`` on success, otherwise ``(False, reason)``.
    """
    if not url.startswith(SHARE_PREFIX):
        return False, "not opened: only workflows.hargs.co.uk share links are opened"
    if len(url) - len(BUILDER_ORIGIN) - 1 > MAX_HASH_LENGTH:
        return False, (
            f"not opened: the workflow is too large for the builder's "
            f"{MAX_HASH_LENGTH:,} character link limit"
        )
    if platform == "darwin":
        argv = ["open", url]
    elif platform.startswith("linux"):
        if which("xdg-open") is None:
            return False, "not opened: no xdg-open found to launch a browser"
        argv = ["xdg-open", url]
    else:
        return False, "not opened: opening a browser is supported on macOS and Linux only"
    try:
        proc = launcher(argv)
    except OSError as exc:
        return False, f"not opened: could not run {argv[0]} ({exc.strerror or type(exc).__name__})"
    try:
        returncode = proc.wait(timeout=OPEN_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        # Still running (xdg-open can wait on the browser): it launched. Reap it in the
        # background when it exits, so it never lingers as a zombie, and never kill it.
        threading.Thread(target=proc.wait, daemon=True).start()
        return True, None
    if returncode != 0:
        return False, f"not opened: {argv[0]} exited with status {returncode}"
    return True, None
