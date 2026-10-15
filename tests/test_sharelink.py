"""Share link encoding, decoding and the browser opener."""

from __future__ import annotations

import base64
import json
import subprocess
import threading
import uuid
import zlib
from typing import Any

import pytest
from conftest import WORKFLOWS

from mcp_solveit_workflows import sharelink
from mcp_solveit_workflows.sharelink import (
    BUILDER_ORIGIN,
    MAX_HASH_LENGTH,
    MAX_INFLATED_BYTES,
    SHARE_LINK_MAX_CHARS,
    SHARE_PREFIX,
    ShareLinkError,
    build_url,
    decode_link,
    encode_hash,
    open_in_browser,
    share_link,
)


def compact(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def raw_deflate(data: bytes) -> bytes:
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    return c.compress(data) + c.flush()


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def big_workflow(nodes: int) -> dict[str, Any]:
    """A workflow whose text barely compresses, so the link grows with node count."""
    return {
        "version": 2,
        "title": "Generated",
        "nodes": [
            {
                "id": str(uuid.uuid4()),
                "type": "technique",
                "x": i * 37,
                "y": i * 53,
                "data": {"label": str(uuid.uuid4()), "note": str(uuid.uuid4())},
            }
            for i in range(nodes)
        ],
        "edges": [],
    }


# ---- encoding ----


def test_encode_hash_format() -> None:
    h = encode_hash('{"version":2}')
    assert h.startswith("~")
    assert "=" not in h
    assert set(h[1:]) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def test_build_url_prefix() -> None:
    url = build_url('{"version":2}')
    assert url.startswith(SHARE_PREFIX)
    assert url == BUILDER_ORIGIN + "#" + encode_hash('{"version":2}')


def test_round_trip_non_ascii() -> None:
    obj = {"version": 2, "title": "Écrit — 日本語 ✓", "nodes": [], "edges": []}
    s = compact(obj)
    assert decode_link(build_url(s)) == json.loads(s)


def test_small_fixture_link_round_trips() -> None:
    obj = json.loads((WORKFLOWS / "synthetic_21_nodes.json").read_text(encoding="utf-8"))
    s = compact(obj)
    result = share_link(s)
    assert result.url is not None
    assert result.reason is None
    assert result.length == len(result.url) <= SHARE_LINK_MAX_CHARS
    assert decode_link(result.url) == obj


def test_63_node_fixture_link_length() -> None:
    obj = json.loads((WORKFLOWS / "synthetic_63_nodes.json").read_text(encoding="utf-8"))
    result = share_link(compact(obj))
    assert result.url is not None
    assert 2700 <= result.length <= 3300  # 3,012; the drift check confirms the builder agrees
    assert decode_link(result.url) == obj


def test_large_workflow_has_no_link_with_reason() -> None:
    s = compact(big_workflow(150))
    result = share_link(s)
    assert result.url is None
    assert result.length > SHARE_LINK_MAX_CHARS
    assert result.reason is not None
    assert str(result.length) in result.reason
    assert "8000" in result.reason
    # The full URL is still buildable for the opener, within the builder's hash cap.
    full = build_url(s)
    assert len(full) == result.length
    assert len(full) - len(BUILDER_ORIGIN) - 1 <= MAX_HASH_LENGTH


@pytest.mark.parametrize("length,fits", [(8000, True), (8001, False), (7999, True)])
def test_limit_boundary(monkeypatch: pytest.MonkeyPatch, length: int, fits: bool) -> None:
    fake = SHARE_PREFIX + "A" * (length - len(SHARE_PREFIX))
    assert len(fake) == length
    monkeypatch.setattr(sharelink, "build_url", lambda _s: fake)
    result = share_link("{}")
    assert result.length == length
    assert (result.url is not None) is fits
    assert (result.reason is None) is fits


# ---- decoding ----


def test_decode_accepts_bare_hash_forms() -> None:
    s = compact({"version": 2, "nodes": []})
    h = encode_hash(s)
    assert decode_link(h) == {"version": 2, "nodes": []}
    assert decode_link("#" + h) == {"version": 2, "nodes": []}
    assert decode_link("https://workflows.hargs.co.uk/some/path?q=1#" + h) == {
        "version": 2,
        "nodes": [],
    }
    assert decode_link("  " + BUILDER_ORIGIN + "#" + h + "\n") == {"version": 2, "nodes": []}


def test_legacy_lzstring_link_refused() -> None:
    with pytest.raises(ShareLinkError, match="fresh share link"):
        decode_link(BUILDER_ORIGIN + "#N4IgLglmA2CmIC4QGEBOBDAzgSwHZQAJMBjA")


def test_fetch_link_refused() -> None:
    with pytest.raises(ShareLinkError, match="(?i)fetch") as exc:
        decode_link(BUILDER_ORIGIN + "#fetch=https://example.com/wf.json")
    assert "export" in str(exc.value).lower()


def test_over_long_hash_refused() -> None:
    with pytest.raises(ShareLinkError, match="200000|200,000") as exc:
        decode_link("~" + "A" * MAX_HASH_LENGTH)
    assert "AAAAAAAA" not in str(exc.value)


def test_deflate_bomb_refused() -> None:
    payload = b'{"a":"' + b"x" * (MAX_INFLATED_BYTES + 1000) + b'"}'
    h = "~" + b64url(raw_deflate(payload))
    assert len(h) <= MAX_HASH_LENGTH
    with pytest.raises(ShareLinkError, match="4000000|4,000,000|4 MB"):
        decode_link(h)


def test_inflated_exactly_at_limit_allowed() -> None:
    body = b"x" * (MAX_INFLATED_BYTES - len(b'{"a":""}'))
    payload = b'{"a":"' + body + b'"}'
    assert len(payload) == MAX_INFLATED_BYTES
    assert decode_link("~" + b64url(raw_deflate(payload))) == {"a": body.decode()}


@pytest.mark.parametrize(
    "hash_",
    [
        "~",  # empty payload
        "~!!!notbase64",  # bad alphabet
        "~" + b64url(b"not deflate data at all"),
        "~" + b64url(raw_deflate(b"{not json")),
        "~" + b64url(raw_deflate(b"\xff\xfe bad utf8")),
    ],
)
def test_bad_payload_refused(hash_: str) -> None:
    with pytest.raises(ShareLinkError):
        decode_link(hash_)


def test_non_object_refused() -> None:
    with pytest.raises(ShareLinkError, match="object"):
        decode_link(encode_hash("[1,2,3]"))


def test_error_does_not_echo_payload() -> None:
    secret = "SECRETVALUE123"
    h = "~" + b64url(raw_deflate(f'["{secret}"]'.encode()))
    with pytest.raises(ShareLinkError) as exc:
        decode_link(h)
    assert secret not in str(exc.value)
    assert h[1:20] not in str(exc.value)


def test_empty_link_refused() -> None:
    with pytest.raises(ShareLinkError):
        decode_link("")
    with pytest.raises(ShareLinkError):
        decode_link(BUILDER_ORIGIN)


# ---- malformed links ----


@pytest.mark.parametrize("body", ["A", "AAAAA", "abc=", "ab+c", "ab/c"])
def test_malformed_base64url_refused(body: str) -> None:
    with pytest.raises(ShareLinkError):
        decode_link("~" + body)


def test_truncated_deflate_refused() -> None:
    full = raw_deflate(compact({"version": 2, "nodes": [], "pad": "x" * 500}).encode())
    with pytest.raises(ShareLinkError, match="truncated|deflate"):
        decode_link("~" + b64url(full[: len(full) // 2]))


# ---- opener ----


class FakeProcess:
    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode
        self.killed = False
        self.waits: list[float | None] = []

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        if self.returncode is None:
            if timeout is None:
                # The background reaper's wait: the process exits eventually.
                return 0
            raise subprocess.TimeoutExpired(["x"], timeout)
        return self.returncode

    def kill(self) -> None:  # pragma: no cover - must never be called
        self.killed = True


class Launcher:
    """Stands in for subprocess.Popen: records argv and returns a fake process."""

    def __init__(self, returncode: int | None = 0, exc: BaseException | None = None) -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.exc = exc
        self.processes: list[FakeProcess] = []

    def __call__(self, argv: list[str]) -> FakeProcess:
        self.calls.append(argv)
        if self.exc is not None:
            raise self.exc
        proc = FakeProcess(self.returncode)
        self.processes.append(proc)
        return proc


def test_opener_darwin_uses_open_argv() -> None:
    url = build_url(compact(big_workflow(150)))
    launcher = Launcher()
    assert open_in_browser(url, launcher=launcher, platform="darwin") == (True, None)
    assert launcher.calls == [["open", url]]
    assert launcher.calls[0][1].startswith("https://workflows.hargs.co.uk/#~")
    assert launcher.processes[0].waits and launcher.processes[0].waits[0]


def test_opener_linux_uses_xdg_open() -> None:
    url = build_url("{}")
    launcher = Launcher()
    ok, reason = open_in_browser(
        url, launcher=launcher, platform="linux", which=lambda _n: "/usr/bin/xdg-open"
    )
    assert (ok, reason) == (True, None)
    assert launcher.calls[0] == ["xdg-open", url]


def test_opener_still_running_counts_as_launched_and_is_not_killed() -> None:
    launcher = Launcher(returncode=None)
    ok, reason = open_in_browser(
        build_url("{}"), launcher=launcher, platform="linux", which=lambda _n: "/usr/bin/xdg-open"
    )
    assert (ok, reason) == (True, None)
    assert not launcher.processes[0].killed


def test_default_launcher_detaches_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_popen(argv: list[str], **kwargs: Any) -> FakeProcess:
        seen["argv"] = argv
        seen.update(kwargs)
        return FakeProcess(0)

    monkeypatch.setattr(sharelink.subprocess, "Popen", fake_popen)
    url = build_url("{}")
    ok, _ = open_in_browser(url, platform="linux", which=lambda _n: "/usr/bin/xdg-open")
    assert ok
    assert seen["argv"] == ["xdg-open", url]
    assert not seen.get("shell", False)
    assert seen["start_new_session"] is True
    for stream in ("stdin", "stdout", "stderr"):
        assert seen[stream] == subprocess.DEVNULL


def test_opener_linux_without_xdg_open() -> None:
    launcher = Launcher()
    ok, reason = open_in_browser(
        build_url("{}"), launcher=launcher, platform="linux", which=lambda _n: None
    )
    assert not ok
    assert reason is not None and "xdg-open" in reason
    assert launcher.calls == []


def test_opener_windows_not_supported() -> None:
    launcher = Launcher()
    ok, reason = open_in_browser(build_url("{}"), launcher=launcher, platform="win32")
    assert not ok
    assert reason is not None and "macOS and Linux" in reason
    assert launcher.calls == []


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.com/#~abc",
        "http://workflows.hargs.co.uk/#~abc",
        "https://workflows.hargs.co.uk.evil.com/#~abc",
        "https://workflows.hargs.co.uk/#N4Ig",
        "file:///etc/passwd",
        "; rm -rf ~",
    ],
)
def test_opener_refuses_non_builder_url(url: str) -> None:
    launcher = Launcher()
    ok, reason = open_in_browser(url, launcher=launcher, platform="darwin")
    assert not ok and reason
    assert launcher.calls == []


def test_opener_refuses_over_long_hash() -> None:
    launcher = Launcher()
    url = SHARE_PREFIX + "A" * MAX_HASH_LENGTH  # hash is "~" + 200000 chars
    ok, _ = open_in_browser(url, launcher=launcher, platform="darwin")
    assert not ok
    assert launcher.calls == []
    ok, _ = open_in_browser(
        SHARE_PREFIX + "A" * (MAX_HASH_LENGTH - 1), launcher=launcher, platform="darwin"
    )
    assert ok


def test_opener_nonzero_exit() -> None:
    ok, reason = open_in_browser(
        build_url("{}"), launcher=Launcher(returncode=1), platform="darwin"
    )
    assert not ok and reason and "status 1" in reason


@pytest.mark.parametrize("exc", [OSError("no such file"), FileNotFoundError(2, "No such file")])
def test_opener_launch_failure(exc: BaseException) -> None:
    ok, reason = open_in_browser(build_url("{}"), launcher=Launcher(exc=exc), platform="darwin")
    assert not ok and reason


class SlowProcess:
    """Still running at the opener's wait, exits later: must be reaped, not killed."""

    def __init__(self) -> None:
        self.waits: list[float | None] = []
        self.reaped = threading.Event()

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        if timeout is not None:
            raise subprocess.TimeoutExpired(["x"], timeout)
        self.reaped.set()
        return 0


def test_opener_reaps_a_process_that_outlives_the_wait() -> None:
    proc = SlowProcess()
    ok, _ = open_in_browser(
        build_url("{}"),
        launcher=lambda _argv: proc,
        platform="linux",
        which=lambda _n: "/usr/bin/xdg-open",
    )
    assert ok is True
    assert proc.reaped.wait(timeout=2)
