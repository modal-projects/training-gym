"""What the dashboard clients put on the wire when proxy auth is configured.

The reporters are exercised against a real local HTTP server rather than a
mocked ``urlopen``: the properties worth protecting here — which headers
leave the process, and where they are allowed to go — are only observable
from the receiving end.
"""

from __future__ import annotations

import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from modal_training_gym.common import config, deployment, proxy_auth, status_reporter
from modal_training_gym.common.launcher_helpers import redact_env_values
from modal_training_gym.common.proxy_auth import modal_proxy_auth_headers
from modal_training_gym.frameworks.slime import reporting

# Both reporters post to the dashboard and both had to be taught the pair;
# patching only the common one would silently drop slime's rollouts and
# advantage distributions, which is invisible until a run reports nothing.
POSTERS = {
    "status_reporter": lambda url: status_reporter._post(
        {"_url": url, "_timeout": 5.0, "_token": "run-token", "phase": "training"}
    ),
    "slime_reporting": lambda url: reporting._post(
        {"_url": url, "_timeout": 5.0, "phase": "training"}
    ),
}


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch, tmp_path):
    """No pair in the environment, and never the developer's real config file."""
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "training-gym.toml")
    monkeypatch.setenv("TRAINING_GYM_FRAMEWORK_STATUS_TOKEN", "run-token")
    # load_proxy_auth() populates os.environ by design, so monkeypatch cannot
    # undo it; restore both variables by hand.
    original = {name: os.environ.get(name) for name in ("MODAL_KEY", "MODAL_SECRET")}
    for name in original:
        monkeypatch.delenv(name, raising=False)
    yield
    for name, value in original.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture
def pair(monkeypatch):
    monkeypatch.setenv("MODAL_KEY", "wk-key")
    monkeypatch.setenv("MODAL_SECRET", "ws-secret")
    # Named because the egress guard withholds the pair from non-Modal hosts.
    monkeypatch.setenv(proxy_auth.PROXY_AUTH_HOSTS_ENV, "127.0.0.1")


class _Handler(BaseHTTPRequestHandler):
    def _record(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        # urllib normalizes header casing on the wire (``Modal-Key`` arrives as
        # ``Modal-key``), so compare lower-cased.
        self.server.seen.append(
            {name.lower(): value for name, value in self.headers.items()}
        )
        status, headers = self.server.reply
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    # GET as well as POST: urllib turns a redirected POST into a GET, so a
    # handler that only recorded POSTs would see a credential leak as silence.
    do_POST = _record
    do_GET = _record

    def log_message(self, *_args):
        pass


def _serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.seen = []
    server.reply = (200, {})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/api/framework-status"


@pytest.fixture
def dashboard():
    server, url = _serve()
    try:
        yield server, url
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("post", POSTERS.values(), ids=POSTERS)
def test_proxy_pair_is_sent_alongside_the_run_token(dashboard, pair, post):
    """Edge auth is additional to per-run authorization, not a replacement:
    dropping the bearer would make every run's reports interchangeable."""
    server, url = dashboard

    post(url)

    assert server.seen[0]["authorization"] == "Bearer run-token"
    assert server.seen[0]["modal-key"] == "wk-key"
    assert server.seen[0]["modal-secret"] == "ws-secret"


@pytest.mark.parametrize("post", POSTERS.values(), ids=POSTERS)
def test_credentials_never_follow_a_redirect(dashboard, pair, post):
    """urllib's redirect handler replays the original headers at the new
    location, whatever origin it names — so a dashboard URL that redirects
    (or is made to) would hand the run token and the workspace pair to
    somewhere else entirely."""
    server, url = dashboard
    elsewhere, _ = _serve()
    try:
        elsewhere_url = f"http://127.0.0.1:{elsewhere.server_address[1]}/"
        server.reply = (302, {"Location": elsewhere_url})

        post(url)

        assert len(server.seen) == 1
        assert elsewhere.seen == []
    finally:
        elsewhere.shutdown()
        elsewhere.server_close()


@pytest.mark.parametrize("post", POSTERS.values(), ids=POSTERS)
@pytest.mark.parametrize("status", [401, 403])
def test_rejected_reports_warn_once_and_leak_nothing(
    dashboard, pair, capsys, monkeypatch, post, status
):
    """Reports are fire-and-forget, so a dashboard that started refusing them
    would otherwise erase observability in silence. Both codes matter and mean
    different things: 401 is Modal's edge turning away a missing pair, 403 is
    the app refusing the run token. The warning also depends on ``HTTPError``
    being caught before ``OSError`` — it is a subclass, so reordering those
    excepts turns this back into silence."""
    server, url = dashboard
    server.reply = (status, {})
    monkeypatch.setattr(proxy_auth, "_next_warn", {})

    post(url)
    post(url)

    err = capsys.readouterr().err
    assert err.count("dashboard rejected a report") == 1
    assert str(status) in err
    assert "wk-key" not in err and "ws-secret" not in err and "run-token" not in err


def test_warning_does_not_echo_a_signed_status_url(capsys, monkeypatch):
    """The URL is operator-supplied (TRAINING_GYM_FRAMEWORK_STATUS_URL), so it
    can carry userinfo or a signed query — and this warning goes to logs the
    whole team reads."""
    monkeypatch.setattr(proxy_auth, "_next_warn", {})

    proxy_auth.warn_auth_rejected(
        401, "https://user:pw@dash.test:8443/api/framework-status?sig=deadbeef"
    )

    err = capsys.readouterr().err
    assert "https://dash.test:8443/api/framework-status" in err
    assert "deadbeef" not in err and "pw" not in err


@pytest.mark.parametrize("half", ["MODAL_KEY", "MODAL_SECRET"])
def test_half_a_pair_is_no_pair(monkeypatch, half):
    """Sending one header without the other authenticates nothing and reads
    like a configuration success at the call site."""
    monkeypatch.setenv(half, "set")
    assert modal_proxy_auth_headers("https://dash--x.modal.run/api") == {}


def test_pair_falls_back_to_the_config_file():
    """How a laptop CLI run gets the pair; containers only ever see the env."""
    config.save_proxy_auth("wk-from-toml", "ws-from-toml")

    assert modal_proxy_auth_headers("https://dash--x.modal.run/api") == {
        "Modal-Key": "wk-from-toml",
        "Modal-Secret": "ws-from-toml",
    }


@pytest.mark.parametrize(
    "url,allowed",
    [
        ("https://dash--x.modal.run/api", True),
        ("https://dash.us-east.modal.direct/api", True),
        ("http://dash--x.modal.run/api", False),  # never over plaintext
        ("https://evilmodal.run/api", False),  # lookalike, not a subdomain
        ("https://dashboard.internal.test/api", False),
        ("not a url", False),
    ],
)
def test_pair_egress_guard(url, allowed):
    """The URL is operator-supplied config, so a typo or poisoned value must
    fail toward withholding the workspace credential."""
    assert proxy_auth.proxy_auth_url_allowed(url) is allowed


def test_allowlisted_host_releases_the_pair(monkeypatch):
    monkeypatch.setenv(proxy_auth.PROXY_AUTH_HOSTS_ENV, "Dash.Corp.test , other.test")

    assert proxy_auth.proxy_auth_url_allowed("https://dash.corp.test/api")
    assert not proxy_auth.proxy_auth_url_allowed("https://third.test/api")
    # Allowlisting a host says where the pair may go, not that it may go in
    # the clear -- the same rule the Modal hostnames get above.
    assert not proxy_auth.proxy_auth_url_allowed("http://dash.corp.test/api")


def test_configured_pair_is_withheld_from_non_modal_urls(pair, monkeypatch):
    monkeypatch.delenv(proxy_auth.PROXY_AUTH_HOSTS_ENV, raising=False)

    assert modal_proxy_auth_headers("https://dashboard.internal.test/api") == {}


def test_rejection_warning_explains_a_withheld_pair(pair, capsys, monkeypatch):
    """ "Set MODAL_KEY" would gaslight an operator whose configured pair was
    withheld by the egress guard; the warning must name the actual cause."""
    monkeypatch.delenv(proxy_auth.PROXY_AUTH_HOSTS_ENV, raising=False)
    monkeypatch.setattr(proxy_auth, "_next_warn", {})

    proxy_auth.warn_auth_rejected(401, "https://dashboard.internal.test/api")

    assert proxy_auth.PROXY_AUTH_HOSTS_ENV in capsys.readouterr().err


def test_warning_survives_a_malformed_dashboard_url(pair, capsys, monkeypatch):
    """The warning runs inside the reporter's ``except HTTPError``, and the
    worker loop has no ``except``: an exception here would unwind the thread and
    end reporting for the rest of the run. ``SplitResult.port`` raises lazily
    for a non-numeric port, which ``urlsplit`` itself accepts."""
    monkeypatch.setattr(proxy_auth, "_next_warn", {})

    proxy_auth.warn_auth_rejected(401, "https://dash.test:abc/api/framework-status")

    assert "<unparseable url>" in capsys.readouterr().err


def test_serve_401_hint_explains_a_withheld_pair(pair, monkeypatch):
    """Same property on the serving client's 401 hint: an endpoint on a
    custom domain must not be told its configured pair is "not set"."""
    monkeypatch.delenv(proxy_auth.PROXY_AUTH_HOSTS_ENV, raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        deployment._raise_for_proxy_auth(401, "https://serve.internal.test/v1")

    message = str(exc_info.value)
    assert proxy_auth.PROXY_AUTH_HOSTS_ENV in message
    assert "not set" not in message


@pytest.mark.parametrize("post", POSTERS.values(), ids=POSTERS)
def test_refused_redirects_warn_instead_of_silent_loss(
    dashboard, pair, capsys, monkeypatch, post
):
    """Refusing a 3xx protects the credentials but drops every report for the
    run — that misconfiguration must be loud, not silent."""
    server, url = dashboard
    server.reply = (302, {"Location": "https://elsewhere.test/"})
    monkeypatch.setattr(proxy_auth, "_next_warn", {})

    post(url)
    post(url)

    assert capsys.readouterr().err.count("302") == 1


def test_config_file_holding_the_pair_is_owner_only():
    """It stores a workspace credential; 0644 would expose it to every local
    account, and the file predates this so existing ones must be tightened."""
    config.CONFIG_PATH.write_text('[dashboard]\nurl = "https://old.test"\n')
    os.chmod(config.CONFIG_PATH, 0o644)

    config.save_proxy_auth("wk-abc", "ws-def")

    assert stat.S_IMODE(os.stat(config.CONFIG_PATH).st_mode) == 0o600
    assert config.get_dashboard_url() == "https://old.test"
    assert list(config.CONFIG_PATH.parent.glob("*.tmp")) == []


def test_concurrent_writers_cannot_clobber_each_other(monkeypatch):
    """Two writers overlap for real: ``training-gym setup`` beside a deploy, or
    two parallel deploys both calling save_dashboard_url.

    Interleaving is forced rather than raced — a second write is driven to
    completion from inside the first one's ``os.replace``. With a shared
    temp-file name the second writer renames the first one's file away, so the
    first ``os.replace`` raises FileNotFoundError and its config is lost.
    """
    real_replace = os.replace
    nested_done = []

    def replace_once(src, dst):
        if not nested_done:
            nested_done.append(True)
            config.save_dashboard_url("https://second-writer.test")
        return real_replace(src, dst)

    # No monkeypatch.undo() here: pytest hands the whole test one MonkeyPatch
    # instance, so undoing would also revert the autouse fixture's CONFIG_PATH
    # and point these assertions at the developer's real config file.
    monkeypatch.setattr(config.os, "replace", replace_once)
    config.save_proxy_auth("wk-abc", "ws-def")

    assert nested_done, "the nested writer never ran; the test proves nothing"
    # Last writer wins, but neither may corrupt the file or lose the section it
    # wrote, and no temp file may survive.
    assert config.get_proxy_auth() == ("wk-abc", "ws-def")
    assert stat.S_IMODE(os.stat(config.CONFIG_PATH).st_mode) == 0o600
    assert list(config.CONFIG_PATH.parent.glob("*.tmp")) == []


def test_runtime_env_echo_masks_credentials():
    """Launchers print the Ray runtime_env, which carries the run's status
    token, into logs the whole team can read."""
    redacted = redact_env_values(
        {
            "TRAINING_GYM_FRAMEWORK_STATUS_TOKEN": "tok",
            "MODAL_SECRET": "ws-1",
            "WANDB_API_KEY": "k",
            "MASTER_ADDR": "10.0.0.1",
            "WANDB_RUN_ID": "run-7",
        }
    )

    assert redacted == {
        "TRAINING_GYM_FRAMEWORK_STATUS_TOKEN": "***",
        "MODAL_SECRET": "***",
        "WANDB_API_KEY": "***",
        "MASTER_ADDR": "10.0.0.1",
        "WANDB_RUN_ID": "run-7",
    }
