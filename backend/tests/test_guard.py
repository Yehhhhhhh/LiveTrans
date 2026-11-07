import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import server
from guard import host_allowed, origin_allowed, rejection

LOCAL = "127.0.0.1:8765"


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1:8765", "localhost:8765", "LOCALHOST", "[::1]:8765", "192.168.1.20:8765", "app.localhost:8765"],
)
def test_local_and_ip_hosts_are_allowed(host):
    assert host_allowed(host)


@pytest.mark.parametrize("host", ["evil.example:8765", "evil.example", "", "127.0.0.1.evil.example"])
def test_domain_hosts_are_refused(host):
    # DNS rebinding arrives with the attacker's domain as Host.
    assert not host_allowed(host)


def test_allowed_hosts_setting_permits_a_name():
    assert host_allowed("my-pc:8765", frozenset({"my-pc"}))


def test_origin_must_be_this_server():
    assert origin_allowed(None, LOCAL)  # scripts, curl, non-browser clients
    assert origin_allowed("http://127.0.0.1:8765", LOCAL)
    assert not origin_allowed("http://127.0.0.1:9999", LOCAL)
    assert not origin_allowed("http://localhost:8765", LOCAL)  # different origin than the page
    assert not origin_allowed("https://evil.example", LOCAL)
    assert not origin_allowed("null", LOCAL)  # sandboxed iframes, file:// pages


def test_rejection_reason():
    assert rejection(LOCAL, "http://127.0.0.1:8765") is None
    assert "host" in rejection("evil.example:8765", None)
    assert "origin" in rejection(LOCAL, "https://evil.example")


@pytest.fixture
def client(monkeypatch):
    # No lifespan: the routes used here don't need the models.
    return TestClient(server.app, base_url="http://127.0.0.1:8765")


def test_http_same_origin_and_scripts_pass(client):
    assert client.get("/api/status").status_code == 200
    assert client.get("/api/status", headers={"Origin": "http://127.0.0.1:8765"}).status_code == 200


def test_http_cross_site_is_refused(client):
    r = client.post("/api/translate", json={}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_http_rebinding_host_is_refused():
    c = TestClient(server.app, base_url="http://evil.example:8765")
    assert c.get("/api/status").status_code == 403
    assert c.get("/").status_code == 403


def test_websocket_from_another_site_is_refused(client, caplog):
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("ws://127.0.0.1:8765/ws", headers={"Origin": "https://evil.example"}):
            pass
    assert e.value.code == 1008
    assert "origin 'https://evil.example'" in caplog.text  # refused for the Origin, not the Host


def test_websocket_rebinding_host_is_refused(client, caplog):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("ws://evil.example:8765/ws"):
            pass
    assert "host 'evil.example:8765'" in caplog.text
