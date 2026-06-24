from fastapi.testclient import TestClient


def test_config_server_emits_access_log(monkeypatch):
    from lightllm.server.config_server import api_http

    messages = []
    monkeypatch.setattr(api_http.logger, "info", lambda msg, *args, **kwargs: messages.append(str(msg)))

    with TestClient(api_http.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert any("GET /health 200" in message for message in messages)
