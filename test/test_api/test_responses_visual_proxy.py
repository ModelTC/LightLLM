import asyncio

import lightllm.server.api_responses as api_responses


def test_responses_with_image_uses_visual_proxy(monkeypatch):
    called = {}

    async def main_handler(_request, _raw_request):
        raise AssertionError("image-bearing Responses request must not bypass the visual proxy")

    async def fake_visual_proxy(**kwargs):
        from fastapi.responses import Response

        called["request"] = kwargs["request"]
        called["main_chat_handler"] = kwargs["main_chat_handler"]
        return Response("proxied")

    class FakeRequestSlot:
        async def __aenter__(self):
            return None

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeRuntime:
        settings = type("Settings", (), {"thinking_policy": "request"})()

        def request_slot(self):
            return FakeRequestSlot()

    class FakeRequest:
        async def json(self):
            return {
                "model": "test-model",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,YWJjZA==",
                            },
                            {"type": "input_text", "text": "What is shown?"},
                        ],
                    }
                ],
            }

    import lightllm.server.api_http as api_http
    import lightllm.server.api_openai as api_openai

    monkeypatch.setattr(api_responses, "visual_chat_completions_impl", fake_visual_proxy)
    monkeypatch.setattr(api_openai, "chat_completions_impl", main_handler)
    monkeypatch.setattr(api_http.g_objs, "args", type("Args", (), {"visual_remote_url": "https://vision.test"})())
    monkeypatch.setattr(api_http.g_objs, "visual_proxy_runtime", FakeRuntime())

    response = asyncio.run(api_responses.responses_impl(FakeRequest()))

    assert response.body == b"proxied"
    assert called["main_chat_handler"] is main_handler
    assert called["request"].messages[0].content[0].type == "image_url"
