from types import SimpleNamespace

from evaluation.codememo import eval as codememo_eval


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_api_call_uses_legacy_chat_args_for_gpt4o():
    client = _FakeClient()

    result = codememo_eval._api_call_with_retry(
        client,
        [{"role": "user", "content": "hi"}],
        max_tokens=77,
        retries=1,
        model="gpt-4o-mini",
    )

    assert result == "ok"
    call = client.chat.completions.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["max_tokens"] == 77
    assert call["temperature"] == 0.0
    assert call["seed"] == 42
    assert "max_completion_tokens" not in call


def test_api_call_uses_gpt5_completion_arg_shape():
    client = _FakeClient()

    result = codememo_eval._api_call_with_retry(
        client,
        [{"role": "user", "content": "hi"}],
        max_tokens=88,
        retries=1,
        model="gpt-5-mini",
    )

    assert result == "ok"
    call = client.chat.completions.calls[0]
    assert call["model"] == "gpt-5-mini"
    assert call["max_completion_tokens"] == 88
    assert "max_tokens" not in call
    assert "temperature" not in call
    assert "seed" not in call
