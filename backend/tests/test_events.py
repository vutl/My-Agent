import json

from app.core.events import sse_event


def test_sse_event_formats_named_event_with_json_payload() -> None:
    raw = sse_event("message.delta", {"delta": "hello"})

    assert raw.startswith("event: message.delta\n")
    assert raw.endswith("\n\n")

    data_line = raw.splitlines()[1]
    assert data_line.startswith("data: ")
    assert json.loads(data_line.removeprefix("data: ")) == {"delta": "hello"}
