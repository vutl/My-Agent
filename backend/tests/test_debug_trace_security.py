import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.responses import Response
from starlette.requests import Request

from app.api.agent import (
    AgentRunRequest,
    get_agent_run_debug_trace,
    run_agent_stream,
)
from app.core.config import Settings
from app.core.network import request_client_is_loopback
from app.core.redaction import redact_and_bound


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("password=supersecret123", "supersecret123"),
        ('password: "super secret 123"', "super secret 123"),
        ('config={"api_key":"supersecret123"}', "supersecret123"),
        ("X-API-Key: supersecret123", "supersecret123"),
        ("OPENAI_API_KEY=supersecret123", "supersecret123"),
        ("client_secret: supersecret123", "supersecret123"),
        ("token=supersecret123", "supersecret123"),
        ("AWS_SECRET_ACCESS_KEY=supersecret123", "supersecret123"),
        (
            "Authorization: Basic dXNlcjpwYXNzd29yZA==",
            "dXNlcjpwYXNzd29yZA==",
        ),
        (
            "github_pat_11AA22BB33CC44DD55EE66FF77GG88HH",
            "github_pat_11AA22BB33CC44DD55EE66FF77GG88HH",
        ),
        (
            "ghr_11AA22BB33CC44DD55EE66FF77GG88HH",
            "ghr_11AA22BB33CC44DD55EE66FF77GG88HH",
        ),
        ("ASIA1234567890ABCDEF", "ASIA1234567890ABCDEF"),
    ],
)
def test_redaction_covers_secrets_embedded_in_prompt_strings(
    text: str,
    secret: str,
) -> None:
    sanitized, replacements, _, _ = redact_and_bound(
        {"final_prompt": text},
        max_bytes=8192,
    )

    assert secret not in sanitized["final_prompt"]
    assert "[REDACTED_SECRET]" in sanitized["final_prompt"]
    assert replacements >= 1


def test_redaction_normalizes_structured_secret_keys_without_hiding_metrics() -> None:
    payload = {
        "x-api-key": "x-secret-value",
        "client_secret": "client-secret-value",
        "oauth_access_token": "access-token-value",
        "token": "generic-token-value",
        "AWS_SECRET_ACCESS_KEY": "aws-secret-value",
        "clientSecret": "camel-client-secret",
        "accessToken": "camel-access-token",
        "secretKey": "camel-secret-key",
        "awsSecretAccessKey": "camel-aws-secret",
        "token_count": 123,
        "tokens_per_second": 45.6,
    }

    sanitized, replacements, _, _ = redact_and_bound(payload, max_bytes=8192)
    encoded = json.dumps(sanitized, ensure_ascii=False)

    for secret in (
        "x-secret-value",
        "client-secret-value",
        "access-token-value",
        "generic-token-value",
        "aws-secret-value",
        "camel-client-secret",
        "camel-access-token",
        "camel-secret-key",
        "camel-aws-secret",
    ):
        assert secret not in encoded
    assert sanitized["token_count"] == 123
    assert sanitized["tokens_per_second"] == 45.6
    assert replacements == 9


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.42.7.9", "::1", "::ffff:127.0.0.1", "localhost"],
)
def test_debug_trace_accepts_actual_loopback_clients(host: str) -> None:
    assert request_client_is_loopback(_request_from(host)) is True


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "192.168.1.20", "203.0.113.5", "testclient", "localhost.attacker.test"],
)
def test_debug_trace_rejects_non_loopback_and_unresolved_clients(host: str) -> None:
    assert request_client_is_loopback(_request_from(host)) is False


def test_debug_trace_fails_closed_when_asgi_client_is_missing() -> None:
    assert request_client_is_loopback(_request_from(None)) is False


def test_debug_trace_read_endpoint_rejects_remote_client_before_store_access() -> None:
    class Store:
        def get_debug_trace(self, run_id: str):
            raise AssertionError(f"store must not be read for remote run {run_id}")

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            get_agent_run_debug_trace(
                "run-1",
                _request_from("203.0.113.5"),
                Response(),
                Store(),
                Settings(agent_debug_trace_enabled=True),
            )
        )

    assert error.value.status_code == 404


def test_debug_trace_read_endpoint_allows_loopback_and_disables_caching() -> None:
    trace = {"run_id": "run-1", "payload": {"capture": {"redacted": True}}}

    class Store:
        def get_debug_trace(self, run_id: str):
            assert run_id == "run-1"
            return trace

    response = Response()
    result = asyncio.run(
        get_agent_run_debug_trace(
            "run-1",
            _request_from("127.0.0.1"),
            response,
            Store(),
            Settings(agent_debug_trace_enabled=True),
        )
    )

    assert result == trace
    assert response.headers["Cache-Control"] == "no-store"


def test_debug_trace_capture_endpoint_rejects_remote_client_before_agent_work() -> None:
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            run_agent_stream(
                request=AgentRunRequest(task="secret task", debug_trace=True),
                http_request=_request_from("203.0.113.5"),
                service=None,
                history=None,
                state_store=None,
                memory_store=None,
                runtime_gate=None,
                historical_search=None,
                long_term_memory=None,
                rag=None,
                run_store=None,
                query_rewriter=None,
                intent_router=None,
                settings=Settings(agent_debug_trace_enabled=True),
            )
        )

    assert error.value.status_code == 403


def _request_from(host: str | None) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": [],
            "server": ("127.0.0.1", 7777),
            "client": (host, 12345) if host is not None else None,
        }
    )
