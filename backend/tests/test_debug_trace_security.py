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
from app.services.debug_trace_service import DebugTraceRecorder


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


def test_oversized_debug_trace_preserves_failed_draft_and_validation():
    class Store:
        def upsert_debug_trace(self, **kwargs):
            self.snapshot = kwargs

    store = Store()
    recorder = DebugTraceRecorder(store=store, run_id="run", enabled=True, max_bytes=8192)
    validation = {
        "valid": False, "reason": "unsupported_metric_values",
        "checked_claims": [{"metric": "f1", "value": "0.8"}] * 500,
        "supported_claims": [{"metric": "f1", "value": "0.8"}] * 500,
        "claim_evidence": [{"source_claim": {"metric": "f1", "value": "0.8"}}] * 500,
        "numeric_assessments": [{"status": "supported", "value": "0.8"}] * 500,
        "unsupported_claims": [{"metric": "ccc", "value": "0.5", "text": "wrong owner"}],
    }
    recorder.payload["generations"] = [{"input": {"final_prompt": "private password=supersecret123"},
        "attempts": [{"kind": "initial", "draft": "Alpha CCC A was 0.5.", "validation": validation}]}]
    recorder.flush()

    saved = store.snapshot
    payload = saved["payload"]
    assert saved["size_bytes"] <= 8192
    assert saved["truncated"] is True
    assert payload["capture"]["compacted"] is True
    attempt = payload["generations"][0]["attempts"][0]
    assert attempt["draft"] == "Alpha CCC A was 0.5."
    assert attempt["validation"]["reason"] == "unsupported_metric_values"
    assert attempt["validation"]["unsupported_claims"] == validation["unsupported_claims"]
    assert attempt["validation"]["trace_omissions"]["supported_claims"] == 500
    assert "supersecret123" not in json.dumps(payload)
    assert len(recorder.payload["generations"][0]["attempts"][0]["validation"]["supported_claims"]) == 500


def test_trace_compaction_keeps_contradicted_numeric_assessments():
    class Store:
        def upsert_debug_trace(self, **kwargs):
            self.snapshot = kwargs

    store = Store()
    recorder = DebugTraceRecorder(store=store, run_id="run", enabled=True, max_bytes=4096)
    failed = {"status": "contradicted", "value": "5", "reason": "context_conflict"}
    recorder.payload["generations"] = [{"attempts": [{"draft": "Beta count was 5.", "validation": {
        "valid": False, "numeric_assessments": [{"status": "supported", "value": "2"}] * 500 + [failed],
    }}]}]
    recorder.flush()
    saved = store.snapshot["payload"]["generations"][0]["attempts"][0]["validation"]
    assert saved["numeric_assessments"] == [failed]
    assert saved["trace_omissions"]["numeric_assessments"] == 500


def test_proof_repetition_is_compacted_when_bounder_clips_draft_before_hard_cap():
    class Store:
        def upsert_debug_trace(self, **kwargs):
            self.snapshot = kwargs

    store = Store()
    recorder = DebugTraceRecorder(store=store, run_id="run", enabled=True, max_bytes=10_000)
    draft = "Start of answer. " + "Owner and measurement details. " * 180 + " End of answer."
    proofs = [{"text": "verified source quote " * 9}] * 20
    recorder.payload["generations"] = [{"attempts": [{"draft": draft, "validation": {
        "valid": False, "checked_claims": proofs, "supported_claims": proofs,
        "unsupported_claims": [{"text": "Keep this failure."}],
    }}]}]
    recorder.flush()
    saved = store.snapshot
    assert saved["size_bytes"] <= 10_000
    assert saved["payload"]["generations"][0]["attempts"][0]["draft"] == draft
    assert saved["payload"]["capture"]["compacted"] is True


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
