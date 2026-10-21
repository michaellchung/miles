"""Failure log line for session model calls: ids, route, status, reason; never request content."""

import logging

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from miles.rollout.session.errors import SessionError, UpstreamGenerationAbortedError
from miles.rollout.session.sessions import install_session_failure_logging, note_session_failure

_SECRET_PROMPT = "secret-prompt-text"


def _app() -> FastAPI:
    app = FastAPI()
    install_session_failure_logging(app)

    @app.exception_handler(SessionError)
    async def handler(request: Request, exc: SessionError):
        note_session_failure(request, type(exc).__name__)
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat(session_id: str):
        raise UpstreamGenerationAbortedError("upstream generation aborted before completion")

    @app.post("/sessions/{session_id}/v1/responses")
    async def responses(request: Request, session_id: str):
        note_session_failure(request, "response_conversion_failed", tokens_recorded=True)
        return JSONResponse(status_code=500, content={})

    @app.post("/sessions/{session_id}/v1/messages")
    async def messages(session_id: str):
        return JSONResponse(status_code=400, content={"error": "context too long"})  # upstream pass-through

    @app.post("/sessions/{session_id}/samples")
    async def samples(session_id: str):
        return JSONResponse(status_code=422, content={})

    return app


@pytest.fixture
def client():
    return TestClient(_app())


def _failure_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "model call failed" in r.getMessage()]


def test_aborted_chat_call_logs_503_with_reason_and_no_tokens(client, caplog):
    caplog.set_level(logging.WARNING, logger="miles.rollout.session.sessions")
    response = client.post("/sessions/s-1/v1/chat/completions", json={"messages": [{"content": _SECRET_PROMPT}]})
    assert response.status_code == 503
    assert _failure_lines(caplog) == [
        "[session] model call failed: session_id=s-1 route=chat status=503 "
        "reason=UpstreamGenerationAbortedError tokens_recorded=False"
    ]
    assert _SECRET_PROMPT not in caplog.text


def test_post_commit_failure_reports_recorded_tokens(client, caplog):
    caplog.set_level(logging.WARNING, logger="miles.rollout.session.sessions")
    assert client.post("/sessions/s-2/v1/responses", json={}).status_code == 500
    assert _failure_lines(caplog) == [
        "[session] model call failed: session_id=s-2 route=responses status=500 "
        "reason=response_conversion_failed tokens_recorded=True"
    ]


def test_pass_through_status_has_default_reason(client, caplog):
    caplog.set_level(logging.WARNING, logger="miles.rollout.session.sessions")
    assert client.post("/sessions/s-3/v1/messages", json={}).status_code == 400
    assert _failure_lines(caplog) == [
        "[session] model call failed: session_id=s-3 route=messages status=400 "
        "reason=upstream_status tokens_recorded=False"
    ]


def test_non_model_routes_are_not_logged(client, caplog):
    caplog.set_level(logging.WARNING, logger="miles.rollout.session.sessions")
    assert client.post("/sessions/s-4/samples").status_code == 422
    assert _failure_lines(caplog) == []
