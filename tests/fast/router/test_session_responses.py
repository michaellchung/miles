"""``POST /sessions/{session_id}/v1/responses``: the OpenAI Responses API (Codex) on the session path.

A Responses call must be recorded exactly like the equivalent chat call (same
record path, TITO input ids, sampled tokens and log-probabilities), and a full
replay of the previous turn (Codex sends ``store: false`` with the whole
history) must extend the session's trajectory instead of starting over.
"""

import json
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import requests
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session import responses_adapter as ra
from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

_TOOL_CALL_TEXT = '<tool_call>\n{"name": "exec_command", "arguments": {"cmd": "ls"}}\n</tool_call>'
_TOOLS = [
    {"type": "function", "name": "exec_command", "description": "run a shell command",
     "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}},
    {"type": "namespace", "name": "subagents", "tools": []},  # Codex sends these; the policy is not offered them
]


def _process_fn(prompt: str) -> ProcessResult:
    if "file-a" in prompt:
        return ProcessResult(text="done", finish_reason="stop")
    if "list files" in prompt:
        return ProcessResult(text=_TOOL_CALL_TEXT, finish_reason="stop")
    return ProcessResult(text="hello", finish_reason="stop")


_V2_ARGS = {
    "use_session_server": "v2",
    "session_sample_picker_path": "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
    "session_sample_postprocessor_path": "miles.rollout.session.v2.postprocessor_hub.default_postprocess",
}


@contextmanager
def _env(extra: dict | None = None):
    with with_mock_server(process_fn=_process_fn) as backend:
        config = make_session_server_config(
            backend_url=backend.url, timeout=30, hf_checkpoint="Qwen/Qwen3-0.6B",
            apply_chat_template_kwargs={"enable_thinking": False}, tito_model="default",
            instance_id=uuid.uuid4().hex, **({"pause_generation_mode": "retract"} | (extra or {})),
        )
        port = find_available_port(31000)
        server = UvicornThreadServer(SessionServer(config).app, host="127.0.0.1", port=port)
        server.start()
        try:
            yield SimpleNamespace(url=f"http://127.0.0.1:{port}", backend=backend)
        finally:
            server.stop()


@pytest.fixture(scope="module", params=["v1", "v2"])
def env(request):
    with _env(_V2_ARGS if request.param == "v2" else None) as e:
        yield e


def _session(url: str) -> str:
    return requests.post(f"{url}/sessions", timeout=5.0).json()["session_id"]


def _post(url: str, sid: str, payload: dict) -> requests.Response:
    return requests.post(f"{url}/sessions/{sid}/v1/responses", json=payload, timeout=30.0)


def _records(url: str, sid: str) -> list[dict]:
    return requests.get(f"{url}/sessions/{sid}", timeout=5.0).json()["records"]


def _codex_request(items: list[dict], **extra) -> dict:
    return {"model": "policy", "instructions": "You are Codex.", "input": items, "tools": _TOOLS,
            "tool_choice": "auto", "parallel_tool_calls": False, "store": False, "stream": False, **extra}


def _user(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _sse(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.split("\n\n"):
        if block.strip():
            event, data = block.split("\n", 1)
            out.append((event[len("event: "):], json.loads(data[len("data: "):])))
    return out


class TestTranslation:
    def test_codex_history_becomes_chat_messages(self):
        chat = ra.to_chat_request(_codex_request([
            _user("list files"),
            {"type": "function_call", "call_id": "call_1", "name": "exec_command", "arguments": '{"cmd":"ls"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "file-a"},
        ], max_output_tokens=64, temperature=1.0))
        assert chat["messages"] == [
            {"role": "system", "content": "You are Codex."},
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "exec_command", "arguments": '{"cmd":"ls"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "file-a"},
        ]
        assert [t["function"]["name"] for t in chat["tools"]] == ["exec_command"]  # namespace left out
        assert chat["stream"] is False and chat["max_tokens"] == 64 and chat["temperature"] == 1.0

    @pytest.mark.parametrize("bad", [
        {"previous_response_id": "resp_1"},
        {"input": []},
        {"input": [{"type": "compaction"}]},
        {"tools": [{"type": "custom", "name": "x"}]},
    ])
    def test_unsupported_shapes_are_refused(self, bad):
        with pytest.raises(ra.ResponsesRequestError):
            ra.to_chat_request({**_codex_request([_user("hi")]), **bad})

    def test_completion_renders_as_response_items_and_events(self):
        completion = {"id": "chatcmpl-1", "created": 1, "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                                                                     "total_tokens": 7},
                      "choices": [{"finish_reason": "tool_calls", "message": {
                          "role": "assistant", "content": None, "tool_calls": [
                              {"id": "call_9", "type": "function",
                               "function": {"name": "exec_command", "arguments": '{"cmd":"ls"}'}}]}}]}
        response = ra.to_response(completion, _codex_request([_user("x")]))
        assert response["status"] == "completed" and response["usage"]["output_tokens"] == 2
        assert response["output"] == [{"type": "function_call", "id": "fc_call_9", "call_id": "call_9",
                                       "status": "completed", "name": "exec_command", "arguments": '{"cmd":"ls"}'}]
        events = [e for e, _ in _sse(ra.to_sse(response).decode())]
        assert events == ["response.created", "response.output_item.added", "response.function_call_arguments.delta",
                          "response.function_call_arguments.done", "response.output_item.done", "response.completed"]

    def test_length_cut_is_incomplete(self):
        completion = {"id": "c", "choices": [{"finish_reason": "length", "message": {"content": "par"}}]}
        response = ra.to_response(completion, {})
        assert response["status"] == "incomplete" and response["incomplete_details"] == {"reason": "max_output_tokens"}
        assert _sse(ra.to_sse(response).decode())[-1][0] == "response.incomplete"


class TestRoute:
    def test_responses_call_is_recorded_like_a_chat_call(self, env):
        sid = _session(env.url)
        response = _post(env.url, sid, _codex_request([_user("say hi")]))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "response" and body["output"][0]["content"][0]["text"] == "hello"
        (record,) = _records(env.url, sid)
        assert record["path"] == "/v1/chat/completions" and record["status_code"] == 200
        choice = record["response"]["choices"][0]
        assert choice["meta_info"]["output_token_logprobs"]  # sampled tokens and log-probs kept for training

    def test_streamed_tool_call_then_full_replay_extends_the_trajectory(self, env):
        sid = _session(env.url)
        first = _post(env.url, sid, _codex_request([_user("list files")], stream=True))
        assert first.status_code == 200 and first.headers["content-type"].startswith("text/event-stream")
        events = _sse(first.text)
        assert events[0][0] == "response.created" and events[-1][0] == "response.completed"
        (call,) = [d["item"] for e, d in events if e == "response.output_item.done"]
        assert call["type"] == "function_call" and call["name"] == "exec_command"

        # Codex replays the whole history with store:false.
        second = _post(env.url, sid, _codex_request([
            _user("list files"),
            {"type": "function_call", "call_id": call["call_id"], "name": call["name"], "arguments": call["arguments"]},
            {"type": "function_call_output", "call_id": call["call_id"], "output": "file-a"},
        ]))
        assert second.status_code == 200, second.text
        assert second.json()["output"][0]["content"][0]["text"] == "done"
        records = _records(env.url, sid)
        assert len(records) == 2
        # The second prompt starts with the first prompt plus the first completion: one trajectory.
        first_ids = records[0]["request"]["input_ids"]
        first_out = [t for _, t in records[0]["response"]["choices"][0]["meta_info"]["output_token_logprobs"]]
        assert records[1]["request"]["input_ids"][: len(first_ids) + len(first_out)] == first_ids + first_out

    def test_bad_request_is_a_400_in_openai_error_shape_and_not_recorded(self, env):
        sid = _session(env.url)
        response = _post(env.url, sid, _codex_request([_user("x")], previous_response_id="resp_1"))
        assert response.status_code == 400 and "previous_response_id" in response.json()["error"]["message"]
        assert _records(env.url, sid) == []

    def test_unknown_session_is_reported(self, env):
        response = _post(env.url, "no-such-session", _codex_request([_user("x")]))
        assert response.status_code == 404 and "error" in response.json()
