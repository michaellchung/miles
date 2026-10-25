"""Codex (Responses API) on the session path with Nemotron-3 Nano: one trajectory, one sample.

The backend is a mock SGLang server that uses the real Nemotron-3 Nano
tokenizer and SGLang's real ``nemotron_3`` reasoning parser and
``qwen3_coder`` tool-call parser (the parsers ``--tito-model nemotron3``
selects).  The client replays the history the way Codex 0.145 does
(``store: false``; checked against the real ``codex exec`` binary): every
earlier output item comes back with its exact text, a reasoning item keeps
its ``reasoning_text`` content and gets ``encrypted_content: null``, and ids,
statuses and annotations are dropped.

What must hold for training:

* no rollback: every Responses call extends the stored trajectory, so the
  session records one turn per call;
* each prompt starts with the previous prompt plus its sampled tokens
  (TITO prefix reuse; the ``<|im_end|>\\n`` boundary is completed);
* the collected sample is the whole trajectory: loss mask 1 exactly on the
  sampled tokens of every turn, rollout log-probabilities for each of them.
"""

from __future__ import annotations

import copy
import json
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import requests
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.rollout.session.server import SessionServer
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, ProcessResult
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

NEMOTRON = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

sglang_parsers = pytest.importorskip("sglang.srt.function_call.function_call_parser")
from pydantic import TypeAdapter  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import Tool  # noqa: E402
from sglang.srt.function_call.function_call_parser import FunctionCallParser  # noqa: E402
from sglang.srt.parser.reasoning_parser import ReasoningParser  # noqa: E402

# Codex 0.145 tool shapes (a subset of what `codex exec` sends): function tools
# with nested JSON schemas, and a namespace tool the adapter leaves out.
CODEX_TOOLS = [
    {"type": "function", "name": "exec_command", "strict": False,
     "description": "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
     "parameters": {"type": "object", "additionalProperties": False, "required": ["cmd"], "properties": {
         "cmd": {"type": "string", "description": "Shell command to execute."},
         "workdir": {"type": "string", "description": "Working directory for the command."},
         "yield_time_ms": {"type": "number", "description": "Wait before yielding."},
         "tty": {"type": "boolean", "description": "True allocates a PTY."}}}},
    {"type": "function", "name": "update_plan", "strict": False, "description": "Updates the task plan.",
     "parameters": {"type": "object", "additionalProperties": False, "required": ["plan"], "properties": {
         "plan": {"type": "array", "items": {"type": "object", "properties": {
             "step": {"type": "string"}, "status": {"type": "string"}}}}}}},
    {"type": "namespace", "name": "multi_agent_v1", "tools": []},
]

# Raw generations after the Nemotron generation prompt (which already ends
# with "<think>\n"); the mock appends <|im_end|> as the stop token.
TURNS = [
    "The user wants the file list. I will run ls.\n</think>\nLet me look at the files.\n"
    "<tool_call>\n<function=exec_command>\n<parameter=cmd>\nls -la\n</parameter>\n"
    "<parameter=yield_time_ms>\n1000\n</parameter>\n</function>\n</tool_call>\n",
    "There is a README. Read it.\n</think>\n"
    "<tool_call>\n<function=exec_command>\n<parameter=cmd>\ncat README.md\n</parameter>\n</function>\n</tool_call>\n",
    "The README says hello.\n</think>\nThe directory holds README.md; it says hello.",
]


class NemotronMockServer(MockSGLangServer):
    """Mock SGLang chat endpoint with the Nemotron tokenizer and SGLang's real parsers."""

    def _compute_chat_completions_response(self, payload: dict) -> dict:
        tok = self.tokenizer
        prompt = tok.decode(payload["input_ids"], skip_special_tokens=False)
        text = TURNS[min(prompt.count("<tool_response>"), len(TURNS) - 1)]
        output_ids = tok.encode(text, add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
        reasoning, content = ReasoningParser(model_type="nemotron_3", stream_reasoning=False,
                                             force_reasoning=True).parse_non_stream(text)
        tool_calls, finish_reason = None, "stop"
        tools = payload.get("tools")
        if tools:
            parser = FunctionCallParser(tools=TypeAdapter(list[Tool]).validate_python(tools),
                                        tool_call_parser="qwen3_coder")
            if parser.has_tool_call(content):
                content, calls = parser.parse_non_stream(content)
                finish_reason = "tool_calls"
                tool_calls = [{"id": f"call_{uuid.uuid4().hex[:8]}", "index": call.tool_index, "type": "function",
                               "function": {"name": call.name, "arguments": call.parameters}} for call in calls]
        logprobs = [(-0.01 * (i + 1), tid) for i, tid in enumerate(output_ids)]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion", "created": 1, "model": "mock",
            "choices": [{
                "index": 0, "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content or None, "reasoning_content": reasoning,
                            "tool_calls": tool_calls},
                "logprobs": {"content": [{"token": tok.convert_ids_to_tokens(t), "token_id": t, "logprob": lp}
                                         for lp, t in logprobs]},
                "meta_info": {"output_token_logprobs": logprobs, "completion_tokens": len(output_ids)},
            }],
            "usage": {"prompt_tokens": len(payload["input_ids"]), "completion_tokens": len(output_ids),
                      "total_tokens": len(payload["input_ids"]) + len(output_ids)},
        }


@contextmanager
def _env():
    try:
        backend = NemotronMockServer(model_name=NEMOTRON, process_fn=lambda _: ProcessResult(text=""),
                                     host="127.0.0.1", port=find_available_port(32000))
    except OSError as exc:  # no network and no cached tokenizer
        pytest.skip(f"Nemotron-3 Nano tokenizer unavailable: {exc}")
    backend.start()
    config = make_session_server_config(backend_url=backend.url, timeout=30, hf_checkpoint=NEMOTRON,
                                        tito_model="nemotron3", instance_id=uuid.uuid4().hex)
    port = find_available_port(31200)
    server = UvicornThreadServer(SessionServer(config).app, host="127.0.0.1", port=port)
    server.start()
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{port}", tokenizer=backend.tokenizer)
    finally:
        server.stop()
        backend.stop()


@pytest.fixture(scope="module")
def env():
    with _env() as e:
        yield e


def _codex_request(items: list[dict]) -> dict:
    # The fields Codex 0.145 sends on every call (no max_output_tokens, reasoning {}).
    return {"model": "model", "instructions": "You are Codex, a coding agent.", "input": items,
            "tools": CODEX_TOOLS, "tool_choice": "auto", "parallel_tool_calls": False, "reasoning": {},
            "store": False, "stream": True, "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "k", "client_metadata": {"session_id": "s"}}


def _codex_replay(item: dict) -> dict:
    """An output item as Codex sends it back in the next request's input."""
    item = copy.deepcopy(item)
    for key in ("id", "status"):
        item.pop(key, None)
    if item["type"] == "reasoning":
        item["encrypted_content"] = None
    if item["type"] == "message":
        item["content"] = [{"type": c["type"], "text": c["text"]} for c in item["content"]]
    return item


def _sse_output(body: str) -> list[dict]:
    done = []
    for block in body.split("\n\n"):
        if block.startswith("event: response.output_item.done"):
            done.append(json.loads(block.split("\ndata: ", 1)[1])["item"])
    return done


def _run_codex(env) -> tuple[str, list[list[dict]]]:
    sid = requests.post(f"{env.url}/sessions", timeout=5.0).json()["session_id"]
    history = [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "<permissions/>"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context/>"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What is in this directory?"}]},
    ]
    outputs = []
    for turn in range(len(TURNS)):
        reply = requests.post(f"{env.url}/sessions/{sid}/v1/responses", json=_codex_request(history), timeout=30.0)
        assert reply.status_code == 200, reply.text
        items = _sse_output(reply.text)
        outputs.append(items)
        history += [_codex_replay(item) for item in items]
        for item in items:
            if item["type"] == "function_call":
                history.append({"type": "function_call_output", "call_id": item["call_id"],
                                "output": f"Process exited with code 0\nOutput:\nREADME.md (turn {turn})\n"})
    return sid, outputs


def test_codex_trajectory_is_one_growing_session(env):
    sid, outputs = _run_codex(env)
    # what the parsers gave back: reasoning, then text and/or one tool call per turn.
    # The text after </think> keeps its leading "\n" (also a lone "\n" before a
    # tool call); Codex replays it as it came, so the history still matches.
    assert [[item["type"] for item in items] for items in outputs] == [
        ["reasoning", "message", "function_call"], ["reasoning", "message", "function_call"],
        ["reasoning", "message"]]
    assert outputs[1][1]["content"][0]["text"] == "\n"
    assert json.loads(outputs[0][2]["arguments"]) == {"cmd": "ls -la", "yield_time_ms": 1000}

    records = requests.get(f"{env.url}/sessions/{sid}", timeout=5.0).json()["records"]
    assert len(records) == len(TURNS)  # no rollback: every call extended the stored trajectory
    newline = env.tokenizer.encode("\n", add_special_tokens=False)
    for before, after in zip(records, records[1:]):
        prompt = before["request"]["input_ids"]
        sampled = [t for _, t in before["response"]["choices"][0]["meta_info"]["output_token_logprobs"]]
        # TITO: the next prompt is the previous prompt, its sampled tokens, the
        # "\n" after <|im_end|>, then the new tool/user messages
        assert after["request"]["input_ids"][: len(prompt) + len(sampled) + 1] == prompt + sampled + newline
    # Re-rendering the assistant turns differs from the sampled text only by
    # the parser's trailing "\n" in reasoning_content (a diagnostic, Nemotron3TITOTokenizer
    # docstring); the training tokens are the sampled ones, and no
    # non-assistant segment differs.
    mismatch = requests.get(f"{env.url}/sessions/{sid}", timeout=5.0).json()["metadata"].get(
        "tito_session_mismatch") or []
    assert {m["type"] for m in mismatch} <= {"assistant_text"}


def test_collected_sample_trains_every_turn(env):
    sid, _ = _run_codex(env)
    records = requests.get(f"{env.url}/sessions/{sid}", timeout=5.0).json()["records"]
    reply = requests.post(f"{env.url}/sessions/{sid}/samples", json={"max_seq_len": None}, timeout=30.0)
    assert reply.status_code == 200, reply.text
    (sample,) = decode_samples_and_merge_input_sample(reply.content, Sample()).samples

    last = records[-1]
    last_sampled = [t for _, t in last["response"]["choices"][0]["meta_info"]["output_token_logprobs"]]
    assert sample.tokens == last["request"]["input_ids"] + last_sampled
    first_prompt = records[0]["request"]["input_ids"]
    assert sample.response_length == len(sample.tokens) - len(first_prompt)
    assert len(sample.loss_mask) == sample.response_length == len(sample.rollout_log_probs)

    # loss mask 1 exactly on the sampled tokens of the three turns, 0 on the
    # tool outputs and generation prompts in between
    expected_mask, expected_logprobs = [], []
    for i, record in enumerate(records):
        pairs = record["response"]["choices"][0]["meta_info"]["output_token_logprobs"]
        if i:
            gap = len(record["request"]["input_ids"]) - len(records[i - 1]["request"]["input_ids"]) - len(
                records[i - 1]["response"]["choices"][0]["meta_info"]["output_token_logprobs"])
            expected_mask += [0] * gap
            expected_logprobs += [0.0] * gap
        expected_mask += [1] * len(pairs)
        expected_logprobs += [lp for lp, _ in pairs]
    assert sample.loss_mask == expected_mask
    sampled_positions = [i for i, m in enumerate(expected_mask) if m]
    assert [sample.rollout_log_probs[i] for i in sampled_positions] == pytest.approx(
        [expected_logprobs[i] for i in sampled_positions])
