"""OpenAI Responses API on the session path (``POST /sessions/{id}/v1/responses``).

Codex (0.145 and later) speaks only the Responses API.  The session server
records tokens on its chat-completions path (TITO input ids, sampled token
ids, log-probabilities), so this adapter translates a Responses request into
a chat-completions request, runs it through ``SessionCore.chat_completions``
unchanged, and renders the chat completion back as a Responses object (or a
Responses event stream).  A Responses call is therefore recorded exactly like
the equivalent chat call; the client only sees Responses wire shapes.

Scope (fail closed with 400 on anything else):

* full replay only: ``previous_response_id`` is refused (the session must see
  the whole history to match it against its checkpoints);
* input items: ``message`` (user / system / developer / assistant),
  ``reasoning``, ``function_call``, ``function_call_output``;
* tools: ``function``; ``namespace`` and OpenAI-hosted tool types are left out
  of the prompt (the served model cannot call them); others are refused;
* no token streaming: with ``stream: true`` the whole answer is sent as one
  event sequence (created, per-item added/done, completed) after the
  completion finishes, as the Anthropic route does.

The module has no SGLang or FastAPI dependency so the translation is unit-testable.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any

# Responses tool types the policy is not offered: ``namespace`` (Codex sends its
# sub-agent tools so) and OpenAI hosted tools.
OMITTED_TOOL_TYPES = frozenset(
    {"namespace", "web_search", "web_search_preview", "image_generation", "file_search", "code_interpreter"}
)
_SAMPLING_FIELDS = ("temperature", "top_p")


class ResponsesRequestError(ValueError):
    """The Responses request uses a shape this adapter does not translate."""


def _text(content: Any, kinds: tuple[str, ...]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") not in kinds:
                raise ResponsesRequestError(f"unsupported content block {block!r}")
            parts.append(str(block.get("text", "")))
        return "".join(parts)
    raise ResponsesRequestError(f"unsupported content {type(content).__name__}")


def _tools(tools: Any) -> list[dict[str, Any]]:
    out = []
    for tool in tools or []:
        kind = tool.get("type") if isinstance(tool, dict) else None
        if kind in OMITTED_TOOL_TYPES:
            continue
        if kind != "function":
            raise ResponsesRequestError(f"unsupported tool type {kind!r}")
        function = {k: tool[k] for k in ("name", "description", "parameters", "strict") if k in tool}
        out.append({"type": "function", "function": function})
    return out


def _tool_choice(choice: Any) -> Any:
    if choice is None or isinstance(choice, str):
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function" and "name" in choice:
        return {"type": "function", "function": {"name": choice["name"]}}
    raise ResponsesRequestError(f"unsupported tool_choice {choice!r}")


def _messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    assistant: dict[str, Any] | None = None  # the assistant turn being rebuilt from its items
    reasoning: str | None = None

    def flush() -> None:
        nonlocal assistant, reasoning
        if assistant is not None:
            messages.append(assistant)
        assistant, reasoning = None, None

    def open_assistant() -> dict[str, Any]:
        nonlocal assistant
        if assistant is None:
            assistant = {"role": "assistant", "content": ""}
            if reasoning is not None:
                assistant["reasoning_content"] = reasoning
        return assistant

    items = body.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    if not isinstance(items, list) or not items:
        raise ResponsesRequestError("Responses input must be a string or a non-empty list")
    for item in items:
        kind = item.get("type", "message") if isinstance(item, dict) else None
        if kind == "message":
            role = item.get("role")
            if role in ("user", "system", "developer"):
                flush()
                messages.append({"role": "user" if role == "user" else "system",
                                 "content": _text(item.get("content"), ("input_text", "text"))})
            elif role == "assistant":
                open_assistant()["content"] += _text(item.get("content"), ("output_text", "text"))
            else:
                raise ResponsesRequestError(f"unsupported message role {role!r}")
        elif kind == "reasoning":
            flush()
            reasoning = "".join(str(c.get("text", "")) for c in item.get("content") or [])
        elif kind == "function_call":
            arguments = item.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            open_assistant().setdefault("tool_calls", []).append(
                {"id": str(item["call_id"]), "type": "function",
                 "function": {"name": str(item["name"]), "arguments": arguments}})
        elif kind == "function_call_output":
            flush()
            output = item.get("output")
            messages.append({"role": "tool", "tool_call_id": str(item["call_id"]),
                             "content": output if isinstance(output, str) else json.dumps(output)})
        else:
            raise ResponsesRequestError(f"unsupported Responses input item {kind!r}")
    flush()
    return messages


def to_chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """The chat-completions request (non-streaming) that a Responses request stands for."""
    if not isinstance(body, dict):
        raise ResponsesRequestError("Responses request must be a JSON object")
    if body.get("previous_response_id"):
        raise ResponsesRequestError("previous_response_id is not supported: send the full input (store: false)")
    chat: dict[str, Any] = {"messages": _messages(body), "stream": False}
    if "model" in body:
        chat["model"] = body["model"]
    tools = _tools(body.get("tools"))
    if tools:
        chat["tools"] = tools
        if (choice := _tool_choice(body.get("tool_choice"))) is not None:
            chat["tool_choice"] = choice
    if body.get("parallel_tool_calls") is not None:
        chat["parallel_tool_calls"] = bool(body["parallel_tool_calls"])
    if body.get("max_output_tokens") is not None:
        chat["max_tokens"] = int(body["max_output_tokens"])
    for key in _SAMPLING_FIELDS:
        if body.get(key) is not None:
            chat[key] = body[key]
    return chat


def to_response(completion: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """The Responses object for a chat completion of ``request``."""
    choice = completion["choices"][0]
    message = choice.get("message") or {}
    rid = str(completion.get("id") or f"resp_{int(time.time() * 1000)}")
    output: list[dict[str, Any]] = []
    if message.get("reasoning_content"):
        output.append({"type": "reasoning", "id": f"rs_{rid}", "summary": [],
                       "content": [{"type": "reasoning_text", "text": message["reasoning_content"]}]})
    if message.get("content"):
        output.append({"type": "message", "id": f"msg_{rid}", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": message["content"], "annotations": []}]})
    for call in message.get("tool_calls") or []:
        output.append({"type": "function_call", "id": f"fc_{call['id']}", "call_id": call["id"],
                       "status": "completed", "name": call["function"]["name"],
                       "arguments": call["function"]["arguments"]})
    usage = completion.get("usage") or {}
    incomplete = choice.get("finish_reason") == "length"
    return {
        "id": rid if rid.startswith("resp_") else f"resp_{rid}",
        "object": "response",
        "created_at": int(completion.get("created") or time.time()),
        "status": "incomplete" if incomplete else "completed",
        "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
        "error": None,
        "model": request.get("model") or completion.get("model"),
        "instructions": request.get("instructions"),
        "output": output,
        "parallel_tool_calls": bool(request.get("parallel_tool_calls", False)),
        "tool_choice": request.get("tool_choice", "auto"),
        "tools": request.get("tools") or [],
        "store": False,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                  "output_tokens": usage.get("completion_tokens", 0),
                  "total_tokens": usage.get("total_tokens", 0)},
    }


def to_sse(response: dict[str, Any]) -> bytes:
    """Responses event stream for a finished response: created, per item added/(arguments)/done, completed."""
    final_type = "response.completed" if response["status"] == "completed" else "response.incomplete"
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}}]
    for index, item in enumerate(response["output"]):
        added = copy.deepcopy(item)
        added["status"] = "in_progress"
        if item["type"] == "function_call":
            added["arguments"] = ""
        events.append({"type": "response.output_item.added", "output_index": index, "item": added})
        if item["type"] == "function_call":
            events.append({"type": "response.function_call_arguments.delta", "item_id": item["id"],
                           "output_index": index, "delta": item["arguments"]})
            events.append({"type": "response.function_call_arguments.done", "item_id": item["id"],
                           "output_index": index, "arguments": item["arguments"]})
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
    events.append({"type": final_type, "response": response})
    for number, event in enumerate(events):
        event["sequence_number"] = number
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n".encode() for e in events)


def error_body(message: str, *, kind: str = "invalid_request_error") -> dict[str, Any]:
    """OpenAI-style error envelope."""
    return {"error": {"message": message, "type": kind, "param": None, "code": None}}
