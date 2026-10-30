"""Shared, bounded execution of validated read-only recommendation tools."""

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, ValidationError

MAX_TOOL_CALLS = 4


def run_agent(
    model: Any,
    messages: list[BaseMessage],
    schemas: Mapping[str, type[BaseModel]],
    execute: Callable[[BaseModel], str],
) -> str | None:
    calls = 0
    while True:
        answer = model.invoke(messages)
        if not answer.tool_calls:
            return StrOutputParser().invoke(answer).strip() or None
        messages.append(answer)
        for call in answer.tool_calls:
            if calls >= MAX_TOOL_CALLS or not call.get("id"):
                return None
            calls += 1
            schema = schemas.get(call["name"])
            try:
                action = schema.model_validate(call["args"]) if schema else None
            except ValidationError:
                action = None
            result = (
                execute(action)
                if action is not None
                else "Invalid action or arguments. Use an advertised tool with its declared schema."
            )
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
