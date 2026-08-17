import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from src.llm.backend import CompletionResult, StreamChunk, ToolCallResult
from src.llm.providers.nvidia_provider import NVIDIAInferenceProvider, InferenceRequest
from src.llm.structured_output import empty_structured_output

logger = logging.getLogger(__name__)


class NVIDIABackend:
    """Provider backend wrapping the custom NVIDIAInferenceProvider."""

    def __init__(self, provider: NVIDIAInferenceProvider) -> None:
        self._provider = provider

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
        max_output_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> CompletionResult:
        # Construct the InferenceRequest
        request = InferenceRequest(
            messages=messages,
            model_name=model,
            max_tokens=max_output_tokens or max_tokens,
            temperature=temperature if temperature is not None else 0.7,
            extra_body=extra_params.get("extra_body") if extra_params else None,
            tools=self._convert_tools(tools) if tools else None,
            task_type="honcho_deriver_completion",
        )

        response = await self._provider.generate(request)

        # Map to Honcho CompletionResult
        tool_calls: list[ToolCallResult] = []
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_input: dict[str, Any] = {}
                try:
                    if tc["function"]["arguments"]:
                        tool_input = json.loads(tc["function"]["arguments"])
                except Exception as e:
                    logger.warning("Malformed tool arguments for %s: %s", tc["function"]["name"], e)
                
                tool_calls.append(
                    ToolCallResult(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        input=tool_input,
                    )
                )

        # Handle basic JSON response_format since NVIDIA doesn't strictly have Parse object yet in this wrapper
        content = response.content or ""
        
        # If the request requires a structured output, and we have one (or it's supposed to be JSON mode)
        # Note: the full robust structured output parsing is omitted in this simple adaptation, 
        # but the content is passed back directly.
        if isinstance(response_format, type) and not content and not tool_calls:
            # Fallback to empty structured output
            try:
                content = empty_structured_output(response_format)
            except Exception:
                content = ""

        return CompletionResult(
            content=content,
            input_tokens=response.usage.get("prompt_tokens", 0),
            output_tokens=response.usage.get("completion_tokens", 0),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            finish_reason="stop",
            tool_calls=tool_calls,
            thinking_content=None,
            reasoning_details=[],
            raw_response=None,
        )

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
        max_output_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        request = InferenceRequest(
            messages=messages,
            model_name=model,
            max_tokens=max_output_tokens or max_tokens,
            temperature=temperature if temperature is not None else 0.7,
            extra_body=extra_params.get("extra_body") if extra_params else None,
            tools=self._convert_tools(tools) if tools else None,
            task_type="honcho_streaming_completion",
        )

        async for chunk in self._provider.generate_stream(request):
            yield StreamChunk(
                content=chunk.delta_content,
                is_done=chunk.finish_reason is not None,
                finish_reason=chunk.finish_reason,
                output_tokens=0,
            )

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # NVIDIA typically supports standard OpenAI tool shapes
        if not tools or tools[0].get("type") == "function":
            return tools
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]
