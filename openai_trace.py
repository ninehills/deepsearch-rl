import json
import functools
import asyncio
import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Union
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.resources.chat.completions.completions import AsyncCompletions
from openai._types import NOT_GIVEN


def safe_patch(obj: Any, method_name: str, patched_fn):
    """Safely patch a method on an object, storing the original method."""
    original_method = getattr(obj, method_name)
    setattr(obj, method_name, patched_fn(original_method))
    return original_method


def filter_not_given(data: Any) -> Any:
    """Recursively filter out NOT_GIVEN values from data structure."""
    if data is NOT_GIVEN:
        return None
    elif isinstance(data, dict):
        return {k: filter_not_given(v) for k, v in data.items() if v is not NOT_GIVEN}
    elif isinstance(data, (list, tuple)):
        return [filter_not_given(item) for item in data if item is not NOT_GIVEN]
    else:
        return data


class AsyncOpenAITrace(AsyncOpenAI):
    def __init__(self, trace_file: Union[str, Path] = "openai_trace.log", **kwargs):
        super().__init__(**kwargs)
        self.trace_file = str(trace_file)
        self._setup_trace_logger()
        self._patch_create_method()

    def _setup_trace_logger(self):
        """Setup a dedicated logger for tracing."""
        self.trace_logger = logging.getLogger(f"{__name__}.trace")
        self.trace_logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        for handler in self.trace_logger.handlers[:]:
            self.trace_logger.removeHandler(handler)

        # Create file handler
        file_handler = logging.FileHandler(self.trace_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Create formatter for JSON output
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)

        self.trace_logger.addHandler(file_handler)
        self.trace_logger.propagate = False

    def _log_trace(self, endpoint: str, input_data: Dict[str, Any], output_data: Any):
        """Log the input and output data using the trace logger."""
        try:
            # Filter out NOT_GIVEN values from input data
            filtered_input = filter_not_given(input_data)

            # Convert output to serializable format
            if hasattr(output_data, 'model_dump'):
                serializable_output = output_data.model_dump()
            elif hasattr(output_data, '__dict__'):
                serializable_output = str(output_data)
            else:
                serializable_output = output_data

            trace_entry = {
                "time": datetime.now(timezone.utc).isoformat(),
                "endpoint": endpoint,
                "input": filtered_input,
                "output": serializable_output
            }

            # Use logging library for atomic write
            self.trace_logger.info(json.dumps(trace_entry, ensure_ascii=False))

        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to write trace: {e}")

    def _create_async_patched_call(self, original_create):
        """Create a patched version of the async create method."""
        @functools.wraps(original_create)
        async def async_patched_call(self_completions, *args, **kwargs):
            # Execute the original function
            result = await original_create(self_completions, *args, **kwargs)

            # Check if it's a streaming response
            if hasattr(result, '__aiter__'):
                # Handle streaming response
                return self._wrap_stream(result, kwargs)
            else:
                # Log non-streaming trace
                self._log_trace("chat.completions.create", kwargs, result)
                return result

        return async_patched_call

    def _wrap_stream(self, stream, input_kwargs):
        """Wrap a streaming response to collect chunks and log merged result."""

        class StreamWrapper:
            def __init__(self, stream, tracer, input_kwargs):
                self.stream = stream
                self.tracer = tracer
                self.input_kwargs = input_kwargs
                self.chunks = []

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    chunk = await self.stream.__anext__()
                    self.chunks.append(chunk)
                    return chunk
                except StopAsyncIteration:
                    # Stream finished, log merged result
                    if self.chunks:
                        merged_response = self.tracer._merge_chunks(self.chunks)
                        self.tracer._log_trace("chat.completions.create", self.input_kwargs, merged_response)
                    raise

        return StreamWrapper(stream, self, input_kwargs)

    def _merge_chunks(self, chunks):
        """Merge streaming chunks into a final response format."""
        if not chunks:
            return None

        # Use the first chunk as template
        first_chunk = chunks[0]
        merged = {
            "id": first_chunk.id,
            "object": "chat.completion",  # Change from chunk to completion
            "created": first_chunk.created,
            "model": first_chunk.model,
            "system_fingerprint": getattr(first_chunk, 'system_fingerprint', None),
            "choices": [],
            "usage": None
        }

        # Merge content from all chunks
        choices_map = {}
        for chunk in chunks:
            for choice in chunk.choices:
                index = choice.index
                if index not in choices_map:
                    choices_map[index] = {
                        "index": index,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning": "",
                            "function_call": None,
                            "tool_calls": None
                        },
                        "finish_reason": None
                    }

                # Merge delta content
                if choice.delta and choice.delta.content:
                    choices_map[index]["message"]["content"] += choice.delta.content

                # Merge delta reasoning
                if choice.delta and hasattr(choice.delta, 'reasoning') and choice.delta.reasoning:
                    choices_map[index]["message"]["reasoning"] += choice.delta.reasoning

                # Merge tool calls
                if choice.delta and choice.delta.tool_calls:
                    if choices_map[index]["message"]["tool_calls"] is None:
                        choices_map[index]["message"]["tool_calls"] = []

                    for delta_tool_call in choice.delta.tool_calls:
                        call_index = delta_tool_call.index if delta_tool_call.index is not None else 0

                        # Ensure we have enough tool calls in the list
                        while len(choices_map[index]["message"]["tool_calls"]) <= call_index:
                            choices_map[index]["message"]["tool_calls"].append({
                                "id": None,
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })

                        tool_call = choices_map[index]["message"]["tool_calls"][call_index]

                        # Merge tool call data
                        if delta_tool_call.id:
                            tool_call["id"] = delta_tool_call.id
                        if delta_tool_call.function:
                            if delta_tool_call.function.name:
                                tool_call["function"]["name"] += delta_tool_call.function.name
                            if delta_tool_call.function.arguments:
                                tool_call["function"]["arguments"] += delta_tool_call.function.arguments

                # Set finish reason from last chunk
                if choice.finish_reason:
                    choices_map[index]["finish_reason"] = choice.finish_reason

        merged["choices"] = list(choices_map.values())

        # Add usage from last chunk if available
        if chunks and hasattr(chunks[-1], 'usage') and chunks[-1].usage:
            merged["usage"] = chunks[-1].usage.model_dump() if hasattr(chunks[-1].usage, 'model_dump') else chunks[-1].usage

        return merged

    def _patch_create_method(self):
        """Patch the AsyncCompletions.create method to add tracing."""
        safe_patch(
            AsyncCompletions,
            "create",
            self._create_async_patched_call
        )


if __name__ == "__main__":
    async def test_patch():
        # Load environment variables
        load_dotenv()

        # Get configuration from environment
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return

        # Create traced client
        client = AsyncOpenAITrace(
            api_key=api_key,
            base_url=base_url,
            trace_file="test_trace.log"
        )

        # Test streaming + function calling
        try:
            logger.info("Testing streaming + function calling...")

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a given location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA"
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "The temperature unit"
                                }
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]

            stream = await client.chat.completions.create(
                model="openrouter/gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "What's the weather like in Tokyo? Please explain your answer."}
                ],
                tools=tools,
                tool_choice="auto",
                stream=True
            )

            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
                if chunk.choices[0].delta.content:
                    logger.info(f"Content chunk: {chunk.choices[0].delta.content}")
                elif chunk.choices[0].delta.tool_calls:
                    logger.info(f"Tool call chunk: {chunk.choices[0].delta.tool_calls}")

            logger.info(f"Total chunks received: {len(chunks)}")
            logger.info("Streaming + function calling trace logged to test_trace.log")

        except Exception as e:
            logger.error(f"Error: {e}")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    asyncio.run(test_patch())