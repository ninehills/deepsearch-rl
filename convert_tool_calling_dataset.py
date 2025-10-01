#!/usr/bin/env python3
"""Convert tool calling dataset to different formats."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def convert_assistant_message_to_swift(message: Dict[str, Any], with_thinking: bool = False) -> List[Dict[str, Any]]:
    """Convert assistant message with tool_calls to swift format."""
    result = []

    # Build content with reasoning_content if present
    content = message.get("content", "")
    reasoning_content = message.get("reasoning_content", "")

    # Prepend reasoning_content wrapped in <think> tags if it exists and with_thinking is enabled
    if with_thinking:
        # Qwen: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507/blob/main/tokenizer_config.json#L229
        # {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
        content = f"<think>\n{reasoning_content.strip('\n')}\n</think>\n\n{content.lstrip('\n')}"

    # If there's content and it's not empty, add it as the first message
    if content and content.strip():
        result.append({
            "role": "assistant",
            "content": content
        })

    # Convert each tool call to a separate message
    tool_calls = message.get("tool_calls", []) or []
    for tool_call in tool_calls:
        function_info = tool_call.get("function", {})
        tool_message = {
            "role": "tool_call",
            "content": json.dumps({
                "name": function_info.get("name", ""),
                "arguments": json.loads(function_info.get("arguments", "{}"))
            }, ensure_ascii=False)
        }
        result.append(tool_message)

    return result


def convert_messages_to_swift(messages: List[Dict[str, Any]], with_thinking: bool = False) -> List[Dict[str, Any]]:
    """Convert messages list to swift format."""
    result = []

    for message in messages:
        message = convert_reasoning_content(message)
        if message.get("role") == "assistant" and "tool_calls" in message:
            # Convert assistant message with tool_calls
            converted = convert_assistant_message_to_swift(message, with_thinking)
            result.extend(converted)
        else:
            # Keep other messages as is
            result.append(message)

    return result


def convert_reasoning_content(message: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single message for TRL format."""
    converted_message = message.copy()

    # If reasoning_content doesn't exist or is empty, look for reasoning field
    if not converted_message.get("reasoning_content"):
        if "reasoning" in message and message["reasoning"]:
            converted_message["reasoning_content"] = message["reasoning"]
            # Remove the original reasoning field
            del converted_message["reasoning"]

    return converted_message


def convert_to_swift(data: List[Dict[str, Any]], with_thinking: bool = False) -> List[Dict[str, Any]]:
    """Convert dataset to swift format."""
    result = []

    for entry in data:
        # Only keep messages and tools fields
        swift_entry = {
            "messages": convert_messages_to_swift(entry.get("messages", []), with_thinking),
            "tools": entry.get("tools", [])
        }
        result.append(swift_entry)

    return result


def convert_to_trl(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert dataset to trl format."""
    result = []

    for entry in data:
        # Convert each message and only keep messages and tools fields
        converted_messages = [convert_reasoning_content(msg) for msg in entry.get("messages", [])]

        trl_entry = {
            "messages": converted_messages,
            "tools": entry.get("tools", [])
        }
        result.append(trl_entry)

    return result


def main():
    parser = argparse.ArgumentParser(description="Convert tool calling dataset to different formats")
    parser.add_argument("--type", required=True, choices=["swift", "trl"], help="Conversion type")
    parser.add_argument("--input_path", required=True, help="Input JSONL file path")
    parser.add_argument("--output_path", required=True, help="Output JSONL file path")
    parser.add_argument("--thinking", action="store_true", help="Wrap reasoning_content with <think> tags for swift format")

    args = parser.parse_args()

    # Load input data
    print(f"Loading data from {args.input_path}")
    data = load_jsonl(args.input_path)
    print(f"Loaded {len(data)} entries")

    # Convert based on type
    if args.type == "swift":
        converted_data = convert_to_swift(data, args.thinking)
    elif args.type == "trl":
        converted_data = convert_to_trl(data)
    else:
        raise ValueError(f"Unsupported conversion type: {args.type}")

    # Save converted data
    save_jsonl(converted_data, args.output_path)
    print(f"Converted data saved to {args.output_path}")
    print(f"Converted {len(converted_data)} entries")


if __name__ == "__main__":
    main()