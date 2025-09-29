#!/usr/bin/env python3
"""Analyze trajectory from OpenAI trace logs and evaluation results."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON or JSONL file based on extension."""
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            return json.load(f)
        else:  # JSONL
            return [json.loads(line.strip()) for line in f if line.strip()]


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_user_question(messages: List[Dict[str, Any]]) -> str:
    """Extract the first user message from messages."""
    for message in messages:
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def find_matching_trace_entries(question: str, trace_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find trace entries that match the given question."""
    matching_entries = []

    for entry in trace_data:
        if entry.get("endpoint") != "chat.completions.create":
            continue

        input_data = entry.get("input", {})
        messages = input_data.get("messages", [])

        # Extract first user message
        user_message = extract_user_question(messages)

        # Simple matching: check if question is contained in user message
        if question.strip() in user_message.strip():
            matching_entries.append(entry)

    return matching_entries


def extract_final_answer_from_content(content: str) -> str:
    """Extract answer from content, looking for <answer> tags."""
    if "<answer>" in content and "</answer>" in content:
        start = content.find("<answer>") + len("<answer>")
        end = content.find("</answer>")
        return content[start:end].strip()
    return content.strip()


def count_conversation_rounds(messages: List[Dict[str, Any]]) -> int:
    """Count the number of conversation rounds (user-assistant pairs)."""
    user_count = sum(1 for msg in messages if msg.get("role") == "user")
    assistant_count = sum(1 for msg in messages if msg.get("role") == "assistant")
    return min(user_count, assistant_count)


def find_final_trace_entry(matching_entries: List[Dict[str, Any]], expected_answer: str) -> Optional[Dict[str, Any]]:
    """Find the trace entry with the most rounds and final result matching expected answer."""
    if not matching_entries:
        return None

    # Group entries by conversation rounds and filter by matching final answer
    candidates = []

    for entry in matching_entries:
        input_data = entry.get("input", {})
        messages = input_data.get("messages", [])
        rounds = count_conversation_rounds(messages)

        output_data = entry.get("output", {})
        choices = output_data.get("choices", [])

        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")

            if content:
                extracted_answer = extract_final_answer_from_content(content)
                candidates.append((entry, rounds, extracted_answer))

    if not candidates:
        return matching_entries[-1] if matching_entries else None

    # First priority: entries with matching answer, sorted by rounds (descending)
    matching_answer_candidates = [
        (entry, rounds) for entry, rounds, answer in candidates
        if answer == expected_answer
    ]

    if matching_answer_candidates:
        # Return the entry with the most rounds among those with matching answers
        return max(matching_answer_candidates, key=lambda x: x[1])[0]

    # Second priority: entries with most rounds, regardless of answer match
    return max(candidates, key=lambda x: x[1])[0]


def extract_request_meta(trace_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract request metadata from trace entry."""
    input_data = trace_entry.get("input", {})

    meta = {}
    # Extract relevant fields
    for key in ["model", "temperature", "max_tokens", "stream", "top_p", "frequency_penalty", "presence_penalty"]:
        if key in input_data:
            meta[key] = input_data[key]

    return meta


def extract_response_meta(trace_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract response metadata from trace entry (excluding choices)."""
    output_data = trace_entry.get("output", {})

    # Extract all fields except 'choices'
    response_meta = {}
    for key, value in output_data.items():
        if key != "choices":
            response_meta[key] = value

    return response_meta


def analyze_trajectory(output_dir: str, with_eval: bool = False) -> None:
    """Analyze trajectory from trace logs and evaluation results."""
    output_path = Path(output_dir)

    # Load evaluation data
    eval_file = output_path / ("intermediate_data.json" if with_eval else "results.jsonl")
    if not eval_file.exists():
        raise FileNotFoundError(f"{eval_file.name} not found in {output_dir}")
    eval_data = load_data(str(eval_file))

    # Load trace data
    trace_file = output_path / "openai_trace.log"
    if not trace_file.exists():
        raise FileNotFoundError(f"openai_trace.log not found in {output_dir}")
    trace_data = load_data(str(trace_file))

    print(f"Loaded {len(eval_data)} evaluation entries and {len(trace_data)} trace entries")

    # Process each evaluation entry
    trajectory_data = []

    for eval_entry in eval_data:
        question = eval_entry.get("question", "")
        if not question:
            continue

        # Find matching trace entries
        matching_entries = find_matching_trace_entries(question, trace_data)

        if not matching_entries:
            print(f"Warning: No matching trace entries found for question: {question[:50]}...")
            continue

        # Get expected answer/pred
        expected_answer = eval_entry.get("output", {}).get("pred", "")

        # Find the final trace entry
        final_trace = find_final_trace_entry(matching_entries, expected_answer)

        if not final_trace:
            print(f"Warning: No final trace entry found for question: {question[:50]}...")
            continue

        # Extract trajectory information
        input_data = final_trace.get("input", {})
        messages = input_data.get("messages", []).copy()  # Make a copy to avoid modifying original
        tools = input_data.get("tools", [])
        request_meta = extract_request_meta(final_trace)
        response_meta = extract_response_meta(final_trace)

        # Append the output message to complete the conversation
        output_data = final_trace.get("output", {})
        choices = output_data.get("choices", [])
        if choices:
            assistant_message = choices[0].get("message", {})
            if assistant_message:
                messages.append(assistant_message)

        # Build trajectory entry
        output = eval_entry.get("output", {})
        trajectory_entry = {
            "messages": messages,
            "tools": tools,
            "request_meta": request_meta,
            "response_meta": response_meta,
            "id": eval_entry.get("id", ""),
            "question": question,
            "golden_answers": eval_entry.get("golden_answers", []),
            "answer": output.get("answer", ""),
            "pred": output.get("pred", ""),
            "error": output.get("error", ""),
            "metric_score": output.get("metric_score", {}),
        }

        trajectory_data.append(trajectory_entry)

    # Save trajectory data
    trajectory_file = output_path / "trajectory.jsonl"
    save_jsonl(trajectory_data, str(trajectory_file))

    print(f"Trajectory analysis complete. Saved {len(trajectory_data)} entries to {trajectory_file}")

    # If with_eval is enabled, generate trajectory_success.jsonl for high F1 scores
    if with_eval:
        success_data = [
            entry for entry in trajectory_data
            if entry.get("metric_score", {}).get("f1", 0) >= 0.8
        ]

        success_file = output_path / "trajectory_success.jsonl"
        save_jsonl(success_data, str(success_file))

        print(f"Success trajectory analysis complete. Saved {len(success_data)} entries with F1 >= 0.8 to {success_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory from OpenAI trace logs")
    parser.add_argument("--output_dir", required=True, help="Output directory containing trace and evaluation files")
    parser.add_argument("--with_eval", action="store_true", help="Use intermediate_data.json instead of results.jsonl")

    args = parser.parse_args()

    try:
        analyze_trajectory(args.output_dir, args.with_eval)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())