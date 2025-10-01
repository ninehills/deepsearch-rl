#!/usr/bin/env python3
"""Analyze trajectory from OpenAI trace logs and evaluation results."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Fields that contain reasoning/thinking process in assistant messages
REASONING_FIELDS = ["reasoning", "reasoning_content"]


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


def extract_reasoning_by_round(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract reasoning fields from each conversation round.

    Thinking models don't include previous reasoning in the last round,
    so we need to extract reasoning from each round separately.

    Returns:
        List of dictionaries containing reasoning fields for each round
    """
    reasoning_by_round = []

    for entry in entries:
        output_data = entry.get("output", {})
        choices = output_data.get("choices", [])

        if choices:
            message = choices[0].get("message", {})

            # Extract reasoning fields for this round
            round_reasoning = {}
            for field in REASONING_FIELDS:
                if field in message:
                    round_reasoning[field] = message[field]

            if round_reasoning:  # Only add if there are reasoning fields
                reasoning_by_round.append(round_reasoning)

    return reasoning_by_round


def extract_final_answer_from_content(content: str) -> str:
    """Extract answer from content, looking for <answer> tags."""
    if "<answer>" in content and "</answer>" in content:
        start = content.find("<answer>") + len("<answer>")
        end = content.find("</answer>")
        return content[start:end].strip()
    return content.strip()


def count_conversation_rounds(messages: List[Dict[str, Any]]) -> int:
    """Count the number of conversation rounds (user messages, excluding system)."""
    return sum(1 for msg in messages if msg.get("role") == "user")


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


def sum_tokens_across_rounds(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    """Sum token usage across all conversation rounds."""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for entry in entries:
        output_data = entry.get("output", {})
        usage = output_data.get("usage", {})
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)
        total_tokens += usage.get("total_tokens", 0)

    return {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }


def compute_statistics(trajectory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics from trajectory data."""
    import statistics

    if not trajectory_data:
        return {}

    # Basic counts
    total = len(trajectory_data)

    # Separate data by success (F1 >= 0.8)
    success_data = [e for e in trajectory_data if e.get("metric_score", {}).get("f1", 0) >= 0.8]
    failure_data = [e for e in trajectory_data if e.get("metric_score", {}).get("f1", 0) < 0.8]

    # Collect metrics for all/success/failure
    def init_metric_lists():
        return {
            "all": [],
            "success": [],
            "failure": []
        }

    rounds_list = init_metric_lists()
    tool_calls_list = init_metric_lists()
    em_scores = []
    f1_scores = []
    acc_scores = []
    precision_scores = []
    recall_scores = []

    # Token usage - separated for last turn and all turns
    last_turn_prompt_tokens = init_metric_lists()
    last_turn_completion_tokens = init_metric_lists()
    last_turn_total_tokens = init_metric_lists()

    all_turns_prompt_tokens = init_metric_lists()
    all_turns_completion_tokens = init_metric_lists()
    all_turns_total_tokens = init_metric_lists()

    reasoning_lengths = []

    # Distribution by rounds and tool calls - separated by success/failure
    rounds_dist = {"all": {}, "success": {}, "failure": {}}
    tool_calls_dist = {"all": {}, "success": {}, "failure": {}}

    def process_entry(entry, category):
        """Process a single entry and categorize metrics."""
        messages = entry.get("messages", [])

        # Count conversation rounds
        rounds = entry.get("rounds", 1)
        rounds_list["all"].append(rounds)
        rounds_list[category].append(rounds)

        # Count tool calls
        tool_calls = 0
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls += len(msg["tool_calls"])
        tool_calls_list["all"].append(tool_calls)
        tool_calls_list[category].append(tool_calls)

        # Collect token usage - last turn
        response_meta = entry.get("response_meta", {})
        usage = response_meta.get("usage", {})
        last_turn_prompt_tokens["all"].append(usage.get("prompt_tokens", 0))
        last_turn_prompt_tokens[category].append(usage.get("prompt_tokens", 0))
        last_turn_completion_tokens["all"].append(usage.get("completion_tokens", 0))
        last_turn_completion_tokens[category].append(usage.get("completion_tokens", 0))
        last_turn_total_tokens["all"].append(usage.get("total_tokens", 0))
        last_turn_total_tokens[category].append(usage.get("total_tokens", 0))

        # Collect token usage - all turns
        all_turns_usage = entry.get("all_turns_tokens", {})
        all_turns_prompt_tokens["all"].append(all_turns_usage.get("prompt_tokens", 0))
        all_turns_prompt_tokens[category].append(all_turns_usage.get("prompt_tokens", 0))
        all_turns_completion_tokens["all"].append(all_turns_usage.get("completion_tokens", 0))
        all_turns_completion_tokens[category].append(all_turns_usage.get("completion_tokens", 0))
        all_turns_total_tokens["all"].append(all_turns_usage.get("total_tokens", 0))
        all_turns_total_tokens[category].append(all_turns_usage.get("total_tokens", 0))

        return rounds, tool_calls

    # Process all entries
    for entry in trajectory_data:
        messages = entry.get("messages", [])
        metric_score = entry.get("metric_score", {})
        f1 = metric_score.get("f1", 0)
        category = "success" if f1 >= 0.8 else "failure"

        rounds, tool_calls = process_entry(entry, category)

        # Distribution for all/success/failure
        rounds_dist["all"][rounds] = rounds_dist["all"].get(rounds, 0) + 1
        rounds_dist[category][rounds] = rounds_dist[category].get(rounds, 0) + 1

        tool_calls_dist["all"][tool_calls] = tool_calls_dist["all"].get(tool_calls, 0) + 1
        tool_calls_dist[category][tool_calls] = tool_calls_dist[category].get(tool_calls, 0) + 1

        # Collect metric scores
        em_scores.append(metric_score.get("em", 0))
        f1_scores.append(f1)
        acc_scores.append(metric_score.get("acc", 0))
        precision_scores.append(metric_score.get("precision", 0))
        recall_scores.append(metric_score.get("recall", 0))

        # Collect reasoning lengths
        for msg in messages:
            if msg.get("role") == "assistant":
                for field in REASONING_FIELDS:
                    if field in msg:
                        reasoning_lengths.append(len(str(msg[field])))

    def safe_mean(lst):
        return statistics.mean(lst) if lst else 0

    def safe_stdev(lst):
        return statistics.stdev(lst) if len(lst) > 1 else 0

    def compute_stats(data_dict):
        """Compute mean/stdev for all/success/failure."""
        return {
            "all": {"mean": safe_mean(data_dict["all"]), "stdev": safe_stdev(data_dict["all"])},
            "success": {"mean": safe_mean(data_dict["success"]), "stdev": safe_stdev(data_dict["success"])},
            "failure": {"mean": safe_mean(data_dict["failure"]), "stdev": safe_stdev(data_dict["failure"])},
        }

    # Success counts
    success_em = sum(1 for s in em_scores if s >= 1.0)
    success_f1_08 = len(success_data)
    failure_count = len(failure_data)

    return {
        "total": total,
        "success_count": success_f1_08,
        "failure_count": failure_count,
        "success_em": success_em,
        "rounds": {
            **compute_stats(rounds_list),
            "distribution": rounds_dist,
        },
        "tool_calls": {
            **compute_stats(tool_calls_list),
            "total": sum(tool_calls_list["all"]),
            "distribution": tool_calls_dist,
        },
        "metrics": {
            "em": {"mean": safe_mean(em_scores), "stdev": safe_stdev(em_scores)},
            "f1": {"mean": safe_mean(f1_scores), "stdev": safe_stdev(f1_scores)},
            "acc": {"mean": safe_mean(acc_scores), "stdev": safe_stdev(acc_scores)},
            "precision": {"mean": safe_mean(precision_scores), "stdev": safe_stdev(precision_scores)},
            "recall": {"mean": safe_mean(recall_scores), "stdev": safe_stdev(recall_scores)},
        },
        "tokens_last_turn": {
            "prompt": compute_stats(last_turn_prompt_tokens),
            "completion": compute_stats(last_turn_completion_tokens),
            "total": compute_stats(last_turn_total_tokens),
        },
        "tokens_all_turns": {
            "prompt": compute_stats(all_turns_prompt_tokens),
            "completion": compute_stats(all_turns_completion_tokens),
            "total": compute_stats(all_turns_total_tokens),
        },
        "reasoning": {
            "mean": safe_mean(reasoning_lengths),
            "stdev": safe_stdev(reasoning_lengths),
            "count": len(reasoning_lengths),
        },
    }


def print_statistics_console(stats: Dict[str, Any], output_dir: str) -> None:
    """Print statistics to console using rich."""
    if not HAS_RICH:
        print("\n" + "="*80)
        print("TRAJECTORY STATISTICS")
        print("="*80)
        print(f"Dataset: {output_dir}")
        print(f"Total Questions: {stats['total']}")
        print(f"Success (EM=1.0): {stats['success_em']} ({stats['success_em']/stats['total']*100:.1f}%)")
        print(f"Success (F1≥0.8): {stats['success_f1_08']} ({stats['success_f1_08']/stats['total']*100:.1f}%)")
        print(f"\nAvg Rounds: {stats['rounds']['mean']:.2f} ± {stats['rounds']['stdev']:.2f}")
        print(f"Avg Tool Calls: {stats['tool_calls']['mean']:.2f} ± {stats['tool_calls']['stdev']:.2f}")
        print(f"\nEM: {stats['metrics']['em']['mean']:.3f} ± {stats['metrics']['em']['stdev']:.3f}")
        print(f"F1: {stats['metrics']['f1']['mean']:.3f} ± {stats['metrics']['f1']['stdev']:.3f}")
        print("="*80 + "\n")
        return

    console = Console()

    # Header
    console.print(Panel.fit(
        f"[bold cyan]TRAJECTORY ANALYSIS STATISTICS[/bold cyan]\n"
        f"Dataset: {output_dir}\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="cyan"
    ))

    # Summary table
    summary = Table(title="Dataset Summary", show_header=True, header_style="bold magenta")
    summary.add_column("Metric", style="cyan", width=30)
    summary.add_column("Value", justify="right", style="green")

    summary.add_row("Total Questions", str(stats['total']))
    summary.add_row("Success (F1 ≥ 0.8)", f"{stats['success_count']} ({stats['success_count']/stats['total']*100:.1f}%)")
    summary.add_row("Failure (F1 < 0.8)", f"{stats['failure_count']} ({stats['failure_count']/stats['total']*100:.1f}%)")
    summary.add_row("Success (EM = 1.0)", f"{stats['success_em']} ({stats['success_em']/stats['total']*100:.1f}%)")
    console.print(summary)

    # Performance metrics
    perf = Table(title="Performance Metrics", show_header=True, header_style="bold magenta")
    perf.add_column("Metric", style="cyan", width=20)
    perf.add_column("Mean ± Std", justify="right", style="green")

    perf.add_row("Exact Match (EM)", f"{stats['metrics']['em']['mean']:.3f} ± {stats['metrics']['em']['stdev']:.3f}")
    perf.add_row("F1 Score", f"{stats['metrics']['f1']['mean']:.3f} ± {stats['metrics']['f1']['stdev']:.3f}")
    perf.add_row("Accuracy", f"{stats['metrics']['acc']['mean']:.3f} ± {stats['metrics']['acc']['stdev']:.3f}")
    perf.add_row("Precision", f"{stats['metrics']['precision']['mean']:.3f} ± {stats['metrics']['precision']['stdev']:.3f}")
    perf.add_row("Recall", f"{stats['metrics']['recall']['mean']:.3f} ± {stats['metrics']['recall']['stdev']:.3f}")
    console.print(perf)

    # Conversation dynamics
    conv = Table(title="Conversation Dynamics", show_header=True, header_style="bold magenta")
    conv.add_column("Metric", style="cyan", width=25)
    conv.add_column("All", justify="right", style="green")
    conv.add_column("Success", justify="right", style="blue")
    conv.add_column("Failure", justify="right", style="yellow")

    conv.add_row(
        "Avg Rounds",
        f"{stats['rounds']['all']['mean']:.2f} ± {stats['rounds']['all']['stdev']:.2f}",
        f"{stats['rounds']['success']['mean']:.2f} ± {stats['rounds']['success']['stdev']:.2f}",
        f"{stats['rounds']['failure']['mean']:.2f} ± {stats['rounds']['failure']['stdev']:.2f}"
    )
    conv.add_row(
        "Avg Tool Calls",
        f"{stats['tool_calls']['all']['mean']:.2f} ± {stats['tool_calls']['all']['stdev']:.2f}",
        f"{stats['tool_calls']['success']['mean']:.2f} ± {stats['tool_calls']['success']['stdev']:.2f}",
        f"{stats['tool_calls']['failure']['mean']:.2f} ± {stats['tool_calls']['failure']['stdev']:.2f}"
    )
    console.print(conv)

    # Token usage - Last Turn
    tokens_last = Table(title="Token Usage (Last Turn)", show_header=True, header_style="bold magenta")
    tokens_last.add_column("Metric", style="cyan", width=20)
    tokens_last.add_column("All", justify="right", style="green")
    tokens_last.add_column("Success", justify="right", style="blue")
    tokens_last.add_column("Failure", justify="right", style="yellow")

    tokens_last.add_row(
        "Total",
        f"{stats['tokens_last_turn']['total']['all']['mean']:.0f} ± {stats['tokens_last_turn']['total']['all']['stdev']:.0f}",
        f"{stats['tokens_last_turn']['total']['success']['mean']:.0f} ± {stats['tokens_last_turn']['total']['success']['stdev']:.0f}",
        f"{stats['tokens_last_turn']['total']['failure']['mean']:.0f} ± {stats['tokens_last_turn']['total']['failure']['stdev']:.0f}"
    )
    tokens_last.add_row(
        "Prompt",
        f"{stats['tokens_last_turn']['prompt']['all']['mean']:.0f} ± {stats['tokens_last_turn']['prompt']['all']['stdev']:.0f}",
        f"{stats['tokens_last_turn']['prompt']['success']['mean']:.0f} ± {stats['tokens_last_turn']['prompt']['success']['stdev']:.0f}",
        f"{stats['tokens_last_turn']['prompt']['failure']['mean']:.0f} ± {stats['tokens_last_turn']['prompt']['failure']['stdev']:.0f}"
    )
    tokens_last.add_row(
        "Completion",
        f"{stats['tokens_last_turn']['completion']['all']['mean']:.0f} ± {stats['tokens_last_turn']['completion']['all']['stdev']:.0f}",
        f"{stats['tokens_last_turn']['completion']['success']['mean']:.0f} ± {stats['tokens_last_turn']['completion']['success']['stdev']:.0f}",
        f"{stats['tokens_last_turn']['completion']['failure']['mean']:.0f} ± {stats['tokens_last_turn']['completion']['failure']['stdev']:.0f}"
    )
    console.print(tokens_last)

    # Token usage - All Turns
    tokens_all = Table(title="Token Usage (All Turns - Cumulative)", show_header=True, header_style="bold magenta")
    tokens_all.add_column("Metric", style="cyan", width=20)
    tokens_all.add_column("All", justify="right", style="green")
    tokens_all.add_column("Success", justify="right", style="blue")
    tokens_all.add_column("Failure", justify="right", style="yellow")

    tokens_all.add_row(
        "Total",
        f"{stats['tokens_all_turns']['total']['all']['mean']:.0f} ± {stats['tokens_all_turns']['total']['all']['stdev']:.0f}",
        f"{stats['tokens_all_turns']['total']['success']['mean']:.0f} ± {stats['tokens_all_turns']['total']['success']['stdev']:.0f}",
        f"{stats['tokens_all_turns']['total']['failure']['mean']:.0f} ± {stats['tokens_all_turns']['total']['failure']['stdev']:.0f}"
    )
    tokens_all.add_row(
        "Prompt",
        f"{stats['tokens_all_turns']['prompt']['all']['mean']:.0f} ± {stats['tokens_all_turns']['prompt']['all']['stdev']:.0f}",
        f"{stats['tokens_all_turns']['prompt']['success']['mean']:.0f} ± {stats['tokens_all_turns']['prompt']['success']['stdev']:.0f}",
        f"{stats['tokens_all_turns']['prompt']['failure']['mean']:.0f} ± {stats['tokens_all_turns']['prompt']['failure']['stdev']:.0f}"
    )
    tokens_all.add_row(
        "Completion",
        f"{stats['tokens_all_turns']['completion']['all']['mean']:.0f} ± {stats['tokens_all_turns']['completion']['all']['stdev']:.0f}",
        f"{stats['tokens_all_turns']['completion']['success']['mean']:.0f} ± {stats['tokens_all_turns']['completion']['success']['stdev']:.0f}",
        f"{stats['tokens_all_turns']['completion']['failure']['mean']:.0f} ± {stats['tokens_all_turns']['completion']['failure']['stdev']:.0f}"
    )
    console.print(tokens_all)

    # Round distribution
    if stats['rounds']['distribution']['all']:
        dist = Table(title="Round Distribution", show_header=True, header_style="bold magenta")
        dist.add_column("Rounds", style="cyan", justify="center", width=8)
        dist.add_column("All", justify="right", style="green")
        dist.add_column("Success", justify="right", style="blue")
        dist.add_column("Failure", justify="right", style="yellow")

        # Get all unique round values
        all_rounds = sorted(set(
            list(stats['rounds']['distribution']['all'].keys()) +
            list(stats['rounds']['distribution']['success'].keys()) +
            list(stats['rounds']['distribution']['failure'].keys())
        ))

        for rounds in all_rounds:
            all_count = stats['rounds']['distribution']['all'].get(rounds, 0)
            success_count = stats['rounds']['distribution']['success'].get(rounds, 0)
            failure_count = stats['rounds']['distribution']['failure'].get(rounds, 0)

            all_pct = f"{all_count} ({all_count/stats['total']*100:.1f}%)"
            success_pct = f"{success_count} ({success_count/stats['success_count']*100:.1f}%)" if stats['success_count'] > 0 else "0 (0.0%)"
            failure_pct = f"{failure_count} ({failure_count/stats['failure_count']*100:.1f}%)" if stats['failure_count'] > 0 else "0 (0.0%)"

            dist.add_row(str(rounds), all_pct, success_pct, failure_pct)
        console.print(dist)

    # Tool calls distribution
    if stats['tool_calls']['distribution']['all']:
        tc_dist = Table(title="Tool Calls Distribution", show_header=True, header_style="bold magenta")
        tc_dist.add_column("Tool Calls", style="cyan", justify="center", width=12)
        tc_dist.add_column("All", justify="right", style="green")
        tc_dist.add_column("Success", justify="right", style="blue")
        tc_dist.add_column("Failure", justify="right", style="yellow")

        # Get all unique tool call values
        all_tc = sorted(set(
            list(stats['tool_calls']['distribution']['all'].keys()) +
            list(stats['tool_calls']['distribution']['success'].keys()) +
            list(stats['tool_calls']['distribution']['failure'].keys())
        ))

        for tc_count in all_tc:
            all_count = stats['tool_calls']['distribution']['all'].get(tc_count, 0)
            success_count = stats['tool_calls']['distribution']['success'].get(tc_count, 0)
            failure_count = stats['tool_calls']['distribution']['failure'].get(tc_count, 0)

            all_pct = f"{all_count} ({all_count/stats['total']*100:.1f}%)"
            success_pct = f"{success_count} ({success_count/stats['success_count']*100:.1f}%)" if stats['success_count'] > 0 else "0 (0.0%)"
            failure_pct = f"{failure_count} ({failure_count/stats['failure_count']*100:.1f}%)" if stats['failure_count'] > 0 else "0 (0.0%)"

            tc_dist.add_row(str(tc_count), all_pct, success_pct, failure_pct)
        console.print(tc_dist)

    console.print()


def save_statistics_markdown(stats: Dict[str, Any], output_dir: str, file_path: str) -> None:
    """Save statistics to markdown file."""
    lines = [
        "# Trajectory Analysis Statistics\n",
        f"**Dataset:** {output_dir}  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n",
        "## Dataset Summary\n",
        f"- **Total Questions:** {stats['total']}",
        f"- **Success (F1 ≥ 0.8):** {stats['success_count']} ({stats['success_count']/stats['total']*100:.1f}%)",
        f"- **Failure (F1 < 0.8):** {stats['failure_count']} ({stats['failure_count']/stats['total']*100:.1f}%)",
        f"- **Success (EM = 1.0):** {stats['success_em']} ({stats['success_em']/stats['total']*100:.1f}%)\n",
        "## Performance Metrics\n",
        "| Metric | Mean ± Std |",
        "|--------|------------|",
        f"| Exact Match (EM) | {stats['metrics']['em']['mean']:.3f} ± {stats['metrics']['em']['stdev']:.3f} |",
        f"| F1 Score | {stats['metrics']['f1']['mean']:.3f} ± {stats['metrics']['f1']['stdev']:.3f} |",
        f"| Accuracy | {stats['metrics']['acc']['mean']:.3f} ± {stats['metrics']['acc']['stdev']:.3f} |",
        f"| Precision | {stats['metrics']['precision']['mean']:.3f} ± {stats['metrics']['precision']['stdev']:.3f} |",
        f"| Recall | {stats['metrics']['recall']['mean']:.3f} ± {stats['metrics']['recall']['stdev']:.3f} |\n",
        "## Conversation Dynamics\n",
        "| Metric | All | Success | Failure |",
        "|--------|-----|---------|---------|",
        f"| Avg Rounds | {stats['rounds']['all']['mean']:.2f} ± {stats['rounds']['all']['stdev']:.2f} | {stats['rounds']['success']['mean']:.2f} ± {stats['rounds']['success']['stdev']:.2f} | {stats['rounds']['failure']['mean']:.2f} ± {stats['rounds']['failure']['stdev']:.2f} |",
        f"| Avg Tool Calls | {stats['tool_calls']['all']['mean']:.2f} ± {stats['tool_calls']['all']['stdev']:.2f} | {stats['tool_calls']['success']['mean']:.2f} ± {stats['tool_calls']['success']['stdev']:.2f} | {stats['tool_calls']['failure']['mean']:.2f} ± {stats['tool_calls']['failure']['stdev']:.2f} |\n",
        "## Token Usage (Last Turn)\n",
        "| Metric | All | Success | Failure |",
        "|--------|-----|---------|---------|",
        f"| Total | {stats['tokens_last_turn']['total']['all']['mean']:.0f} ± {stats['tokens_last_turn']['total']['all']['stdev']:.0f} | {stats['tokens_last_turn']['total']['success']['mean']:.0f} ± {stats['tokens_last_turn']['total']['success']['stdev']:.0f} | {stats['tokens_last_turn']['total']['failure']['mean']:.0f} ± {stats['tokens_last_turn']['total']['failure']['stdev']:.0f} |",
        f"| Prompt | {stats['tokens_last_turn']['prompt']['all']['mean']:.0f} ± {stats['tokens_last_turn']['prompt']['all']['stdev']:.0f} | {stats['tokens_last_turn']['prompt']['success']['mean']:.0f} ± {stats['tokens_last_turn']['prompt']['success']['stdev']:.0f} | {stats['tokens_last_turn']['prompt']['failure']['mean']:.0f} ± {stats['tokens_last_turn']['prompt']['failure']['stdev']:.0f} |",
        f"| Completion | {stats['tokens_last_turn']['completion']['all']['mean']:.0f} ± {stats['tokens_last_turn']['completion']['all']['stdev']:.0f} | {stats['tokens_last_turn']['completion']['success']['mean']:.0f} ± {stats['tokens_last_turn']['completion']['success']['stdev']:.0f} | {stats['tokens_last_turn']['completion']['failure']['mean']:.0f} ± {stats['tokens_last_turn']['completion']['failure']['stdev']:.0f} |",
    ]

    if stats['reasoning']['count'] > 0:
        lines.append(f"- **Avg Reasoning Length:** {stats['reasoning']['mean']:.0f} ± {stats['reasoning']['stdev']:.0f} chars")

    lines.extend([
        "\n## Token Usage (All Turns - Cumulative)\n",
        "| Metric | All | Success | Failure |",
        "|--------|-----|---------|---------|",
        f"| Total | {stats['tokens_all_turns']['total']['all']['mean']:.0f} ± {stats['tokens_all_turns']['total']['all']['stdev']:.0f} | {stats['tokens_all_turns']['total']['success']['mean']:.0f} ± {stats['tokens_all_turns']['total']['success']['stdev']:.0f} | {stats['tokens_all_turns']['total']['failure']['mean']:.0f} ± {stats['tokens_all_turns']['total']['failure']['stdev']:.0f} |",
        f"| Prompt | {stats['tokens_all_turns']['prompt']['all']['mean']:.0f} ± {stats['tokens_all_turns']['prompt']['all']['stdev']:.0f} | {stats['tokens_all_turns']['prompt']['success']['mean']:.0f} ± {stats['tokens_all_turns']['prompt']['success']['stdev']:.0f} | {stats['tokens_all_turns']['prompt']['failure']['mean']:.0f} ± {stats['tokens_all_turns']['prompt']['failure']['stdev']:.0f} |",
        f"| Completion | {stats['tokens_all_turns']['completion']['all']['mean']:.0f} ± {stats['tokens_all_turns']['completion']['all']['stdev']:.0f} | {stats['tokens_all_turns']['completion']['success']['mean']:.0f} ± {stats['tokens_all_turns']['completion']['success']['stdev']:.0f} | {stats['tokens_all_turns']['completion']['failure']['mean']:.0f} ± {stats['tokens_all_turns']['completion']['failure']['stdev']:.0f} |",
    ])

    lines.append("\n## Round Distribution\n")
    lines.append("| Rounds | All | Success | Failure |")
    lines.append("|--------|-----|---------|---------|")

    # Get all unique round values
    all_rounds = sorted(set(
        list(stats['rounds']['distribution']['all'].keys()) +
        list(stats['rounds']['distribution']['success'].keys()) +
        list(stats['rounds']['distribution']['failure'].keys())
    ))

    for rounds in all_rounds:
        all_count = stats['rounds']['distribution']['all'].get(rounds, 0)
        success_count = stats['rounds']['distribution']['success'].get(rounds, 0)
        failure_count = stats['rounds']['distribution']['failure'].get(rounds, 0)

        all_pct = f"{all_count} ({all_count/stats['total']*100:.1f}%)"
        success_pct = f"{success_count} ({success_count/stats['success_count']*100:.1f}%)" if stats['success_count'] > 0 else "0 (0.0%)"
        failure_pct = f"{failure_count} ({failure_count/stats['failure_count']*100:.1f}%)" if stats['failure_count'] > 0 else "0 (0.0%)"

        lines.append(f"| {rounds} | {all_pct} | {success_pct} | {failure_pct} |")

    lines.append("\n## Tool Calls Distribution\n")
    lines.append("| Tool Calls | All | Success | Failure |")
    lines.append("|------------|-----|---------|---------|")

    # Get all unique tool call values
    all_tc = sorted(set(
        list(stats['tool_calls']['distribution']['all'].keys()) +
        list(stats['tool_calls']['distribution']['success'].keys()) +
        list(stats['tool_calls']['distribution']['failure'].keys())
    ))

    for tc_count in all_tc:
        all_count = stats['tool_calls']['distribution']['all'].get(tc_count, 0)
        success_count = stats['tool_calls']['distribution']['success'].get(tc_count, 0)
        failure_count = stats['tool_calls']['distribution']['failure'].get(tc_count, 0)

        all_pct = f"{all_count} ({all_count/stats['total']*100:.1f}%)"
        success_pct = f"{success_count} ({success_count/stats['success_count']*100:.1f}%)" if stats['success_count'] > 0 else "0 (0.0%)"
        failure_pct = f"{failure_count} ({failure_count/stats['failure_count']*100:.1f}%)" if stats['failure_count'] > 0 else "0 (0.0%)"

        lines.append(f"| {tc_count} | {all_pct} | {success_pct} | {failure_pct} |")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


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

        # Count rounds (number of API calls for this question)
        num_rounds = len(matching_entries)

        # Get expected answer/pred
        expected_answer = eval_entry.get("output", {}).get("pred", "")

        # Find the final trace entry
        final_trace = find_final_trace_entry(matching_entries, expected_answer)

        if not final_trace:
            print(f"Warning: No final trace entry found for question: {question[:50]}...")
            continue

        # Extract reasoning fields from each round
        reasoning_by_round = extract_reasoning_by_round(matching_entries)

        # Sum tokens across all rounds
        all_turns_tokens = sum_tokens_across_rounds(matching_entries)

        # Extract trajectory information
        input_data = final_trace.get("input", {})
        messages = input_data.get("messages", []).copy()  # Make a copy to avoid modifying original
        tools = input_data.get("tools", [])
        request_meta = extract_request_meta(final_trace)
        response_meta = extract_response_meta(final_trace)

        # Append reasoning fields to corresponding assistant messages
        reasoning_idx = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and reasoning_idx < len(reasoning_by_round):
                # Add reasoning fields to this assistant message
                for field, value in reasoning_by_round[reasoning_idx].items():
                    messages[i][field] = value
                reasoning_idx += 1

        # Append the final output message to complete the conversation
        output_data = final_trace.get("output", {})
        choices = output_data.get("choices", [])
        if choices:
            assistant_message = choices[0].get("message", {}).copy()
            if assistant_message:
                # Add reasoning fields to the final message if available
                if reasoning_idx < len(reasoning_by_round):
                    for field, value in reasoning_by_round[reasoning_idx].items():
                        assistant_message[field] = value
                messages.append(assistant_message)

        # Build trajectory entry
        output = eval_entry.get("output", {})
        trajectory_entry = {
            "messages": messages,
            "tools": tools,
            "request_meta": request_meta,
            "response_meta": response_meta,
            "rounds": num_rounds,
            "all_turns_tokens": all_turns_tokens,
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

    # Compute and display statistics
    if trajectory_data:
        stats = compute_statistics(trajectory_data)
        print_statistics_console(stats, output_dir)

        # Save markdown
        stats_md_file = output_path / "trajectory_stats.md"
        save_statistics_markdown(stats, output_dir, str(stats_md_file))
        print(f"Statistics saved to {stats_md_file}")

        # Save JSON
        stats_json_file = output_path / "trajectory_stats.json"
        with open(stats_json_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Statistics saved to {stats_json_file}")


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