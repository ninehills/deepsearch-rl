"""Simple DeepSearch Agent use OpenAI Agents SDK"""

from __future__ import annotations

import argparse
import json
import os
import asyncio
from pathlib import Path
from typing import Any
from openai_trace import AsyncOpenAITrace

import dotenv
from agents import (
    Agent,
    Runner,
    set_tracing_disabled,
    set_trace_processors,
)
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.mcp import MCPServerSse
from agents.model_settings import ModelSettings

from utils import extract_answer

dotenv.load_dotenv()

if os.getenv("AGENTOPS_API_KEY"):
    # AgentOps Enabled
    try:
        import agentops
        agentops.init()
    except ImportError:
        print("AgentOps not installed. Please install it with 'pip install agentops'")

if os.getenv("OPIK_BASE_URL"):
    print("opik base url: ", os.getenv("OPIK_BASE_URL"))
    try:
        from opik.integrations.openai.agents import OpikTracingProcessor
        set_trace_processors(processors=[OpikTracingProcessor(
            project_name="deepsearch-agent",
        )])
    except ImportError:
        print("Opik not installed. Please install it with 'pip install opik'")
else:
    set_tracing_disabled(True)

AGENT_PROMPTS = {
    "MultiHop-RAG": """You are an assistant who answers questions using retrieval_server. Answer the question using only the retrieved passages. Verify your answer directly against the text.

After each search:
- Summarize findings.
- Decide if info is sufficient.
  - If sufficient: reply in <answer>...</answer> with your answer. The answer must be extremely concise: a single word or entity.
  - If not: reply <answer>Insufficient Information</answer>.
- Explain your reasoning for the chosen action.

Repeat as needed. When done, wrap your final, concise answer in <answer> tags.""",
    "MultiHop-RAG-NoThink": """You are an assistant who answers questions using retrieval_server. Answer the question using only the retrieved passages. Verify your answer directly against the text.

After each search:
- Decide if info is sufficient.
  - If sufficient: reply in <answer>...</answer> with your answer. The answer must be extremely concise: a single word or entity.
  - If not: Generate more refined search queries to gather more relevant information, then perform more search using that queries.
- Max 5 search iterations.

Repeat as needed. When done, wrap your final, concise answer in <answer> tags. If the answer is not found, reply <answer>Insufficient Information</answer>."""
}

class DeepSearchAgent:
    def __init__(self,
            retriever_mcp_server_url: str,
            model: OpenAIChatCompletionsModel,
            prompt_name: str = "MultiHop-RAG",
            max_concurrent: int = 5,
            max_retries: int = 3,
            temperature: float = 0.7,
            max_tokens: int = 4096,
        ) -> None:
        self.retriever_mcp_server_url = retriever_mcp_server_url
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries

        if prompt_name not in AGENT_PROMPTS:
            raise ValueError(f"prompt_name must be one of {list(AGENT_PROMPTS.keys())}")
        self.agent_prompt = AGENT_PROMPTS[prompt_name]
        self.model = model
        self.model_settings = ModelSettings(
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    async def run(self, question: str) -> dict[str, Any]:
        ret = {
            "answer": "",
            "pred": "",
            "error": ""
        }
        try:
            async with MCPServerSse(
                name="retrieval_server",
                params={"url": self.retriever_mcp_server_url},
            ) as server:
                agent = Agent(
                    model=self.model,
                    model_settings=self.model_settings,
                    name="Assistant",
                    instructions=self.agent_prompt,
                    mcp_servers=[server],
                )
                result = await Runner.run(agent, question)
                ret["answer"] = result.final_output
                ret["pred"] = extract_answer(result.final_output)
        except Exception as e:
            print(f"search error: {e}")
            ret["error"] = f"search error: {e}"
        return ret

    async def run_with_retry(self, question: str) -> dict[str, Any]:
        for attempt in range(self.max_retries):
            result = await self.run(question)
            if not result["error"]:
                return result
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {result['error']}")
                await asyncio.sleep(wait_time)
        return result

    async def batch_run(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_task(task: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                result = await self.run_with_retry(task["question"])
                task_copy = task.copy()
                task_copy['output'] = result

                question_truncated = task["question"][:50] + "..." if len(task["question"]) > 50 else task["question"]
                pred_truncated = result["pred"][-30:] if len(result["pred"]) > 30 else result["pred"]
                print(f"Question: {question_truncated}")
                print(f"Pred: {pred_truncated} | Error: {result['error'] or 'None'}")
                print("-" * 50)

                return task_copy

        return await asyncio.gather(*[process_task(task) for task in tasks])


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: list[dict[str, Any]], file_path: str) -> None:
    """Save data to JSONL file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


async def run_batch_evaluation(args) -> None:
    """Run batch evaluation on dataset"""
    print(f"Loading dataset from: {args.dataset}")
    data = load_jsonl(args.dataset)

    if args.sample > 0:
        data = data[:args.sample]
        print(f"Sampled {len(data)} items")
    else:
        print(f"Using all {len(data)} items")

    # Initialize agent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    trace_file = output_dir / "openai_trace.log"

    model = OpenAIChatCompletionsModel(
        model=args.model,
        openai_client=AsyncOpenAITrace(
            base_url=args.base_url or os.getenv("OPENAI_BASE_URL", ""),
            api_key=args.api_key or os.getenv("OPENAI_API_KEY", ""),
            trace_file=trace_file
        )
    )

    agent = DeepSearchAgent(
        retriever_mcp_server_url="http://127.0.0.1:8099/sse",
        model=model,
        prompt_name=args.prompt_name,
        max_concurrent=args.max_concurrent,
        max_retries=3
    )

    # Run batch processing
    print("Starting batch processing...")
    results = await agent.batch_run(data)

    # Save results
    output_file = output_dir / "results.jsonl"
    save_jsonl(results, str(output_file))
    print(f"Results saved to: {output_file}")

    # Evaluation if requested
    if args.do_eval:
        await run_flashrag_evaluation(str(output_file), output_dir)


async def run_flashrag_evaluation(dataset: str, output_dir: Path) -> None:
    """Run FlashRAG evaluation on results"""
    try:
        from flashrag.evaluator import Evaluator
        from flashrag.dataset import Dataset
    except ImportError:
        print("FlashRAG not installed. Cannot run evaluation.")
        return

    print("Running FlashRAG evaluation...")
    dataset = Dataset(dataset_path=dataset)

    evaluator = Evaluator(
        config={
            "dataset_name": "default",
            "save_dir": str(output_dir),
            "save_metric_score": True,
            "save_intermediate_data": True,
            "metrics": ["em", "f1", "acc", "precision", "recall"],
        }
    )

    eval_results = evaluator.evaluate(dataset)
    print("Evaluation completed!")
    print(f"Evaluation results: {eval_results}")

async def run_evaluation_only(args) -> None:
    """Run evaluation only on existing results"""
    print(f"Loading results from: {args.dataset}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    await run_flashrag_evaluation(args.dataset, output_dir)


def main():
    parser = argparse.ArgumentParser(description="DeepSearch Agent CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run batch evaluation')
    run_parser.add_argument('--dataset', required=True, help='Dataset file path (JSONL format)')
    run_parser.add_argument('--prompt-name', choices=list(AGENT_PROMPTS.keys()), default='MultiHop-RAG', help='Prompt name')
    run_parser.add_argument('--sample', type=int, default=0, help='Sample size (0 = all)')
    run_parser.add_argument('--model', default='gpt-4o-mini', help='Model name')
    run_parser.add_argument('--base_url', help='API endpoint, (if not set, read from .env)')
    run_parser.add_argument('--api_key', help='API key (if not set, read from .env)')
    run_parser.add_argument('--do_eval', action='store_true', help='Run evaluation')
    run_parser.add_argument('--output_dir', default='output/', help='Output directory')
    run_parser.add_argument('--max_concurrent', type=int, default=10, help='Max concurrent requests')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Run evaluation only on existing results')
    eval_parser.add_argument('--dataset', required=True, help='Results file path (JSONL format)')
    eval_parser.add_argument('--output_dir', default='output/', help='Output directory')

    args = parser.parse_args()

    # 如果output_dir 已经存在，则报错（否则 openai_trace.log 会继续追加，导致统计不准确）
    if os.path.exists(args.output_dir):
        raise ValueError(f"Output directory {args.output_dir} already exists. Please choose a different directory.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.command == 'run':
        asyncio.run(run_batch_evaluation(args))
    elif args.command == 'eval':
        asyncio.run(run_evaluation_only(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    # python deepsearch-agent.py run --dataset ./data/MultiHop-RAG/_data/val.jsonl --do_eval --model "qwen/qwen3-30b-a3b-instruct-2507" --output_dir output/qwen3-30b-instruct
    # python deepsearch-agent.py eval --dataset ./output/qwen3-30b-instruct/results.jsonl --output_dir output/qwen3-30b-instruct/
    main()