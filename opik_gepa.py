
from typing import Any
import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from fastmcp import Client

import opik  # noqa: E402
from opik_optimizer import ChatPrompt  # noqa: E402
from opik_optimizer.gepa_optimizer import GepaOptimizer  # noqa: E402
from opik.evaluation.metrics.score_result import ScoreResult  # noqa: E402
from utils import compute_scores  # noqa: E402

load_dotenv()

from deepsearch_agent import AGENT_PROMPTS

# MCP Server URL
MCP_SERVER_URL = "http://127.0.0.1:8099/sse"

# Initialize MCP client
mcp_client = Client(MCP_SERVER_URL)


async def get_retrieve_tool_schema():
    """Get retrieve tool schema from MCP server"""
    async with mcp_client:
        tools = await mcp_client.list_tools()
        for tool in tools:
            if tool.name == "retrieve":
                return {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                }
        raise ValueError("retrieve tool not found in MCP server")


async def call_retrieve(query: str, with_meta: bool = False) -> str:
    """Call retrieve tool from MCP server"""
    async with mcp_client:
        result = await mcp_client.call_tool("retrieve", {"query": query, "with_meta": with_meta})

        # Parse the result from MCP CallToolResult
        # result.content is a list of TextContent objects
        # Loop through content list and convert using model_dump()
        ret = result.content[0].model_dump()

        # The first content item's 'text' field contains the actual JSON string with retrieval results
        # Format: [{"chunk": "...", "chunk_id": "...", "score": 0.123}, ...]
        return json.dumps(ret, ensure_ascii=False)


def retrieve(query: str, with_meta: bool = False):
    """Synchronous wrapper for MCP retrieve"""
    try:
        result = asyncio.run(call_retrieve(query, with_meta))
        return result
    except Exception as e:
        print(f"Error calling retrieve: {e}")
        import traceback
        traceback.print_exc()
        return ""


def multihop_rag(split: str, nb_items: int) -> opik.Dataset:
    """
    Dataset containing the first nb_items samples of the MultiHop-RAG dataset.
    """
    dataset_name: str = f"multihop_rag_{split}_{nb_items}"
    client = opik.Opik()
    dataset = client.get_or_create_dataset(dataset_name)

    items = dataset.get_items()
    if len(items) == nb_items:
        return dataset
    elif len(items) != 0:
        raise ValueError(
            f"Dataset {dataset_name} contains {len(items)} items, expected {nb_items}. We recommend deleting the dataset and re-creating it."
        )
    elif len(items) == 0:
        # Load data from file and insert into the dataset
        ds = []
        with open(f"data/MultiHop-RAG/_data/{split}.jsonl", "r") as f:
            for line in f:
                item = json.loads(line)
                item['_id'] = item['id']
                del item['id']
                ds.append(item)
        dataset.insert(ds[:nb_items])
        return dataset


def compute_score(dataset_item: dict[str, Any], llm_output: str) -> ScoreResult:
    score = compute_scores(llm_output, dataset_item["golden_answers"][0])
    # print(f"Answer: {llm_output}, golden_answer: {dataset_item["golden_answers"][0]}, score: {score}")
    return ScoreResult(
        name="compute_score",
        value=score,
    )


def to_dict(obj):
    """Recursively convert Pydantic models to dict using model_dump()"""
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(item) for item in obj]
    else:
        return obj


def test_mcp():
    """Test MCP tool definition and invocation"""
    print("\n" + "="*80)
    print("Testing MCP Tool Definition")
    print("="*80)

    # Expected tool definition (provided by user)
    expected_tool_def = {
        "type": "function",
        "function": {
            "name": "retrieve",
            "description": "retrieve relevant chunks from the corpus, with_meta MUST False",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "title": "Query",
                        "type": "string"
                    },
                    "with_meta": {
                        "default": False,
                        "title": "With Meta",
                        "type": "boolean"
                    }
                }
            }
        }
    }

    print("\n--- Expected Tool Definition (User Provided) ---")
    print(json.dumps(expected_tool_def, indent=2))

    # Get actual tool definition from MCP server
    print("\n--- Fetching Actual Tool Definition from MCP Server ---")
    try:
        actual_tool_def = asyncio.run(get_retrieve_tool_schema())
        print("\n--- Actual Tool Definition (From MCP Server) ---")
        print(json.dumps(actual_tool_def, indent=2))

        # Compare definitions
        print("\n--- Comparison ---")
        if actual_tool_def == expected_tool_def:
            print("✓ Tool definitions match perfectly!")
        else:
            print("✗ Tool definitions differ!")
            print("\nDifferences:")

            # Compare function name
            actual_name = actual_tool_def.get("function", {}).get("name")
            expected_name = expected_tool_def["function"]["name"]
            if actual_name != expected_name:
                print(f"  - Name: '{actual_name}' vs '{expected_name}'")
            else:
                print(f"  ✓ Name matches: '{actual_name}'")

            # Compare description
            actual_desc = actual_tool_def.get("function", {}).get("description")
            expected_desc = expected_tool_def["function"]["description"]
            if actual_desc != expected_desc:
                print(f"  - Description: '{actual_desc}' vs '{expected_desc}'")
            else:
                print(f"  ✓ Description matches")

            # Compare parameters
            actual_params = actual_tool_def.get("function", {}).get("parameters")
            expected_params = expected_tool_def["function"]["parameters"]
            if actual_params != expected_params:
                print(f"  - Parameters differ:")
                print(f"    Actual: {json.dumps(actual_params, indent=6)}")
                print(f"    Expected: {json.dumps(expected_params, indent=6)}")
            else:
                print(f"  ✓ Parameters match")

        # Test tool invocation
        print("\n--- Testing Tool Invocation ---")
        test_query = "What is machine learning?"
        print(f"Test query: '{test_query}'")
        print("Calling retrieve(query='What is machine learning?', with_meta=False)...")

        result = retrieve(test_query, with_meta=False)
        print(f"\n✓ Tool invocation successful!")
        print(f"Result: {result}")

    except Exception as e:
        print(f"✗ Error testing MCP tool: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)


def train(split: str = "train", nb_items: int = 100, n_samples: int = 50, max_metric_calls: int = 60):
    """Train prompt optimization with GEPA"""
    print("\n" + "="*80)
    print("Training GEPA Optimizer")
    print("="*80)

    # Load dataset
    print(f"\nLoading dataset: split={split}, nb_items={nb_items}")
    dataset = multihop_rag(split=split, nb_items=nb_items)
    print(f"Dataset loaded: {len(dataset.get_items())} items")

    # Get tool schema from MCP server
    print("\nFetching tool schema from MCP server...")
    retrieve_tool_schema = asyncio.run(get_retrieve_tool_schema())
    print(f"Tool schema fetched: {retrieve_tool_schema['function']['name']}")

    system_prompt = AGENT_PROMPTS["MultiHop-RAG-NoThink"]

    prompt = ChatPrompt(
        project_name="GEPA-MultiHop-RAG",
        system=system_prompt,
        user="{question}",
        tools=[retrieve_tool_schema],
        function_map={
            "retrieve": opik.track(type="tool")(
                lambda query, with_meta=False: retrieve(query, with_meta)
            )
        },
    )

    # Optimize it with GEPA
    print("\nInitializing GEPA optimizer...")
    optimizer = GepaOptimizer(
        model="hosted_vllm/Qwen3-4B-Instruct-2507",  # smaller task model (valid LiteLLM)
        reflection_model="openai/openrouter/Kimi/kimi-k2-turbo-preview",  # larger reflection model (valid LiteLLM)
        temperature=0.0,  # deterministic completions during optimization
        max_tokens=400,
    )

    print(f"\nStarting optimization: n_samples={n_samples}, max_metric_calls={max_metric_calls}")
    result = optimizer.optimize_prompt(
        prompt=prompt,
        dataset=dataset,
        metric=compute_score,
        n_samples=n_samples,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=5,
        candidate_selection_strategy="best",
        display_progress_bar=True,
    )

    details = result.details or {}
    summary = details.get("candidate_summary", [])

    print("\n=== GEPA Candidate Scores ===")
    for idx, row in enumerate(summary):
        print(
            f"#{idx:02d} source={row.get('source')} GEPA={row.get('gepa_score')} "
            f"Opik={row.get('opik_score')}"
        )

    print("\nSelected candidate:")
    print("  index:", details.get("selected_candidate_index"))
    print("  GEPA score:", details.get("selected_candidate_gepa_score"))
    print("  Opik score:", details.get("selected_candidate_opik_score"))

    print("\nPer-item scores for selected prompt:")
    for record in details.get("selected_candidate_item_scores", []):
        print(
            f"  id={record.get('dataset_item_id')} score={record.get('score'):.4f} "
            f"answer={record.get('answer')} output={record.get('output', '')[:60]}"
        )

    print("\n" + "="*80)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GEPA MultiHop-RAG Optimizer with MCP")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # test_mcp command
    test_parser = subparsers.add_parser('test_mcp', help='Test MCP tool definition and invocation')

    # train command
    train_parser = subparsers.add_parser('train', help='Train prompt optimization with GEPA')
    train_parser.add_argument('--split', type=str, default='train',
                             help='Dataset split (default: train)')
    train_parser.add_argument('--nb_items', type=int, default=100,
                             help='Number of dataset items (default: 100)')
    train_parser.add_argument('--n_samples', type=int, default=50,
                             help='Number of samples for optimization (default: 50)')
    train_parser.add_argument('--max_metric_calls', type=int, default=60,
                             help='Maximum metric calls (default: 60)')

    args = parser.parse_args()

    if args.command == 'test_mcp':
        test_mcp()
    elif args.command == 'train':
        train(
            split=args.split,
            nb_items=args.nb_items,
            n_samples=args.n_samples,
            max_metric_calls=args.max_metric_calls
        )
    else:
        parser.print_help()