from typing import Any
from dotenv import load_dotenv
import json
import opik
import asyncio
from opik.evaluation.metrics.score_result import ScoreResult

from opik_optimizer import (
    ChatPrompt,
    MetaPromptOptimizer,
    EvolutionaryOptimizer
)
from opik_optimizer.datasets import hotpot_300
from opik_agent import OpenAIAgent, get_tools_schema
from deepsearch_agent import AGENT_PROMPTS


from utils import compute_scores, extract_answer

load_dotenv()

def compute_score(dataset_item: dict[str, Any], llm_output: str) -> ScoreResult:
    score = compute_scores(llm_output, dataset_item["golden_answers"][0])
    # print(f"Answer: {llm_output}, golden_answer: {dataset_item["golden_answers"][0]}, score: {score}")
    return ScoreResult(
        name="f1",
        value=score,
    )

def levenshtein_ratio(dataset_item: dict[str, Any], llm_output: str) -> ScoreResult:
    metric = LevenshteinRatio()
    pred = extract_answer(llm_output)
    return metric.score(reference=dataset_item["golden_answers"][0], output=pred)

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


dataset = multihop_rag(split="train", nb_items=100)

system_prompt = AGENT_PROMPTS['MultiHop-RAG-NoThink']

tools_schema = asyncio.run(get_tools_schema())
prompt = ChatPrompt(
    system=system_prompt,
    user="{question}",
    tools=tools_schema
)

# Optimize it:
optimizer = EvolutionaryOptimizer(
    model="openai/openrouter/openai/gpt-5-mini",   # Using gpt-4o-mini for evaluation for speed
    population_size=30,
    num_generations=15,
    enable_moo=False,
    enable_llm_crossover=True,
    infer_output_style=True,
    verbose=1,
    n_threads=12,
)

optimization_result = optimizer.optimize_prompt(
    prompt=prompt,
    agent_class=OpenAIAgent,
    dataset=dataset,
    metric=compute_score,
    n_samples=50,
)
optimization_result.display()