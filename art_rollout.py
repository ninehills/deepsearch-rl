
import os
import weave
from dotenv import load_dotenv
import art
from art.local import LocalBackend
from art.rewards import ruler_score_group
from art.types import MessagesAndChoices, Message, Choice
from art.utils import iterate_dataset
from deepsearch_agent import DeepSearchAgent
from pydantic import BaseModel
from typing import List, Literal, Any, Dict
import json
import functools
import asyncio

from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai.resources.chat.completions.completions import AsyncCompletions
from openai.types.chat.chat_completion import Choice

from utils import compute_scores

load_dotenv()

PROJECT = "deepsearch-agent-art"


if os.getenv("WANDB_API_KEY", ""):
    weave.init(PROJECT, settings={"print_call_link": False})


class AsyncOpenAITrajectory(AsyncOpenAI):
    """OpenAI client that captures all non-streaming requests and responses."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.request_response_pairs: List[Dict[str, Any]] = []
        self._original_create = self.chat.completions.create
        self._patch_instance()

    def _patch_instance(self):
        """Patch this instance's create method to capture trajectories."""
        original_create = self._original_create

        @functools.wraps(original_create)
        async def patched_create(*args, **kwargs):
            # Execute the original function
            result = await original_create(*args, **kwargs)

            # Only capture non-streaming responses
            if not hasattr(result, '__aiter__'):
                self.request_response_pairs.append({
                    "request": kwargs,
                    "response": result
                })

            return result

        self.chat.completions.create = patched_create

def convert_history(request_response_pairs: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], MessagesAndChoices]:
    """
    Convert request-response pairs to tools definition and messages_and_choices format.

    Each request contains the full conversation history up to that point.
    We need to extract only the new messages from each round and the corresponding choices.

    Args:
        request_response_pairs: List of dicts with 'request' and 'response' keys
                                Each request contains cumulative messages, responses contain new choices

    Returns:
        tools: List of tool definitions (extracted from first request with tools)
        messages_and_choices: Alternating list of messages and choices representing the conversation flow
    """
    # print(f"request_response_pairs: {request_response_pairs}")
    tools = []
    messages_and_choices: MessagesAndChoices = []

    # Extract tools from the first request that has them
    tools = request_response_pairs[0]["request"]["tools"]

    # Track the previous message count to identify new messages
    previous_message_count = 0

    # Process each request-response pair
    for pair in request_response_pairs:
        request = pair["request"]
        response = pair["response"]
        
        current_messages = request["messages"]
        current_count = len(current_messages)
        # Add only the new messages from this round
        # (messages that weren't in the previous request)
        new_messages = current_messages[previous_message_count:]
        if new_messages[0].get("role") == "assistant":
            new_messages = new_messages[1:]

        for msg in new_messages:
            messages_and_choices.append(msg)

        previous_message_count = current_count
        messages_and_choices.append(response.choices[0])

    # 在 messages_and_choices 中进行判断，不能有连续的 2 个 Choice 消息
    for i in range(len(messages_and_choices) - 1):
        if isinstance(messages_and_choices[i], Choice) and isinstance(messages_and_choices[i+1], Choice):
            print(f"找到连续的 2 个 Choice 消息: {messages_and_choices}")
            raise ValueError("不能有连续的 2 个 Choice 消息")
    return tools, messages_and_choices

class Scenario(BaseModel):
    id: int
    question: str
    golden_answers: List[str]
    split: Literal["train", "val"]

@weave.op
async def rollout(model: art.Model, scenario: Scenario, max_tokens: int = 4096) -> art.Trajectory:
    traj = art.Trajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "scenario_id": scenario.id,
            "error": "",
        },
        scenario=scenario,
    )

    openai_client = AsyncOpenAITrajectory(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key
    )

    deepsearch_agent = DeepSearchAgent(
        retriever_mcp_server_url="http://127.0.0.1:8099/sse",
        model=OpenAIChatCompletionsModel(
            model=model.name,
            openai_client=openai_client
        ),
        prompt_name='MultiHop-RAG-NoThink',
        temperature=1.0,
        max_tokens=max_tokens,
    )

    ret = await deepsearch_agent.run(scenario.question)
    if not openai_client.request_response_pairs:
        print(f"Agent run not captured any requests/responses.")
        traj.metadata['error'] = "No requests/responses captured."
        return traj.finish()
    
    traj.tools, traj.messages_and_choices = convert_history(openai_client.request_response_pairs)
    traj.metadata['pred'] = ret['pred']
    traj.metadata['answer'] = ret['answer']
    traj.metadata['error'] = ret['error']
    if ret["error"]:
        print(f"Agent run error: {ret['error']}")
        return traj.finish()
    if ret['pred']:
        # 如果能够解析出答案，格式奖励 0.5
        format_reward = 0.5
    else:
        format_reward = 0.0
    # 正确性奖励 0-2
    correct_reward = compute_scores(ret["answer"], scenario.golden_answers[0]) * 2
    traj.reward = correct_reward + format_reward

    return traj

async def load_model(args) -> art.TrainableModel:
    print(f"Loading model: {args}")
    # Declare the model
    model = art.TrainableModel(
        name=args.model_name,
        project=PROJECT,
        base_model=args.base_model,
    )

    # To run on a T4, we need to override some config defaults.
    engine_args = art.dev.EngineArgs(
        enforce_eager=True,
        enable_sleep_mode=not args.disable_sleep_mode
    )
    if args.gpu_memory_utilization:
        engine_args["gpu_memory_utilization"] = args.gpu_memory_utilization

    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=args.max_seq_length,
        ),
        engine_args=engine_args,
        _decouple_vllm_and_unsloth=args.decouple_vllm_and_unsloth,
    )

    # Initialize the server
    backend = LocalBackend(
        # Normally we don't want to run the server in-process, but for the output
        # to show up properly on Google Colab we'll enable this.
        in_process=True,
        path="./.art",
    )

    if args.resume_ckpt:
        print(f"Resume checkpoint: {args.resume_ckpt}")
        r_split = args.resume_ckpt.split(":")
        if len(r_split) != 2:
            raise ValueError(f"Resume checkpoint format error: {args.resume_ckpt}")
        resume_model = r_split[0]
        resume_step = int(r_split[1])
        # Copy the checkpoint from another model
        await backend._experimental_fork_checkpoint(
            model,
            from_model=resume_model,
            not_after_step=resume_step,  # Use checkpoint at or before step ${resume_step}
            verbose=True,
        )

    # Register the model with the local Backend (sets up logging, inference, and training)
    await model.register(backend)

    return model

async def rollout_test(args):
    model = await load_model(args)
    val_dataset_file = "data/MultiHop-RAG/_data/val_mini.jsonl"
    val_inputs: List[TaskInput] = []
    with open(val_dataset_file, "r") as f:
        for line in f:
            data = json.loads(line)
            scenario = Scenario(**data, split="val")
            val_inputs.append(scenario)
    for scenario in val_inputs[:1]:
        traj = await rollout(model, scenario, args.max_tokens)
        print(json.dumps(traj.for_logging(), ensure_ascii=False, indent=2))


async def train(args):
    training_config = {
        "groups_per_step": 2,
        "num_epochs": 1,
        "rollouts_per_group": 8,
        "learning_rate": 3e-5,
    }
    model = await load_model(args)
    train_dataset_file = "data/MultiHop-RAG/_data/train.jsonl"
    train_scenarios: List[Scenario] = []
    with open(train_dataset_file, "r") as f:
        for line in f:
            data = json.loads(line)
            scenario = Scenario(**data, split="train")
            train_scenarios.append(scenario)
    
    # Use iterate_dataset with real training scenarios (similar to train.py)
    train_iterator = iterate_dataset(
        train_scenarios,  # Use real scenarios from Hugging Face
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )
    # Main training loop using iterate_dataset
    for batch in train_iterator:
        print("Gathering trajectory groups with RULER scoring...")

        # Use gather_trajectory_groups with ruler_score_group
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, scenario, args.max_tokens)
                    for _ in range(training_config["rollouts_per_group"])
                )
                for scenario in batch.items
            ),
            pbar_desc=f"train gather step {batch.step}",
        )

        scored_groups = []
        for group in groups:
            # Use RULER to assign relative scores to each trajectory
            #judged_group = await ruler_score_group(
            #    group, judge_model=RULER_MODEL, debug=True, swallow_exceptions=True
            #)
            # 不使用 RULER，使用简单的奖励函数
            judged_group = group
            scored_groups.append(judged_group)

        print("starting train")
        await model.train(
            scored_groups,
            config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
            _config=art.dev.TrainConfig(
                # GSPO
                importance_sampling_level="sequence",
                epsilon=3e-4,
                epsilon_high=4e-4,
                # TIS
                truncated_importance_sampling=2.0,
                # 减少 Logprob 计算所需的内存
                logprob_calculation_chunk_size=8,
            )
        )


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="ART Rollout and Training")
    parser.add_argument("mode", choices=["train", "test"], help="Mode to run: train or test")
    parser.add_argument("base_model", help="Path to the base model")
    parser.add_argument("model_name", help="Name for the model")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Maximum sequence length (default: 8192)")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum tokens to generate (default: 4096)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0, help="GPU memory utilization (default: 0=auto)")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="resume checkpoint, <model_name>:<step> (default: None)")
    parser.add_argument("--disable_sleep_mode", action="store_true", help="Disable sleep mode (default: enabled)")
    parser.add_argument("--decouple_vllm_and_unsloth", action="store_true", help="Decouple vLLM and Unsloth (default: disabled)")

    args = parser.parse_args()

    if args.mode == "test":
        asyncio.run(rollout_test(args))
    elif args.mode == "train":
        asyncio.run(train(args))