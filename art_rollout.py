
import os
import weave
from dotenv import load_dotenv
from transformers import AutoTokenizer
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
import math

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

def calculate_rewards(ret: Dict[str, Any], scenario: Scenario, messages_and_choices: MessagesAndChoices, reward_list: List[str], max_tokens: int = 4096, tokenizer = None) -> Dict[str, float]:
    """
    Calculate individual rewards based on the reward list.

    Args:
        ret: Result dict from agent run
        scenario: The scenario object
        messages_and_choices: The conversation history
        reward_list: List of reward types to calculate
        max_tokens: Maximum tokens for soft_overlong calculation

    Returns:
        Dict mapping reward name to reward value
    """
    import re

    # short_think 奖励的 token 阈值配置
    THINK_TOKEN_MIN = 10       # 最小允许 token 数量下限
    THINK_TOKEN_OPTIMAL = 100  # 最优 token 数量上限
    THINK_TOKEN_MAX = 600      # 最大允许 token 数量上限

    rewards = {}

    for reward_type in reward_list:
        if reward_type == "correct":
            # 正确性奖励 0-2：最终的答案得分（F1分数） x 2
            rewards["correct"] = compute_scores(ret["answer"], scenario.golden_answers[0]) * 2

        elif reward_type == "short_think":
            # short_think (0-0.5)
            # 基于 <think> 标签内容的 tokens 数量计算奖励
            # 严格模式：任何一轮的 <think> 标签提取失败，整体奖励为 0（增大惩罚）
            # 每轮评分规则：
            #   - 没有 <think></think> 或出现多于1个：整体0分
            #   - tokens = 0：0.2分
            #   - 0 < tokens <= THINK_TOKEN_OPTIMAL：1分
            #   - THINK_TOKEN_OPTIMAL < tokens < THINK_TOKEN_MAX：按照 cosine 曲线从1分降低到0分
            #   - tokens >= THINK_TOKEN_MAX：0分
            # 最终奖励 = (各轮得分平均值) * 0.5

            # 检查 tokenizer 是否存在
            if tokenizer is None:
                print("警告: tokenizer 未提供，short_think 奖励设置为 0")
                rewards["short_think"] = 0.0
                continue

            round_scores = []
            has_format_error = False  # 标记是否有格式错误

            # 检查所有 assistant 消息
            for item in messages_and_choices:
                if isinstance(item, Choice):
                    content = item.message.content or ""

                    # 检查 <think> 和 </think> 标签是否各只出现一次
                    think_open_count = content.count('<think>')
                    think_close_count = content.count('</think>')

                    # 如果 <think> 或 </think> 不是恰好各出现一次，标记格式错误并立即退出
                    if think_open_count != 1 or think_close_count != 1:
                        has_format_error = True
                        break

                    # 获取 think 标签内的内容
                    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                    think_content = think_match.group(1).strip()

                    # 使用 tokenizer 计算 tokens 数量
                    tokens = tokenizer.encode(think_content, add_special_tokens=False)
                    token_count = len(tokens)

                    # 根据 token 数量计算得分
                    if token_count < THINK_TOKEN_MIN:
                        score = 0  # 小于 THINK_TOKEN_MIN 时得0分
                    elif token_count <= THINK_TOKEN_OPTIMAL:
                        score = 1.0  # 0 < tokens <= THINK_TOKEN_OPTIMAL，得满分
                    elif token_count < THINK_TOKEN_MAX:
                        # THINK_TOKEN_OPTIMAL < tokens < THINK_TOKEN_MAX，使用 cosine 曲线从1降到0
                        # cosine 曲线: score = (1 + cos(π * (token_count - THINK_TOKEN_OPTIMAL) / (THINK_TOKEN_MAX - THINK_TOKEN_OPTIMAL))) / 2
                        normalized = (token_count - THINK_TOKEN_OPTIMAL) / (THINK_TOKEN_MAX - THINK_TOKEN_OPTIMAL)  # 归一化到 [0, 1]
                        score = (1 + math.cos(math.pi * normalized)) / 2
                    else:
                        score = 0.0  # tokens >= THINK_TOKEN_MAX，得0分

                    round_scores.append(score)

            # 如果有任何一轮格式错误，整体奖励为 0
            if has_format_error:
                rewards["short_think"] = 0.0
            elif round_scores:
                # 所有轮次格式正确，计算平均奖励
                avg_score = sum(round_scores) / len(round_scores)
                rewards["short_think"] = avg_score * 0.5
            else:
                rewards["short_think"] = 0.0

        elif reward_type == "answer_format":
            # answer_format (0-0.5)
            # 检查预测答案是否为空
            # 不为空：0.5分，为空：0分
            pred_answer = ret.get("pred", "").strip()
            if pred_answer:
                rewards["answer_format"] = 0.5
            else:
                rewards["answer_format"] = 0.0

        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

    return rewards

@weave.op
async def rollout(model: art.Model, scenario: Scenario, max_tokens: int = 4096, rewards: List[str] = ["correct"], prompt_name: str = "MultiHop-RAG-NoThink", tokenizer = None) -> art.Trajectory:
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
        prompt_name=prompt_name,
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

    # 计算所有奖励
    reward_dict = calculate_rewards(ret, scenario, traj.messages_and_choices, rewards, max_tokens, tokenizer)

    # 总奖励是所有奖励的和
    traj.reward = sum(reward_dict.values())

    # 保存各个奖励到 metadata
    traj.metadata['reward_dict'] = reward_dict

    return traj

async def load_model(args) -> art.TrainableModel:
    print(f"Loading model: {args}")
    # Declare the model
    model = art.TrainableModel(
        name=args.model_name,
        project=PROJECT,
        base_model=args.base_model,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

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
        trainer_args=art.dev.TrainerArgs(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        ),
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

    return model, tokenizer

async def rollout_test(args):
    model, tokenizer = await load_model(args)
    val_dataset_file = "data/MultiHop-RAG/_data/val_mini.jsonl"
    val_inputs: List[Scenario] = []
    with open(val_dataset_file, "r") as f:
        for line in f:
            data = json.loads(line)
            scenario = Scenario(**data, split="val")
            val_inputs.append(scenario)
    for scenario in val_inputs[:1]:
        traj = await rollout(model, scenario, args.max_tokens, args.rewards, args.prompt_name, tokenizer)
        print(json.dumps(traj.for_logging(), ensure_ascii=False, indent=2))


async def train(args):
    training_config = {
        "groups_per_step": args.groups_per_step,
        "num_epochs": 1,
        # 这个框架不能调整 per_device_train_batch_size=2，那么最好的 gradient_accumulation_steps 是 rollouts_per_group / 2 的倍数
        # 但是因为默认使用了packing，也没啥意义了。
        "rollouts_per_group": 8, 
        "learning_rate": 3e-5,
    }
    model, tokenizer = await load_model(args)
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
                    rollout(model, scenario, args.max_tokens, args.rewards, args.prompt_name, tokenizer)
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
        gspo_config = art.dev.TrainConfig(
            # GSPO
            importance_sampling_level="sequence",
            epsilon=3e-4,
            epsilon_high=4e-4,
            # TIS
            truncated_importance_sampling=2.0,
            # 减少 Logprob 计算所需的内存
            logprob_calculation_chunk_size=8,
        )
        grpo_config = art.dev.TrainConfig(
            # TIS
            truncated_importance_sampling=2.0,
            # 减少 Logprob 计算所需的内存
            logprob_calculation_chunk_size=8,
        )
        await model.train(
            scored_groups,
            config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
            _config=gspo_config if args.method == "gspo" else grpo_config,
            verbose=True,
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
    parser.add_argument("--groups_per_step", type=int, default=2, help="Groups per step (default: 2)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--rewards", type=str, default="correct", help="Comma-separated list of rewards: correct,short_think (default: correct)")
    parser.add_argument("--method", choices=["grpo", "gspo"], default="gspo", help="Training method (default: gspo)")
    parser.add_argument("--prompt_name", type=str, default="MultiHop-RAG-NoThink", help="Prompt name (default: MultiHop-RAG-NoThink)")
    args = parser.parse_args()

    # Parse rewards string into list
    args.rewards = [r.strip() for r in args.rewards.split(",")]

    if args.mode == "test":
        asyncio.run(rollout_test(args))
    elif args.mode == "train":
        asyncio.run(train(args))