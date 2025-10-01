# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any, cast

from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from agents.mcp import MCPServerSse
from agents.model_settings import ModelSettings
from utils import compute_scores

from agentlightning import (
    LLM,
    LitAgent,
    NamedResources,
    Trainer,
    configure_logger,
)

from deepsearch_agent import DeepSearchAgent

configure_logger()


class TrainableAgent(LitAgent[Any]):
    def __init__(self, trained_agents: str | None = None) -> None:
        super().__init__(trained_agents=trained_agents)
        self.temperature = 0.7
        self.deepsearch_agent = DeepSearchAgent(
            retriever_mcp_server_url="http://127.0.0.1:8099/sse",
            model="EMPTY",
            base_url="EMPTY",
            prompt_name='MultiHop-RAG',
            api_key='EMPTY',
            max_tokens=4096,
            temperature=self.temperature,
        )

    async def training_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:  # type: ignore
        llm: LLM = cast(LLM, resources.get("main_llm"))
        print("Training with model:", llm.model, "on endpoint:", llm.endpoint)
        self.deepsearch_agent.set_model(model_name=llm.model, base_url=llm.endpoint)
        # ret = {"answer": "","pred": "","error": ""}
        ret = await self.deepsearch_agent.run(task["question"])
        reward = compute_scores(ret["answer"], task["golden_answers"][0])
        print(f"id: {task['id']} question: {task['question'][:20]} answer: {ret['answer'][:20]} pred: {ret['pred'][:20]} error: {ret['error'][:20]} reward: {reward}")
        if ret["error"]:
            raise Exception(f"Agent run error: {ret['error']}")

        return reward

    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:  # type: ignore
        llm: LLM = cast(LLM, resources.get("main_llm"))
        resources = {
            "main_llm": LLM(
                endpoint=llm.endpoint,
                model=llm.model,
                sampling_parameters={"temperature": self.temperature},
            )
        }
        return await self.training_rollout_async(task, rollout_id, resources)


if __name__ == "__main__":
    Trainer(n_workers=12).fit(RAGAgent(), "http://localhost:9999/")
