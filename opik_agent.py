import os
import asyncio
import dotenv
from opik_optimizer import (
    OptimizableAgent,
    ChatPrompt,
)
from opik import track

from deepsearch_agent import DeepSearchAgent
from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai.resources.chat.completions.completions import AsyncCompletions
from openai.types.chat.chat_completion import Choice
from fastmcp import Client

from utils import compute_scores

dotenv.load_dotenv()

MCP_SERVER_URL = "http://127.0.0.1:8099/sse"

async def get_tools_schema():
    """Get retrieve tool schema from MCP server"""
    mcp_client = Client(MCP_SERVER_URL)
    async with mcp_client:
        tools = await mcp_client.list_tools()
        schemas = []
        for tool in tools:
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
            })
        return schemas


class OpenAIAgent(OptimizableAgent):
    """Agent using Pydantic AI for optimization."""

    project_name: str = "deepsearch-agent-opik"

    def init_agent(self, prompt: ChatPrompt) -> None:
        """Initialize the agent with the provided configuration."""
        self.agent = DeepSearchAgent(
            retriever_mcp_server_url=MCP_SERVER_URL,
            model=OpenAIChatCompletionsModel(
                #model="openrouter/qwen/qwen3-30b-a3b-instruct-2507",
                model="Qwen3-4B-Instruct-2507",
                openai_client=AsyncOpenAI(
                    api_key="EMPTY",
                    base_url="http://127.0.0.1:8000/v1",
                    #api_key=os.getenv("OPENAI_API_KEY"),
                    #base_url=os.getenv("OPENAI_API_BASE_URL"),
                )
            ),
            prompt_name="MultiHop-RAG-NoThink",
            temperature=0.7,
            max_tokens=4096,
        )

    def invoke(self, messages: list[dict[str, str]], seed: int | None = None) -> str:
        system_prompt = ""
        user_prompt = ""

        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            elif message["role"] == "user":
                user_prompt = message["content"]
            else:
                raise Exception("Unknown message type: %r" % message)
        
        self.agent.agent_prompt = system_prompt
        ret = asyncio.run(self.agent.run(user_prompt))
        # if ret["error"]:
        #     raise Exception(f"Agent run error: {ret['error']}")
        
        return ret["answer"]
