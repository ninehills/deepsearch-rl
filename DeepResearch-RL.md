# Deep Research RL 实现

## 0. DeepSearch vs DeepResearch

- DeepSearch: 产品定位为搜索问答，Plan + (Search + Reflect) x N + Final Answer，可以用单 Agent 实现，也可以拆分为多个 Agent
- DeepResearch：产品定位为报告写作，比 DeepSearch 运行时间更久，采集的信息更多，报告更丰富（包含多模态信息）。

本项目会实现单 Agent 的 DeepSearch，甚至简化到 Plan 都没有（可以通过增加 Plan Agent或者 Plan Tool的方式增加）

## 1. Deep Search 复现

### 1.1 工具调用复现方案

- **方案A**：类似于`<think></think><search></search><result></result>` 自定义标签，比如 Search-R1、VerlTool/deepsearch 等实现。
	- 优点是定义简单，利用stop words 实现工具调用。不依赖模型自身的 Tool calling 能力。
	- 缺点是仅支持搜索工具，并没有和模型自身的工具调用相结合。而且不经过训练基本无法开箱即用。
	- 目前多应用在学术研究中， https://github.com/RUC-NLPIR/FlashRAG/blob/main/flashrag/pipeline/reasoning_pipeline.py 对类似方法进行统一支持。
	- AutoCoA 论文实现了类似于百度 ERNIE4.5 的Action in Thinking 功能，比较新颖。
- **方案B**：模型本身支持 Tool Calling 能力（推理可在 vLLM 中配置 tool-parse和chat template）。此时搜索可以作为单独的Tool / MCP Tool。
	- 优点是复用模型自身能力，且训练方法为通用的 Agent 训练方法（不管是SFT还是RL）。同时很容易嵌入任意 Agent 框架（Dify 等）。
	- 缺点是依赖模型自身 ChatTemplate 支持。
		- 可通过 SFT 的方式，选择如 Hermes 等比较通用的 Tool Calling 模板将能力训练到模型中。
		- Reasoning Tool Calling 和 Tool Calling 模板有所不同。
- **方案C**：使用 smolagents 等 Agent 框架，使用 Prompt 的方法实现通用工具调用。
	- 优点是可灵活调整，比如 smolagents 支持 CodeAgent，可方便实现多工具调用。
	- 缺点是和框架强绑定，训练需要适配对应模板调整。

### 1.2 实际选型

1. 搜索仅是第一步，目标是做通用 Agent 训练和落地，所以不选择方案A。
2. 在方案B和方案C 中，我们**选择方案 B**，主要是考虑到目前下游 Agent 平台的兼容性。同时可以将搜索工具也固定为 MCP Tool。
3. RL 训练时考虑 Reasoning Tool Calling 以提升效果。
4. 没有 Reasoning 的模型可以参考agno、claude增加think工具。

## 2. 最终实现全过程

### 2.1 模型选择

支持 Tool Calling 的模型（包括小参数用于训练的模型和大参数用于实现对比效果的模型）
1. **旗舰模型**：Kimi-K2、DeepSeek-V3-0324、Qwen3-235B-2507-Instruct、GLM-4.5
2. **小参数模型**：Qwen3-4B-Instruct-2507、Qwen3-4B-Thinking-2507、Qwen3-4B-Base、Qwen3-1.7B
	- Thinking 的用于训练 Reasoning Tool Calling，或者仅是利用其模板。

### 2.2 Agent框架搭建

使用支持 Tool Calling 的 Agent 框架搭建基本的 DeepSearch Agent

#### 2.2.1 根本实现
1. DeepSearch 和 DeepResearch不同，为单一 Agent 架构（最多是引入并行 Query）。
2. 而 DeepResearch 目前实现更多是MultiAgent或者Workflow的有序编排。

#### 2.2.2 Agent框架选择要求
1. 贴合生产和实际落地，有广泛的社区支持。
2. 核心逻辑简洁清晰。
3. 支持 MCP 工具。
4. 支持 Tool Calling Agent。
5. 【加分】有良好的 Web 展示页面。

#### 2.2.3 Agent框架候选
1. LangGraph
2. OpenAI Agents SDK
3. Dify
4. Agno
5. AgentScope：开源了Studio，同时使用Formatter兼容ToolCall和ReAct 工具调用。
6. Pydantic AI

#### 2.2.4 最终选择
**OpenAI Agents SDK**

### 2.3 数据集选择

选择合适的数据集，用旗舰模型和小参数模型实现 Baseline（对比 w/o search、w/ search、w/ agent）

#### 2.3.1 WikiSearch数据集
**WikiSearch（避免数据成本）**，参考 FlashRAG 建设本地 Wiki 检索，使用 SingleHot、MultiHot 的评测数据。
- 使用 FlashRAG 基线检索方案

#### 2.3.2 自定义领域数据集
自定义领域数据集（金融等），参考 OpenBMB/RAGEval、ChineseSimpleQA 等库合成 QA 对。
- TODO：合成高难度 RAG 评测集的方法
- 使用最佳实践（混合检索+父子检索），可直接使用 AppBuilder 平台。

### 2.4 模型SFT训练

#### 2.4.1 数据集
1. 合成或者利用已有数据集进行模型 SFT 训练
2. TODO：WikiSearch 已有数据集、合成 SFT 数据集的方法

#### 2.4.2 框架
**ms-swift**：方便通过插件机制配置不同部分的 Loss

#### 2.4.3 算力
**1 x RTX4090**

### 2.5 模型RL训练

#### 2.5.1 数据集
TODO，同 SFT 训练，但不需要轨迹。

#### 2.5.2 框架
1. OpenPipe/ART：比较成熟，而且支持RULER无奖励模型训练。
2. Verl-tool：example 中实现了DeepSearch和Search-R1，需要改造为 Tool Calling 的模板。
3. microsoft/agent-lightning：适合各种 Agent框架，example 中有 AgentsSDK + MCP + RAG 训练的例子。
4. https://github.com/Simple-Efficient/RL-Factory 有DeepSearch+MCP 的例子
5. https://github.com/WooooDyy/AgentGym-RL
6. https://github.com/MiroMindAI/MiroRL

#### 2.5.3 算力

**N x RTX4090**

## 3. 其他

### 3.1 进阶
- 用多Agent替代单Agent，比如拆分为 PlanAgent、SearchAgent和WriteAgent，本质是从 DeepSearch 进化为 DeepResearch。
- 使用真实的 RAG 数据集替代 WikiSearch 数据集。

### 3.2 思考
为什么不仅使用提示词实现，而是要做训练？
- 对准确率的要求永无止境。
- 训练的本质是学习模型 Post-training 范式，最终还是为了训练出通用模型。

### 3.3 其他工具
- https://github.com/AgentOps-AI/agentops Trace
- https://github.com/SkyworkAI/DeepResearchAgent 基于SmolAgents的DeepResearch 多Agent实现，不做训练的话价格足够简单。
- https://github.com/openai/openai-agents-python/tree/main/examples/research_bot OpenAI Agents SDK 实现的简单的 DeepResearch Bot

### 3.4 相关论文
SFR-DeepResearch: Towards Effective Reinforcement Learning for Autonomously Reasoning Single Agents
https://arxiv.org/abs/2509.06283

MedResearcher-R1: Expert-Level Medical Deep Researcher via A Knowledge-Informed Trajectory Synthesis Framework
https://arxiv.org/abs/2508.14880

### 3.5 RAG数据集

https://github.com/OpenBMB/RAGEval
- 这个数据集有点问题，方法之间差距不大

https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
- 用这个有意思，图rag 设计就偏向于多跳
- 但是原始数据集过小。 

https://github.com/yixuantt/MultiHop-RAG
- 多跳推理数据集，有700条corpus数据。

### 3.6 RAG框架

FlashRAG、UltraRAG 框架：
1. 引入研究框架可快速复用已有Pipeline（包括检索、生成和评估），但会引入额外复杂性和学习成本。
2. FlashRAG 基于代码，UltraRAG 基于配置文件。氛围上不喜欢 UltraRAG。
3. （结论）选择基于 FlashRAG 框架实现整体 Pipeline，从而降低工程化成本。
