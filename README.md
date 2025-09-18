# DeepSearch-RL

DeepSearch-RL is a reinforcement learning project for train a RAG Agent.

## Setup Environment

```bash
conda create -n deepsearch-rl python=3.12
conda activate deepsearch-rl
python -m pip install --upgrade pip
pip install packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
# This has to be pinned for VLLM to work.
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
# 如果需要很久，去 https://github.com/Dao-AILab/flash-attention/releases 找构建后的版本。

cd agent-lighting/
pip install -e .[dev,agent]

cd ../FlashRAG
pip install -e .
pip install pyseismic-lsr --no-deps
cd ../

pip install -r requirements.txt
```

## 1. Download corpus dataset and setup MCP server

### 1.1 【待办】wiki-18 数据集

参见 data/wiki_retriever_mcp/README.md

### 1.2 【待办】GraphRAG-Bench 数据集

参见 data/GraphRAG-Bench/README.md

### 1.3 【完成】MultiHop-RAG 数据集

参见 data/MultiHop-RAG/README.md

准备好后，启动 MCP 服务

```bash
cd data/MultiHop-RAG/
python ../retriever_mcp.py \
    --vector_index_path _data/e5_Flat.index \
    --bm25_index_path _data/bm25/ \
    --model_path ../wiki_retriever_mcp/_data/e5-base-v2 \
    --instruction "query: " \
    --corpus_path _data/chunks.jsonl \
    --use_multi_retriever \
    --merge_method rrf \
    --device cpu \
    --top_k 3
```

## 2. Agent DeepSearch 实现 & 评测

基于OpenAI Agents SDK 开发最简单的 DeepSearch 实现。

评测 MultiHop-RAG 数据集

```bash
# 需要对应的模型支持 Function callings
model="z-ai/glm-4.5"
model_remove_slash=${model//\//-}
python deepsearch-agent.py run --endpoint https://openrouter.ai/api/v1 --api-key <api_key> --prompt-name "MultiHop-RAG" --dataset ./data/MultiHop-RAG/_data/val.jsonl --do_eval --model "$model" --output_dir output/multihop-rag/"$model_remove_slash"
```

MultiHop-RAG 数据集（200条验证集）评测结果

| 模型 | topK | chunk_size(tokens) | 结果（F1） |
| --- | --- | --- | --- |
| qwen3-30b-a3b-instruct-2507 | 3 | 500 | 0.601 |
| kimi-k2-0905 | 3 | 500 | 0.513 |
| gpt-4o-mini | 3 | 500 | 0.595 |
| deepseek-chat-v3-0324 | 3 | 500 | 0.493 |


小模型启动命令：

```bash
vllm serve models/Qwen3-4B-Instruct-2507 --max-model-len 90000 --enable-auto-tool-choice --tool-call-parser hermes

model="models/Qwen3-4B-Instruct-2507"
model_remove_slash=${model//\//-}
python deepsearch-agent.py run --endpoint http://localhost:8000/v1 --api-key EMPTY --prompt-name "MultiHop-RAG" --dataset ./data/MultiHop-RAG/_data/val.jsonl --do_eval --model "$model" --output_dir output/multihop-rag/"$model_remove_slash"

vllm serve models/Qwen3-4B-Thinking-2507 --max-model-len 90000 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1
model="models/Qwen3-4B-Thinking-2507"

vllm serve models/Qwen3-1.7B --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes
model="models/Qwen3-1.7B"

```

| 模型 | topK | chunk_size(tokens) | 结果（F1） |
| --- | --- | --- | --- |
| Qwen3-4B-Instruct-2507 | 3 | 500 | 0.499 |
| Qwen3-4B-Thinking-2507 | 3 | 500 | 0.444 |
| Qwen3-1.7B | 3 | 500 | 0.375 |

Qwen3-1.7B 使用non-thinking模板，但是第二轮还是会进入thinking状态


## 3. 合成 Agent 轨迹，对小模型进行 SFT

挑选 Qwen3-1.7B non-thinking 模型，对其进行 SFT

1. 使用 qwen3-30B 模型合成轨迹，并过滤出正确的轨迹。
2. 使用 ms-swift / verl 进行 SFT
3. 评估 SFT 效果。

## 4. 使用 RL 进行模型训练

挑选 Qwen3-1.7B non-thinking 模型，对其进行 RL 训练，使用 Lora 训练


 


