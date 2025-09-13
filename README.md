# DeepSearch-RL

DeepSearch-RL is a reinforcement learning project for train a RAG Agent.

## Setup Environment

```bash
conda create -n deepsearch-rl python=3.12
conda activate deepsearch-rl
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
git clone https://github.com/RUC-NLPIR/FlashRAG.git
pip install pyseismic-lsr --no-deps
cd FlashRAG
pip install -e .

```

## 1. Download corpus dataset and setup MCP server

### 1.1 wiki-18 数据集（暂时不用）

参见 data/wiki_retriever_mcp/README.md

### 1.2 GraphRAG-Bench 数据集

参见 data/GraphRAG-Bench/README.md

### 1.3 MultiHop-RAG 数据集

参见 data/MultiHop-RAG/README.md

todo: 
1. 增加中文BM25搜索

## 2. Agent DeepSearch 实现

## 3. RAG Evaluation

## 4. 更换到最佳 Baseline

Best Retriever + Best Model + Best Workflow

## 5. 使用 RL 进行模型训练

## 6. 合成 Agent 轨迹，使用 SFT 进行模型训练
 


