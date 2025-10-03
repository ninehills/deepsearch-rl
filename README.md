# DeepSearch-RL

DeepSearch-RL is a reinforcement learning project for train a RAG Agent.

## Setup Environment

如下环境用于检索服务、Agent以及模型 SFT 训练。

```bash
git submodule update --init --recursive
conda create -n deepsearch-rl python=3.12
conda activate deepsearch-rl
python -m pip install --upgrade pip
pip install packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
pip install -r requirements.txt

cd FlashRAG/
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
# follow the README.md to prepare data.
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

TODO：优化 Search MCP 的返回，更非结构一些。

## 2. Agent DeepSearch 实现 & 评测

基于OpenAI Agents SDK 开发最简单的 DeepSearch 实现。

评测 MultiHop-RAG 数据集，我们选择两个 Prompt：

- MultiHop-RAG：适合 Thinking 模型，参考 https://github.com/microsoft/agent-lightning/blob/main/examples/rag/rag_agent.py
- MultiHop-RAG-NoThink：适合 No-Thinking 模型（在多轮函数调用时没有思考过程）。

此外服务端默认支持并行工具调用。

```bash
# 需要对应的模型支持 Function callings
# 写入api_key base_url 到 .env
# 这里的模型填你的服务商的 model 名称。
model="openrouter/qwen/qwen3-30b-a3b-instruct-2507"
model_name=`echo $model | tr '/:' '-'`
prompt_name="MultiHop-RAG-NoThink"
# 运行 Agent，从 .env 读取 base_url 和 api_key
python deepsearch_agent.py run --prompt-name "$prompt_name" --dataset ./data/MultiHop-RAG/_data/val.jsonl --do_eval --model "$model" --output_dir output/"$prompt_name"/"$model_name"
# 分析日志，得到抽取的Tool calling 数据集
python analyze_trajectory.py --output_dir output/"$prompt_name"/"$model_name" --with_eval
# 日志输出到 trajectory.jsonl 文件，这个文件可以用来训练和评估模型
```

小模型启动命令：

```bash
# vllm 在 deepsearch-rl-vllm 环境中启动，避免冲突
vllm serve models/Qwen3-4B-Instruct-2507 --max-model-len 90000 --enable-auto-tool-choice --tool-call-parser hermes

model="models/Qwen3-4B-Instruct-2507"
model_name=`echo $model | tr '/:' '-'`
prompt_name="MultiHop-RAG-NoThink"
# prompt_name="MultiHop-RAG"
python deepsearch_agent.py run --base_url http://localhost:8000/v1 --api_key EMPTY --prompt-name "$prompt_name" --dataset ./data/MultiHop-RAG/_data/val.jsonl --do_eval --model "$model" --output_dir output/"$prompt_name"/"$model_name"
python analyze_trajectory.py --output_dir output/"$prompt_name"/"$model_name" --with_eval
```


```bash
vllm serve models/Qwen3-4B-Thinking-2507 --max-model-len 90000 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1
model="models/Qwen3-4B-Thinking-2507"
prompt_name="MultiHop-RAG-NoThink"
python deepsearch_agent.py run --base_url http://localhost:8000/v1 --api_key EMPTY --prompt-name "$prompt_name" --dataset ./data/MultiHop-RAG/_data/val.jsonl --do_eval --model "$model" --output_dir output/"$prompt_name"/"$model_name"
python analyze_trajectory.py --output_dir output/"$prompt_name"/"$model_name" --with_eval

model="models/Qwen3-4B-Thinking-2507"
prompt_name="MultiHop-RAG"
python deepsearch_agent.py run --base_url http://localhost:8000/v1 --api_key EMPTY --prompt-name "$prompt_name" --dataset ./data/MultiHop-RAG/_data/val.jsonl --do_eval --model "$model" --output_dir output/"$prompt_name"/"$model_name"
python analyze_trajectory.py --output_dir output/"$prompt_name"/"$model_name" --with_eval
```

| 模型 | Prompt | topK | chunk_size(tokens) | 结果（F1） |
| --- | --- | --- | --- | --- | 
| Qwen3-4B-Instruct-2507 | MultiHop-RAG-NoThink | 3 | 200 | 0.505 |
| Qwen3-4B-Instruct-2507 | MultiHop-RAG | 3 | 200 | 0.453 |
| Qwen3-4B-Thinking-2507 | MultiHop-RAG-NoThink | 3 | 200 | 0.458 |
| Qwen3-4B-Thinking-2507 | MultiHop-RAG | 3 | 200 | 0.385 |
| openrouter/qwen/qwen3-30b-a3b-instruct-2507 | MultiHop-RAG-NoThink |3 | 200 | 0.678 |
| openrouter/qwen/qwen3-30b-a3b-instruct-2507 | MultiHop-RAG |3 | 200 | 0.583 |

还是 NoThink Prompt 比较好，后续不管思考模型还是非思考模型都用这个 Prompt

## 3. 合成 Agent 轨迹，对小模型进行 SFT

挑选 Qwen3-4B-Instruct-2507 / Qwen3-4B-Thinking-2507模型，对其进行 SFT

1. 使用 qwen3-30B 模型合成轨迹，并过滤出正确的轨迹。
2. 使用 ms-swift 进行 SFT
3. 评估 SFT 效果。

### 3.1 Non-Thinking 数据合成

```bash
model="openrouter/qwen/qwen3-30b-a3b-instruct-2507"
# 将 model 中的 /: 替换为 -
model_name=`echo $model | tr '/:' '-'`
python deepsearch_agent.py run --prompt-name "MultiHop-RAG" --dataset ./data/MultiHop-RAG/_data/train.jsonl --do_eval --model "$model" --output_dir output/multihop-rag/train/"$model_name"

Evaluation results: {'em': 0.5980475382003395, 'f1': 0.6017025089605734, 'acc': 0.6035653650254669, 'precision': 0.6027164685908319, 'recall': 0.6014936534885601}


# 过滤出1411条成功的轨迹： output/multihop-rag/train/openrouter-qwen-qwen3-30b-a3b-instruct-2507/trajectory_success.jsonl
python analyze_trajectory.py --output_dir output/multihop-rag/train/"$model_name" --with_eval

```

### 3.2 Non-Thinking 模型训练和评估

```
python convert_tool_calling_dataset.py --type swift --input_path output/multihop-rag/train/openrouter-qwen-qwen3-30b-a3b-instruct-2507/trajectory_success.jsonl --output_path output/multihop-rag/train/openrouter-qwen-qwen3-30b-a3b-instruct-2507/trajectory_success_swift.jsonl

swift sft \
    --model models/Qwen3-4B-Instruct-2507 \
    --dataset output/multihop-rag/train/openrouter-qwen-qwen3-30b-a3b-instruct-2507/trajectory_success_swift.jsonl \
    --load_from_cache_file true \
    --train_type lora \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --agent_template hermes \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --report_to tensorboard \
    --do_eval true \
    --split_dataset_ratio 0.05 \
    --eval_steps 25 \
    --save_strategy steps \
    --save_steps 25 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --max_length 8192 \
    --save_only_model true \
    --packing true \
    --output_dir outputs \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --attn_impl flash_attn \
    --dataloader_num_workers 4 \
    --dataset_num_proc 16 \
    --use_liger_kernel true

# 启动 tensorboard
tensorboard --logdir outputs
```

启动推理
```bash
# 使用 --enforce-eager 不加载 CUDA Graph，提升启动速度
vllm serve models/Qwen3-4B-Instruct-2507 --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes --enable-lora --max-lora-rank 64 --lora-modules 4b-sft-cpkt100=outputs/v4-20250930-115648/checkpoint-100  --enforce-eager

model="4b-sft-cpkt100"
model_name=`echo $model | tr '/:' '-'`
python deepsearch_agent.py run --base_url http://localhost:8000/v1 --api_key EMPTY --prompt-name "MultiHop-RAG" --dataset ./data/MultiHop-RAG/_data/val.jsonl --do_eval --model "$model" --output_dir output/multihop-rag/"$model_name"
Evaluation results: {'em': 0.655, 'f1': 0.6658333333333333, 'acc': 0.66, 'precision': 0.67, 'recall': 0.6641666666666667}

vllm serve models/Qwen3-4B-Instruct-2507 --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes --enable-lora --max-lora-rank 64 --lora-modules 4b-sft-cpkt50=outputs/v4-20250930-115648/checkpoint-50  --enforce-eager

model="4b-sft-cpkt50"
model_name=`echo $model | tr '/:' '-'`
python deepsearch_agent.py run --base_url http://localhost:8000/v1 --api_key EMPTY --prompt-name "MultiHop-RAG" --dataset ./data/MultiHop-RAG/_data/val.jsonl --do_eval --model "$model" --output_dir output/multihop-rag/"$model_name"
Evaluation results: {'em': 0.675, 'f1': 0.6825, 'acc': 0.68, 'precision': 0.685, 'recall': 0.6816666666666668}


python analyze_trajectory.py --output_dir output/multihop-rag/"$model_name" --with_eval

```

发现大数据量训练的结果还不如小数据量，这主要是因为数据分布其实不太均匀。增加多tools call和多轮次数据，来进行数据平衡，效果会好一些。

### 3.3 Thinking 数据合成

### 3.4 Thinking 模型训练和评估


## 4. 使用 RL 进行模型训练

挑选 Qwen3-4B-Thinking-2507 / Qwen3-4B-Instruct-2507 模型，对其进行 RL 训练，使用 Lora 训练

### 4.1 Agent-Lightning【不支持Lora导致显存不足】

环境安装（和之前的不安装到一起）
```bash
conda create -n agent-lightning python=3.12
conda activate agent-lightning
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
pip install vllm==0.9.2
pip install verl==0.5.0

cd agent-lightning/
pip install -e .[dev,agent]
```

问题：
- verl 0.5.0 不兼容 vllm + lora:  https://github.com/volcengine/verl/issues/3271
- 升级 verl 为 main 分支后，agent-lightning 不兼容。

```bash
wandb login
shuf data/MultiHop-RAG/_data/val.jsonl  | head -20 > data/MultiHop-RAG/_data/val_mini.jsonl

# 转换 train.jsonl 和 val_mini.jsonl 为 parquet 格式，使用 Python -C
python convert_jsonl_to_parquet.py  data/MultiHop-RAG/_data/train.jsonl data/MultiHop-RAG/_data/train.parquet
python convert_jsonl_to_parquet.py  data/MultiHop-RAG/_data/val_mini.jsonl data/MultiHop-RAG/_data/val_mini.parquet

bash -x agent-lightning-train.sh
```

### 4.2 OpenPipe-ART

没有使用LLM的评分，而是使用绝对评分。

```bash
conda create -n openpipe-art python=3.12
conda activate openpipe-art
cd ART && pip install -e .[backend]
pip install "openai-agents==0.3.3"

python art_rollout.py train "qwen3-4b-001"
```
