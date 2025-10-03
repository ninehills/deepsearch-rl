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
pip install flash-attn --no-build-isolation

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
model_name=`echo $model | tr '/:' '-'`
prompt_name="MultiHop-RAG-NoThink"
python deepsearch_agent.py run --base_url http://localhost:8000/v1 --api_key EMPTY --prompt-name "$prompt_name" --dataset ./data/MultiHop-RAG/_data/val.jsonl --do_eval --model "$model" --output_dir output/"$prompt_name"/"$model_name"
python analyze_trajectory.py --output_dir output/"$prompt_name"/"$model_name" --with_eval

model="models/Qwen3-4B-Thinking-2507"
model_name=`echo $model | tr '/:' '-'`
prompt_name="MultiHop-RAG"
python deepsearch_agent.py run --base_url http://localhost:8000/v1 --api_key EMPTY --prompt-name "$prompt_name" --dataset ./data/MultiHop-RAG/_data/val.jsonl --do_eval --model "$model" --output_dir output/"$prompt_name"/"$model_name"
python analyze_trajectory.py --output_dir output/"$prompt_name"/"$model_name" --with_eval
```

| 模型 | Prompt | topK | chunk_size(tokens) | 结果（F1） |
| --- | --- | --- | --- | --- | 
| Qwen3-4B-Instruct-2507 | MultiHop-RAG-NoThink | 3 | 200 | 0.521 |
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
model_name=`echo $model | tr '/:' '-'`
prompt_name="MultiHop-RAG-NoThink"
python deepsearch_agent.py run --prompt-name "$prompt_name" --dataset ./data/MultiHop-RAG/_data/train.jsonl --do_eval --model "$model" --output_dir output/train/"$prompt_name"/"$model_name"
python analyze_trajectory.py --output_dir output/train/"$prompt_name"/"$model_name" --with_eval

# 过滤出1504条成功的轨迹
Evaluation results: {'em': 0.6370967741935484, 'f1': 0.6417889788008634, 'acc': 0.6481324278438031, 'precision': 0.642225
5234861347, 'recall': 0.6426651305683564}

                         Conversation Dynamics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric                    ┃         All ┃     Success ┃     Failure ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Avg Rounds                │ 3.37 ± 0.97 │ 3.26 ± 0.96 │ 3.57 ± 0.96 │
│ Avg Tool Calls            │ 2.51 ± 0.93 │ 2.37 ± 0.93 │ 2.77 ± 0.89 │
└───────────────────────────┴─────────────┴─────────────┴─────────────┘
                    Token Usage (Last Turn)
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Metric               ┃        All ┃    Success ┃    Failure ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Total                │ 2211 ± 699 │ 2096 ± 687 │ 2415 ± 673 │
│ Prompt               │ 2176 ± 698 │ 2067 ± 683 │ 2368 ± 683 │
│ Completion           │    35 ± 59 │    28 ± 60 │    47 ± 55 │
└──────────────────────┴────────────┴────────────┴────────────┘
               Token Usage (All Turns - Cumulative)
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric               ┃         All ┃     Success ┃     Failure ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Total                │ 4726 ± 2503 │ 4391 ± 2395 │ 5318 ± 2580 │
│ Prompt               │ 4594 ± 2471 │ 4271 ± 2362 │ 5163 ± 2556 │
│ Completion           │    132 ± 78 │    120 ± 74 │    155 ± 81 │
└──────────────────────┴─────────────┴─────────────┴─────────────┘

                  Round Distribution
┏━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃  Rounds  ┃         All ┃     Success ┃     Failure ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│    2     │ 497 (21.1%) │ 380 (25.3%) │ 117 (13.7%) │
│    3     │ 824 (35.0%) │ 525 (34.9%) │ 299 (35.1%) │
│    4     │ 710 (30.1%) │ 436 (29.0%) │ 274 (32.2%) │
│    5     │ 317 (13.5%) │ 159 (10.6%) │ 158 (18.5%) │
│    6     │    8 (0.3%) │    4 (0.3%) │    4 (0.5%) │
└──────────┴─────────────┴─────────────┴─────────────┘
                 Tool Calls Distribution
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃  Tool Calls  ┃         All ┃     Success ┃     Failure ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│      1       │ 280 (11.9%) │ 263 (17.5%) │   17 (2.0%) │
│      2       │ 982 (41.7%) │ 608 (40.4%) │ 374 (43.9%) │
│      3       │ 734 (31.2%) │ 459 (30.5%) │ 275 (32.3%) │
│      4       │ 337 (14.3%) │ 164 (10.9%) │ 173 (20.3%) │
│      5       │    9 (0.4%) │    6 (0.4%) │    3 (0.4%) │
│      6       │   14 (0.6%) │    4 (0.3%) │   10 (1.2%) │
└──────────────┴─────────────┴─────────────┴─────────────┘
```

### 3.2 Non-Thinking 模型训练和评估

```bash
python convert_tool_calling_dataset.py --type swift --input_path output/train/MultiHop-RAG-NoThink/openrouter-qwen-qwen3-30b-a3b-instruct-2507/trajectory_success.jsonl  --output_path output/train/MultiHop-RAG-NoThink/openrouter-qwen-qwen3-30b-a3b-instruct-2507/trajectory_success_swift.jsonl

swift sft \
    --model models/Qwen3-4B-Instruct-2507 \
    --dataset output/train/MultiHop-RAG-NoThink/openrouter-qwen-qwen3-30b-a3b-instruct-2507/trajectory_success_swift.jsonl\
    --load_from_cache_file true \
    --train_type lora \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --agent_template hermes \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
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
    --eval_steps 10 \
    --save_strategy steps \
    --save_steps 10 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --max_length 8192 \
    --save_only_model true \
    --packing true \
    --output_dir output/lora/ \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --attn_impl flash_attn \
    --dataloader_num_workers 4 \
    --dataset_num_proc 16 \
    --use_liger_kernel true

# 启动 tensorboard
tensorboard --logdir output/lora/
```

启动推理
```bash
# 使用 --enforce-eager 不加载 CUDA Graph，提升启动速度,同时可以加载多个lora modules
vllm serve models/Qwen3-4B-Instruct-2507 --enforce-eager \
    --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes \
    --enable-lora --max-lora-rank 64 \
    --lora-modules 4b-sft-cpkt96=output/lora/v1-20251003-202557/checkpoint-96


model=4b-sft-cpkt96
model_name=`echo $model | tr '/:' '-'`
prompt_name="MultiHop-RAG-NoThink"
python deepsearch_agent.py run --base_url http://localhost:8000/v1 --api_key EMPTY --prompt-name "$prompt_name" --dataset ./data/MultiHop-RAG/_data/val.jsonl --do_eval --model "$model" --output_dir output/"$prompt_name"/"$model_name"
python analyze_trajectory.py --output_dir output/"$prompt_name"/"$model_name" --with_eval
```

比对微调前后，有较大提升

| 模型 | Prompt | topK | chunk_size(tokens) | 结果（F1） |
| --- | --- | --- | --- | --- | 
| Qwen3-4B-Instruct-2507 | MultiHop-RAG-NoThink | 3 | 200 | 0.521 |
| 4b-sft-cpkt96 | MultiHop-RAG-NoThink |3 | 200 | 0.751 |

比较轮次和工具使用，发现微调后，轮次和工具调用的数据量都有所增长。

```bash
# 微调前
                         Conversation Dynamics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Metric                    ┃         All ┃     Success ┃     Failure ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Avg Rounds                │ 2.37 ± 0.70 │ 2.34 ± 0.68 │ 2.39 ± 0.73 │
│ Avg Tool Calls            │ 1.96 ± 1.04 │ 1.81 ± 1.10 │ 2.11 ± 0.95 │
└───────────────────────────┴─────────────┴─────────────┴─────────────┘
                 Round Distribution
┏━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃  Rounds  ┃         All ┃    Success ┃    Failure ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│    2     │ 153 (76.5%) │ 80 (77.7%) │ 73 (75.3%) │
│    3     │  21 (10.5%) │ 11 (10.7%) │ 10 (10.3%) │
│    4     │  26 (13.0%) │ 12 (11.7%) │ 14 (14.4%) │
└──────────┴─────────────┴────────────┴────────────┘
                Tool Calls Distribution
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃  Tool Calls  ┃        All ┃    Success ┃    Failure ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│      1       │ 88 (44.0%) │ 59 (57.3%) │ 29 (29.9%) │
│      2       │ 55 (27.5%) │ 18 (17.5%) │ 37 (38.1%) │
│      3       │ 37 (18.5%) │ 15 (14.6%) │ 22 (22.7%) │
│      4       │  18 (9.0%) │   9 (8.7%) │   9 (9.3%) │
│      5       │   2 (1.0%) │   2 (1.9%) │   0 (0.0%) │
└──────────────┴────────────┴────────────┴────────────┘
# 微调后
Conversation Dynamics

| Metric | All | Success | Failure |
|--------|-----|---------|---------|
| Avg Rounds | 3.35 ± 0.90 | 3.34 ± 0.94 | 3.41 ± 0.75 |
| Avg Tool Calls | 2.35 ± 0.90 | 2.34 ± 0.94 | 2.41 ± 0.75 |

Round Distribution
┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃  Rounds  ┃        All ┃    Success ┃    Failure ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│    2     │ 33 (16.5%) │ 29 (19.5%) │   4 (7.8%) │
│    3     │ 87 (43.5%) │ 61 (40.9%) │ 26 (51.0%) │
│    4     │ 56 (28.0%) │ 39 (26.2%) │ 17 (33.3%) │
│    5     │ 24 (12.0%) │ 20 (13.4%) │   4 (7.8%) │
└──────────┴────────────┴────────────┴────────────┘
                Tool Calls Distribution
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃  Tool Calls  ┃        All ┃    Success ┃    Failure ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│      1       │ 33 (16.5%) │ 29 (19.5%) │   4 (7.8%) │
│      2       │ 87 (43.5%) │ 61 (40.9%) │ 26 (51.0%) │
│      3       │ 56 (28.0%) │ 39 (26.2%) │ 17 (33.3%) │
│      4       │ 24 (12.0%) │ 20 (13.4%) │   4 (7.8%) │
└──────────────┴────────────┴────────────┴────────────┘
```

### 3.3 Thinking 数据合成

TODO：使用 Thinking 模型合成

### 3.4 Thinking 模型训练和评估

Thinking 模型训练的问题是会Chat Template 会删除掉历史 Thinking 信息，需要将多轮调用打平成单轮调用以保留 Thinking 信息。

分别测试不打平和打平两种实现方式的效果。


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

# 使用的 GSPO
python art_rollout.py train "models/Qwen3-4B-Instruct-2507" "qwen3-4b-rlvr-03" --max_seq_length 8192
# 配置8192 限制就会限制总的轮次，如果输入超长就会400错误，奖励为0

# 默认参数在 https://github.com/OpenPipe/ART/blob/main/src/art/dev/get_model_config.py
# 通过 https://github.com/OpenPipe/ART/blob/main/src/art/dev/model.py 中的内容进行修改。

```
