# wiki-18 数据集

Search-R1 项目提供了 wiki-18 数据集 corpus 文件和 intfloat/e5-base-v2 模型的 index 文件，我们需要下载后启动为检索用的 MCP Server。

也可以下载 FlashRAG 提供的物料： https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/files 中的 retrieval_corpus 目录。 

```bash
cd data/wiki18/
save_path="_data"

## 下载数据集
python download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
rm $save_path/part_*

# 默认是 e5_Flat.index 索引，Flat 版本需要 60G 内存占用，单次检索 3s 左右。
```

问题：
1. faiss-cpu 需要3-4s 检索时间
2. 需要60G 内存

解决办法：
1. 【放弃】使用faiss-gpu + 60+GB 显存
2. 【放弃】将FLAT索引修改为 HNSW索引，内存需求 60G+，但是检索速度更快。
3. FlashRAG 使用 datasets 库加载corpus，转成 Arrow 格式，使用memory mapping避免内存占用。
4. 【选中】使用IVF量化索引。

## 将索引转换为 量化索引

配置：
```bash
"nlist": 1024,    # IVF聚类中心数
"m": 96,          # PQ子向量数 (768/96=8维每个子向量)
"nbits": 8,       # 每个子向量量化比特数 (INT8)
```

对应的索引大小是2.1G。

```bash
python convert_index.py build
```

评测：
```bash
python convert_index.py benchmark
--- 开始运行CPU基准测试 ---

[1/3] 准备 100 个查询向量...
    - 索引加载成功。ntotal=21015324, d=768
    - 查询向量准备完毕。

[2/3] 在 CPU 上测试 IndexFlat (获取基准)...
    - 预热完成，开始测试。
    - 搜索耗时: 84001.19 ms
    - 速度 (QPS): 1.19

[3/3] 在 CPU 上测试 IndexIVFPQ...

----------------------------------------------------------------------
nprobe   | Recall@5  | Search Time (ms) | QPS      | Speedup vs Flat
---------|----------|--------------------|----------|-----------------
1        | 0.5980   | 12.35              | 8097.58  |     7266.77    x
2        | 0.6800   | 23.66              | 4226.08  |     3792.49    x
4        | 0.7560   | 41.70              | 2398.18  |     2152.13    x
8        | 0.7980   | 72.63              | 1376.92  |     1235.65    x
16       | 0.8120   | 138.54             | 721.82   |     647.76     x
32       | 0.8200   | 280.30             | 356.76   |     320.15     x
64       | 0.8200   | 568.12             | 176.02   |     157.96     x
128      | 0.8220   | 1160.07            | 86.20    |      77.36     x
256      | 0.8220   | 2321.04            | 43.08    |      38.66     x
512      | 0.8220   | 4428.88            | 22.58    |      20.26     x
1024     | 0.8220   | 11824.80           | 8.46     |      7.59      x
-----------------------------------------------------------------

--- 基准测试完成 ---

nprobe=16时性价比最好，对比HNSW的效果如下：
----------------------------------------------------------------------
efSearch | Recall@5  | Search Time (ms) | QPS      | Speedup vs Flat
---------|----------|--------------------|----------|-----------------
16       | 0.4180   | 59.17              | 1690.05  |     1536.82    x
32       | 0.5400   | 42.89              | 2331.57  |     2120.18    x
64       | 0.6260   | 44.80              | 2232.07  |     2029.70    x
128      | 0.7460   | 42.04              | 2378.68  |     2163.02    x
256      | 0.7580   | 45.03              | 2220.87  |     2019.51    x
512      | 0.7940   | 93.16              | 1073.43  |     976.11     x
1024     | 0.8020   | 181.67             | 550.46   |     500.56     x
2048     | 0.8080   | 353.80             | 282.65   |     257.02     x
4096     | 0.8160   | 924.90             | 108.12   |      98.32     x
8192     | 0.8180   | 3076.53            | 32.50    |      29.56     x
```

## 启动 MCP Server

```bash
# 启动MCP server
python ../retriever_mcp.py \
    --vector_index_path _data/e5_ivfpq.faiss \
    --model_path ../../models/e5-base-v2 \
    --instruction "query: " \
    --corpus_path _data/wiki-18.jsonl \
    --faiss_args "nprobe=16" \
    --device cpu \
    --top_k 3


# Debug
npx @modelcontextprotocol/inspector
```