# MultiHop-RAG

https://github.com/yixuantt/MultiHop-RAG.git

```bash
mkdir -p _data/
git clone https://github.com/yixuantt/MultiHop-RAG.git
mv MultiHop-RAG _data/

# 转换为 FlashRAG 格式
python convert_dataset.py
python ../corpus_to_chunk.py --input_path _data/corpus.jsonl --output_path _data/chunks.jsonl --chunk_by recursive --chunk_size 500 --tokenizer_name_or_path ../wiki_retriever_mcp/_data/e5-base-v2/

# 构建 embedding index
python -m flashrag.retriever.index_builder \
  --retrieval_method e5 \
  --model_path ../wiki_retriever_mcp/_data/e5-base-v2/ \
  --corpus_path _data/chunks.jsonl \
  --save_dir _data/ \
  --use_fp16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method mean \
  --instruction "passage: " \
  --faiss_type Flat

# 构建 bm25 index
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path _data/chunks.jsonl \
  --bm25_backend bm25s \
  --save_dir _data/

# 启动MCP server
python ../retriever_mcp.py \
    --vector_index_path _data/e5_Flat.index \
    --bm25_index_path _data/bm25/ \
    --model_path ../wiki_retriever_mcp/_data/e5-base-v2 \
    --instruction "query: " \
    --corpus_path _data/chunks.jsonl \
    --use_multi_retriever \
    --merge_method rrf \
    --device cpu \
    --top_k 5


# Debug
npx @modelcontextprotocol/inspector
```
