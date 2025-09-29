import os
import sys
import logging
import argparse
from fastmcp import FastMCP

# 添加FlashRAG到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FlashRAG'))

from flashrag.retriever.retriever import DenseRetriever, BM25Retriever, MultiRetrieverRouter

def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval MCP Server")
    
    # 数据路径配置
    parser.add_argument("--vector_index_path", type=str, required=False,
                        help="Vector index file path")
    parser.add_argument("--bm25_index_path", type=str, required=False,
                        help="BM25 index directory path")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Model path")
    parser.add_argument("--corpus_path", type=str, required=True,
                        help="Corpus file path")
    
    # 检索配置
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top results to return")
    parser.add_argument("--retrieval_method", type=str, default="e5",
                        help="Retrieval method name")
    parser.add_argument("--use_multi_retriever", action="store_true",
                        help="Use multi-retriever with both vector and BM25")
    parser.add_argument("--merge_method", type=str, default="rrf",
                        choices=["concat", "rrf", "rerank"],
                        help="Method to merge multi-retriever results")
    parser.add_argument("--rerank_model_path", type=str, default=None,
                        help="Reranker model path (required for rerank merge method)")
    
    # 模型配置
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "auto"],
                        help="Device to run model on (cpu/cuda/auto)")
    parser.add_argument("--use_fp16", action="store_true",
                        help="Use FP16 precision")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for encoding")
    parser.add_argument("--pooling_method", type=str, default="mean",
                        choices=["mean", "cls", "max"],
                        help="Pooling method")
    parser.add_argument("--instruction", type=str, default="",
                        help="Instruction prefix for queries")
    
    # 服务器配置
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8099,
                        help="Server port")
    
    # 日志配置
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    return parser.parse_args()

def build_retriever_config(args):
    """构建FlashRAG配置"""
    config = {
        # 基本配置
        "retrieval_method": args.retrieval_method,
        "retrieval_topk": args.top_k,
        "corpus_path": args.corpus_path,
        "device": args.device,
        
        # 缓存配置
        "save_retrieval_cache": False,
        "use_retrieval_cache": False,
        "retrieval_cache_path": None,
        
        # 重排配置
        "use_reranker": False,
        
        # 模型配置
        "retrieval_model_path": args.model_path,
        "retrieval_query_max_length": args.max_length,
        "retrieval_pooling_method": args.pooling_method,
        "retrieval_use_fp16": args.use_fp16,
        "retrieval_batch_size": args.batch_size,
        "instruction": args.instruction,
        "use_sentence_transformer": True,
        "faiss_gpu": args.device == "cuda",
        "silent_retrieval": True,
    }
    
    if args.vector_index_path:
        config["index_path"] = args.vector_index_path
    
    return config

def build_multi_retriever_config(args):
    """构建多检索器配置"""
    retriever_list = []
    
    # 添加向量检索器
    if args.vector_index_path:
        vector_config = build_retriever_config(args)
        vector_config["index_path"] = args.vector_index_path
        retriever_list.append(vector_config)
    
    # 添加BM25检索器
    if args.bm25_index_path:
        bm25_config = {
            "retrieval_method": "bm25",
            "retrieval_topk": args.top_k,
            "corpus_path": args.corpus_path,
            "index_path": args.bm25_index_path,
            "bm25_backend": "bm25s",
            "retrieval_model_path": None,  # BM25不需要模型路径，但MultiRetrieverRouter需要这个字段
            "save_retrieval_cache": False,
            "use_retrieval_cache": False,
            "retrieval_cache_path": None,
            "use_reranker": False,
            "silent_retrieval": True,
        }
        retriever_list.append(bm25_config)
    
    config = {
        "device": args.device,  # reranker需要device信息
        "multi_retriever_setting": {
            "retriever_list": retriever_list,
            "merge_method": args.merge_method,
            "topk": args.top_k,
        }
    }
    
    # 如果使用rerank方法，添加reranker配置
    if args.merge_method == "rerank":
        if not args.rerank_model_path:
            raise ValueError("--rerank_model_path is required when using rerank merge method")
        config["multi_retriever_setting"].update({
            "rerank_method": "cross_encoder",
            "rerank_model_path": args.rerank_model_path,
            "rerank_topk": args.top_k,
            "device": args.device,
        })
    
    return config

# 解析命令行参数
args = parse_args()

# 验证参数
if args.merge_method == "rerank" and not args.rerank_model_path:
    raise ValueError("--rerank_model_path is required when using rerank merge method")

# 配置日志
logging.basicConfig(level=getattr(logging, args.log_level))

logging.info(f"Configuration: vector_index_path={args.vector_index_path}, "
             f"bm25_index_path={args.bm25_index_path}, "
             f"model_path={args.model_path}, corpus_path={args.corpus_path}, "
             f"top_k={args.top_k}, device={args.device}, use_multi_retriever={args.use_multi_retriever}, "
             f"host={args.host}, port={args.port}")

# 初始化检索器
if args.use_multi_retriever and args.vector_index_path and args.bm25_index_path:
    logging.info("Initializing multi-retriever...")
    config = build_multi_retriever_config(args)
    retriever = MultiRetrieverRouter(config)
    logging.info("Multi-retriever initialized successfully.")
elif args.vector_index_path:
    logging.info("Initializing dense retriever...")
    config = build_retriever_config(args)
    retriever = DenseRetriever(config)
    logging.info("Dense retriever initialized successfully.")
elif args.bm25_index_path:
    logging.info("Initializing BM25 retriever...")
    config = build_retriever_config(args)
    config.update({
        "retrieval_method": "bm25",
        "index_path": args.bm25_index_path,
        "bm25_backend": "bm25s",
    })
    retriever = BM25Retriever(config)
    logging.info("BM25 retriever initialized successfully.")
else:
    raise ValueError("Must specify either --vector_index_path or --bm25_index_path")

def format_retrieval_results(docs, scores, with_meta=False):
    """格式化检索结果，处理score信息"""
    formatted_results = []
    for doc, score in zip(docs, scores):
        if isinstance(doc, str):
            doc = {"contents": doc, "id": ""}
        metadata = doc.get("metadata", {})
        result = {
            "chunk": doc.get("contents", ""),
            "chunk_id": doc.get("id", ""),
            "score": float(score)
        }
        # 如果需要返回metadata且doc是字典类型
        if with_meta:
            result['metadata'] = metadata
        formatted_results.append(result)

    return formatted_results

mcp = FastMCP(name="retrieval_server")

@mcp.tool(
    name="retrieve",
    description="retrieve relevant chunks from the corpus, with_meta MUST False",
    enabled=True,
)
def retrieve(query: str, with_meta: bool = False) -> list:
    """
    Retrieve relevant chunks from the corpus.

    Args:
        query (str): The query string to search for.
        with_meta (bool): Whether to include metadata in the results. Defaults to False.

    Returns:
        list: A list of dictionaries containing the retrieved chunks and their metadata.
    """
    try:
        results = retriever.search(query, num=args.top_k, return_score=True)
        
        if isinstance(results, tuple):
            docs, scores = results
        else:
            docs = results
            scores = [0.0] * len(docs)
        
        # 格式化结果
        formatted_results = format_retrieval_results(docs, scores, with_meta)

        logging.info(f"Query '{query}' retrieved {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")
        return []

@mcp.tool(
    name="batch_retrieve",
    description="retrieve relevant chunks for multiple queries, with_meta MUST False",
    enabled=False
)
def batch_retrieve(queries: list, with_meta: bool = False) -> list:
    """
    Retrieve relevant chunks for multiple queries.

    Args:
        queries (list): List of query strings.
        with_meta (bool): Whether to include metadata in the results. Defaults to False.

    Returns:
        list: A list of lists, each containing retrieved chunks for the corresponding query.
    """
    try:
        results = retriever.batch_search(queries, num=args.top_k, return_score=True)
        
        if isinstance(results, tuple):
            all_docs, all_scores = results
        else:
            all_docs = results
            all_scores = [[0.0] * len(docs) for docs in all_docs]
        
        # 格式化结果
        formatted_results = []
        for query_docs, query_scores in zip(all_docs, all_scores):
            query_results = format_retrieval_results(query_docs, query_scores, with_meta)
            formatted_results.append(query_results)
        
        logging.info(f"Batch retrieval for {len(queries)} queries completed")
        return formatted_results
        
    except Exception as e:
        logging.error(f"Error during batch retrieval: {e}")
        return []

if __name__ == "__main__":
    logging.info(f"Starting MCP server on {args.host}:{args.port}")
    mcp.run(transport="sse", host=args.host, port=args.port)
    logging.info("MCP Server started successfully.")
    
    config = f"""{{
    "mcpServers": {{
        "retrieval_server": {{
            "type": "sse",
            "url": "http://<ip>:{args.port}/sse"
        }}
    }}
}}"""
    print("Configuration for Claude Code:")
    print(config)