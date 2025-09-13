import os
import faiss
from sentence_transformers import SentenceTransformer
import logging
from fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)

DATA_DIR = os.environ.get("DATA_DIR", "_data")
# index = faiss.read_index("/mnt/input/agent_lightning/nq_hnsw_faiss_n32e40.index")
logging.info("Loading index...")
index = faiss.read_index("_data/e5_Flat.index")
vector = index.reconstruct(0)  # Returns a NumPy array
logging.info(f"Index loaded successfully. index={index}, dtype={vector.dtype}, dim={index.d}")

logging.info("Loading model...")
model = SentenceTransformer(os.path.join(DATA_DIR, "e5-base-v2"))
logging.info("Model loaded successfully.")

logging.info("Loading chunks...")
import datasets
corpus_path = os.path.join(DATA_DIR, "wiki-18.jsonl")
chunks = datasets.load_dataset(
    'json', 
    data_files=corpus_path,
    split="train",
    num_proc=4
)
logging.info("Chunks loaded successfully.")

mcp = FastMCP(name="wiki retrieval mcp")


@mcp.tool(
    name="retrieve",
    description="retrieve relevant chunks from the wikipedia",
)
def retrieve(query: str) -> list:
    """
    Retrieve relevant chunks from the Wikipedia dataset.

    Args:
        query (str): The query string to search for.

    Returns:
        list: A list of dictionaries containing the retrieved chunks and their metadata.
    """
    top_k = 4  # Number of top results to return
    embedding = model.encode_query([query], normalize_embeddings=True)

    D, I = index.search(embedding, top_k)
    logging.info(f"Query '{query}' search results: {I} with distances {D}")

    results = []
    for i in range(top_k):
        if I[0][i] != -1:
            chunk = chunks[int(I[0][i])]
            results.append({"chunk": chunk, "chunk_id": int(I[0][i]), "distance": float(D[0][i])})
    return results


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8099)
    logging.info("MCP Server started successfully. ")
    config = """{
    "mcpServers": {
        "wiki_retriever": {
            "type": "streamable-http",
            "url": "http://<ip>:8099/mcp"
        }
    }
}
    """
    print(config)