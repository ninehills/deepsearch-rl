import json
corpus_input_file="_data/MultiHop-RAG/dataset/corpus.json"
corpus_output_file="_data/corpus.jsonl"

with open(corpus_output_file, "w") as fo:
    with open(corpus_input_file, "r") as fi:
        corpus = json.load(fi)
        for i, item in enumerate(corpus):
            data = {
                "id": str(i),
                "contents": f'{item["title"]}\n{item["body"]}',
            }
            del item["body"]
            del item["title"]
            data['metadata']=item
            fo.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"corpus size: {i+1}, save to {corpus_output_file}")
