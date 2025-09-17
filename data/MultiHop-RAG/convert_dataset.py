import json
import random

random.seed(42)
corpus_input_file="_data/MultiHop-RAG/dataset/corpus.json"
corpus_output_file="_data/corpus.jsonl"
qa_input_file="_data/MultiHop-RAG/dataset/MultiHopRAG.json"
train_file="_data/train.jsonl"
val_file="_data/val.jsonl"

VAL_SIZE = 200

# 格式参考 FlashRAG https://github.com/RUC-NLPIR/FlashRAG/tree/main?tab=readme-ov-file#datasets

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


data = []
with open(qa_input_file, "r") as fi:
    items = json.load(fi)
    for i, item in enumerate(items):
        data.append({
            "id": str(i),
            "question": item["query"],
            "golden_answers": [item["answer"]],
            "metadata": {
                "question_type": item["question_type"],
                "evidence_list": item["evidence_list"],
            }
        })

random.shuffle(data)
val_data = data[:VAL_SIZE]
train_data = data[VAL_SIZE:]

with open(train_file, "w") as fo:
    for item in train_data:
        fo.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(val_file, "w") as fo:
    for item in val_data:
        fo.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"train size: {len(train_data)}, val size: {len(val_data)}, save to {train_file} and {val_file}")