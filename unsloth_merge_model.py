import sys
from unsloth import FastLanguageModel

MAX_SEQ_LENGTH = 8192
LOAD_IN_4BIT = True

def merge_and_save_model(adapter_model, save_path):
    # 配置和 art_rollout.py 的init_args 保持一致
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_model, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = LOAD_IN_4BIT,
    )
    model.save_pretrained_merged(save_path, tokenizer, save_method = "merged_16bit")

if __name__ == "__main__":
    adapter_model_path = sys.argv[1]
    save_path = sys.argv[2]

    print(f"adapter_model_path: {adapter_model_path}")
    merge_and_save_model(adapter_model_path, save_path)
    print(f"Model merged and saved to {save_path}")