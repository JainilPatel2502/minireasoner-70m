

import regex as re
from datasets import load_dataset


GPT_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
pattern = re.compile(GPT_SPLIT_PATTERN)

def gpt4_style_pre_tokenize(text: str) -> str:
    tokens = pattern.findall(text)
    return " ".join(tokens)


dataset = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True)

output_file = "corpus.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for i, example in enumerate(dataset):
        if i >= 50000:
            break
        print(i)
        inp = str(example.get("input", "")).strip()
        out = str(example.get("output", "")).strip()
        sol = str(example.get("solution", "")).strip()

        if not inp or not out:
            continue
        line = gpt4_style_pre_tokenize(f"<bos> {inp} {out} {sol} <eos>\n")
        f.write(line)
