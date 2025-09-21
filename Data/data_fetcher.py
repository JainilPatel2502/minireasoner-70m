import numpy as np
from datasets import load_dataset
import sentencepiece as spm

# ---------------- CONFIG ----------------
split_name = "split_0"              # dataset split
max_tokens_total = 3_000_000_000    # stop after this many tokens
tokens_per_shard = 50_000_000       # each shard file size
tokenizer_path = "tokenizer.model"  # your SentencePiece tokenizer

# ----------------------------------------
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_path)

# Streaming mode: no full load into RAM
dataset = load_dataset("nvidia/OpenCodeReasoning", split_name,split=split_name, streaming=True)

def merge(example):
    """Merge input/output/solution into one text."""
    parts = []
    if example.get("input"): parts.append("<bos> " + example["input"])
    if example.get("output"): parts.append(example["output"])
    if example.get("solution"): parts.append(example["solution"] + " <eos>")
    return "\n\n".join(parts)

# ----------------------------------------
token_count = 0
shard_count = 0
current_buffer = []

def write_shard(tokens, idx):
    arr = np.memmap(f"train_{idx:04d}.bin", dtype=np.uint32, mode="w+", shape=(len(tokens),))
    arr[:] = tokens
    arr.flush()
    print(f"✅ wrote train_{idx:04d}.bin with {len(tokens):,} tokens")

# ----------------------------------------
for example in dataset:
    text = merge(example)
    ids = sp.encode(text, out_type=int)
    ids.append(sp.eos_id())

    current_buffer.extend(ids)
    token_count += len(ids)

    if len(current_buffer) >= tokens_per_shard:
        write_shard(current_buffer[:tokens_per_shard], shard_count)
        current_buffer = current_buffer[tokens_per_shard:]
        shard_count += 1

    if token_count >= max_tokens_total:
        break

# Write leftovers
if current_buffer:
    write_shard(current_buffer, shard_count)

print(f"\n🎉 Done! Total tokens written: {min(token_count, max_tokens_total):,} "
      f"across {shard_count+1} shards.")
