import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")


def test_sentence(sentence: str):
    print("\n=== Test Input ===")
    print(repr(sentence))


    ids = sp.encode(sentence, out_type=int)
    tokens = sp.encode(sentence, out_type=str)
    decoded = sp.decode(ids)
    print("\nDecoded:", repr(decoded))

test_sentence("print('Hello')")
test_sentence("    print('Indented')")
test_sentence("def foo():\n    return 42")
test_sentence("world")
test_sentence(r"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from Transformer import Transformer
from count_par import count_parameters
torch.set_float32_matmul_precision("medium")
checkpoint_path = "transformer_checkpoint.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


batch_size = 4
seq_len = 1024
vocab_size = 8192
embd_dim = 512
num_heads = 8
num_layers = 21
lr = 3e-4
epochs = 2
steps_per_epoch = 200


model = Transformer(vocab_size, embd_dim, seq_len, num_heads, num_layers).to(device)
params = count_parameters(model)
print(f"Total params: {params['Total']:,}")
print(f"Trainable params: {params['Trainable']:,}")
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)




start_epoch, start_step = 0, 0
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path} ...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    start_step = checkpoint["step"] + 1
    print(f"Resumed from epoch {start_epoch}, step {start_step}")




def get_fake_batch(batch_size, seq_len, vocab_size, device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y




scaler = torch.amp.GradScaler("cuda")
global_step = start_step




for epoch in range(start_epoch, epochs):
    model.train()
    print(f"\nEpoch {epoch+1}/{epochs}")

    for step in range(start_step, steps_per_epoch):
        t0 = time.time()

        x, y = get_fake_batch(batch_size, seq_len, vocab_size, device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize() if device == "cuda" else None
        t1 = time.time()
        elapsed_ms = (t1 - t0) * 1000
        elapsed_s = (t1 - t0)
        tokens = batch_size * seq_len
        throughput = tokens / elapsed_s

        global_step += 1
        print(f"  Step {global_step} | "
              f"Loss: {loss.item():.4f} | "
              f"Time: {elapsed_ms:.2f} ms | "
              f"Throughput: {throughput:.2f} tokens/s")

        if global_step % 50 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step
            }, checkpoint_path)
            print(f"Checkpoint saved at step {global_step}")

    start_step = 0
""")
