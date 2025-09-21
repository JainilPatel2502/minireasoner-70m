import os
import torch
import torch.nn.functional as F
from Transformer import Transformer
import sentencepiece as spm

checkpoint_path = "model_weights.pt"
tokenizer_path = "Data/tokenizer.model"
device = "cuda" if torch.cuda.is_available() else "cpu"

vocab_size = 8192
embd_dim = 512
seq_len = 1024
num_heads = 8
num_layers = 21

sp = spm.SentencePieceProcessor()
sp.load(tokenizer_path)

model = Transformer(vocab_size, embd_dim, seq_len, num_heads, num_layers).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

def sample_token(logits, temperature=0.8, top_k=40, top_p=0.9):
    logits = logits / temperature
    
    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.full_like(logits, float('-inf'))
        logits[top_k_indices] = top_k_logits
    
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def generate_text(prompt, max_tokens=1000, temperature=0.8, top_k=40, top_p=0.9):
    if not prompt.startswith("<bos>"):
        prompt = "<bos> " + prompt
    
    prompt_tokens = sp.encode(prompt, out_type=int)
    generated = torch.tensor([prompt_tokens], device=device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            input_tokens = generated[:, -seq_len:] if generated.size(1) > seq_len else generated
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_tokens)
            
            next_token = sample_token(logits[0, -1, :].float(), temperature, top_k, top_p)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == sp.eos_id():
                break
    
    generated_tokens = generated.flatten().tolist()
    new_tokens = generated_tokens[len(prompt_tokens):]
    return sp.decode(new_tokens)

def generate_tokens_stream(prompt, max_tokens=1000, temperature=0.8, top_k=40, top_p=0.9):
    if not prompt.startswith("<bos>"):
        prompt = "<bos> " + prompt
    
    prompt_tokens = sp.encode(prompt, out_type=int)
    generated = torch.tensor([prompt_tokens], device=device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            input_tokens = generated[:, -seq_len:] if generated.size(1) > seq_len else generated
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_tokens)
            
            next_token = sample_token(logits[0, -1, :].float(), temperature, top_k, top_p)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            token_id = next_token.item()
            token_text = sp.decode([token_id])
            yield token_text
            
            if token_id == sp.eos_id():
                break

if __name__ == "__main__":
    prompt = input("Enter prompt: ")
    
    # print("\nGenerating complete text:")
    # result = generate_text(prompt, max_tokens=100)
    # print(result)
    
    print("\nGenerating token stream:")
    for token in generate_tokens_stream(prompt, max_tokens=100):
        print(token, end=' ', flush=True)
    print()