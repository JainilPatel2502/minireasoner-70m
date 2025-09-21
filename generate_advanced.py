import os
import torch
import torch.nn.functional as F
from Transformer import Transformer
import sentencepiece as spm
import math

# Setup
checkpoint_path = "transformer_checkpoint.pt"
tokenizer_path = "Data/tokenizer.model"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Config
vocab_size = 8192
embd_dim = 512
seq_len = 1024
num_heads = 8
num_layers = 21

# Advanced generation parameters for better quality
temperature = 0.5 # Lower for more focused responses
top_k = 40         # Limit to top 40 tokens
top_p = 0.9        # Nucleus sampling
repetition_penalty = 1.1  # Reduce repetition

# Load tokenizer
print("Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.load(tokenizer_path)
print("Tokenizer loaded!")

# Load model
print("Loading model...")
model = Transformer(vocab_size, embd_dim, seq_len, num_heads, num_layers).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
print("Model loaded!")

# Get prompt from user with better handling
print("\n" + "="*60)
print("ADVANCED TEXT GENERATOR")
print("="*60)
print("Enter your prompt (or press Enter for default coding prompt):")
prompt_text = input("> ").strip()

# Use default coding prompt if empty
if not prompt_text:
    prompt_text = "def fibonacci(n):\n    \"\"\"Calculate the nth Fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    else:"
    print(f"Using default prompt: '{prompt_text}'")
else:
    print(f"Using your prompt: '{prompt_text}'")

# Add BOS token if not present
if not prompt_text.startswith("<bos>"):
    prompt_text = "<bos> " + prompt_text

# Encode prompt
prompt_tokens = sp.encode(prompt_text, out_type=int)
generated = torch.tensor([prompt_tokens], device=device)
print(f"\nPrompt encoded to {len(prompt_tokens)} tokens")
print(f"Starting generation from {len(prompt_tokens)} tokens...")

# Advanced sampling function
def apply_repetition_penalty(logits, generated_tokens, penalty=1.1):
    """Apply repetition penalty to reduce repetitive text"""
    for token in set(generated_tokens[-50:]):  # Look at last 50 tokens
        if token < logits.size(0):
            if logits[token] > 0:
                logits[token] /= penalty
            else:
                logits[token] *= penalty
    return logits

def sample_with_nucleus(logits, temperature=0.8, top_k=40, top_p=0.9):
    """Advanced nucleus (top-p) sampling with temperature and top-k"""
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-k filtering
    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.full_like(logits, float('-inf'))
        logits[top_k_indices] = top_k_logits
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
    
    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

# Generate 1000 tokens with advanced sampling
print("Generating 1000 tokens with advanced sampling...")
print(f"Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}, Repetition penalty: {repetition_penalty}")

with torch.no_grad():
    for i in range(5000):
        # Get input (last seq_len tokens)
        input_tokens = generated[:, -seq_len:] if generated.size(1) > seq_len else generated
        
        # Forward pass with mixed precision
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_tokens)
        
        # Get logits for next token
        next_token_logits = logits[0, -1, :].float()
        
        # Apply repetition penalty
        generated_tokens = generated.flatten().tolist()
        next_token_logits = apply_repetition_penalty(
            next_token_logits, generated_tokens, repetition_penalty
        )
        
        # Advanced sampling
        next_token = sample_with_nucleus(
            next_token_logits, temperature, top_k, top_p
        )
        
        # Append token
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        # Early stopping on EOS token
        if next_token.item() == sp.eos_id():
            print(f"Stopped early at EOS token after {i + 1} tokens")
            break
        
        # Progress update
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/1000 tokens")

# Decode to text
print("\nDecoding tokens to text...")
generated_tokens = generated.flatten().tolist()
prompt_token_count = len(prompt_tokens)
new_tokens = generated_tokens[prompt_token_count:]

# Decode full text and new text separately
full_decoded_text = sp.decode(generated_tokens)
new_decoded_text = sp.decode(new_tokens)

# Save to file
output_file = "generated_output.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("ADVANCED TEXT GENERATION RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Original Prompt:\n{prompt_text}\n\n")
    f.write("="*60 + "\n\n")
    f.write("NEW GENERATED TEXT (excluding prompt):\n")
    f.write(new_decoded_text + "\n\n")
    f.write("="*60 + "\n\n")
    f.write("GENERATION STATISTICS:\n")
    f.write(f"Prompt tokens: {prompt_token_count}\n")
    f.write(f"New tokens generated: {len(new_tokens)}\n")
    f.write(f"Total tokens: {len(generated_tokens)}\n")

