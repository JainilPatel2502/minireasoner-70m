import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Data.DataLoader import ShardedDataset
from Transformer import Transformer
from count_par import count_parameters
import torch.nn.functional as F

# ---------------- SETUP ----------------
torch.set_float32_matmul_precision("medium")
checkpoint_path = "transformer_checkpoint.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------------- CONFIG ----------------
# Same config as training
data_dir = './Data'  # validation data directory
batch_size = 4
seq_len = 1024
vocab_size = 8192
embd_dim = 512
num_heads = 8
num_layers = 21

# ---------------- LOAD MODEL ----------------
def load_model_from_checkpoint(checkpoint_path):
    model = Transformer(vocab_size, embd_dim, seq_len, num_heads, num_layers).to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return None
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    
    params = count_parameters(model)
    print(f"✅ Model loaded successfully!")
    print(f"Total params: {params['Total']:,}")
    print(f"Checkpoint info: Global Step {checkpoint.get('global_step', 0)}")
    print(f"Tokens trained: {checkpoint.get('total_tokens_trained', 0):,}")
    if 'val_loss' in checkpoint:
        print(f"Last validation loss: {checkpoint['val_loss']:.4f}")
        print(f"Last validation perplexity: {checkpoint['val_perplexity']:.2f}")
    
    return model

# ---------------- VALIDATION LOSS ----------------
def calculate_validation_loss(model, val_loader, max_batches=100):
    """Calculate validation loss on a subset of validation data"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_batches = 0
    total_tokens = 0
    
    print(f"\n🔍 Calculating validation loss (max {max_batches} batches)...")
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
                
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            
            total_loss += loss.item()
            total_batches += 1
            total_tokens += x.numel()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{max_batches} | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    print(f"📊 Validation Results:")
    print(f"   Average Loss: {avg_loss:.4f}")
    print(f"   Perplexity: {perplexity:.2f}")
    print(f"   Evaluated on {total_batches} batches ({total_tokens:,} tokens)")
    
    return avg_loss, perplexity.item()

# ---------------- TEXT SAMPLING ----------------
def sample_text(model, prompt_tokens, max_new_tokens=200, temperature=1.0, top_k=50, top_p=0.9):
    """Generate text from the model"""
    model.eval()
    
    # Start with prompt tokens
    generated = prompt_tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get the last seq_len tokens to stay within context window
            input_tokens = generated[:, -seq_len:] if generated.size(1) > seq_len else generated
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_tokens)
            
            # Get logits for the last position
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    return generated

def decode_tokens(tokens, vocab_size):
    """Simple token decoder - you'll need to replace this with your actual tokenizer"""
    # This is a placeholder - replace with your actual detokenization logic
    # For now, just return the token IDs as string
    return " ".join([str(token.item()) for token in tokens.flatten()])

def generate_samples(model, num_samples=3, prompt_length=10):
    """Generate multiple text samples"""
    print(f"\n🎯 Generating {num_samples} text samples...")
    
    for i in range(num_samples):
        print(f"\n--- Sample {i+1} ---")
        
        # Create random prompt (replace this with actual text tokens if you have a tokenizer)
        prompt = torch.randint(0, vocab_size, (1, prompt_length), device=device)
        
        # Generate text with different temperatures
        temperatures = [0.7, 1.0, 1.2]
        temp = temperatures[i % len(temperatures)]
        
        generated = sample_text(
            model, 
            prompt, 
            max_new_tokens=100, 
            temperature=temp,
            top_k=50,
            top_p=0.9
        )
        
        # Decode tokens (you'll need to implement proper decoding)
        text = decode_tokens(generated[:, prompt_length:], vocab_size)  # Only new tokens
        
        print(f"Temperature: {temp}")
        print(f"Generated tokens: {text[:200]}...")  # First 200 chars

# ---------------- MAIN EXECUTION ----------------
if __name__ == "__main__":
    # Load model
    model = load_model_from_checkpoint(checkpoint_path)
    if model is None:
        exit(1)
    
    # Load validation dataset (you might want to use different files for validation)
    print(f"\n📂 Loading validation dataset...")
    val_dataset = ShardedDataset(data_dir, seq_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate validation loss
    val_loss, perplexity = calculate_validation_loss(model, val_loader, max_batches=50)
    
    # Generate text samples
    generate_samples(model, num_samples=5, prompt_length=20)
    
    print(f"\n✅ Evaluation complete!")
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"Final perplexity: {perplexity:.2f}")
