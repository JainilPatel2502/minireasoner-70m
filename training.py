import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from Data.DataLoader import ShardedDataset   # your dataset class
from Transformer import Transformer
from count_par import count_parameters
import torch.nn.functional as F

# ---------------- SETUP ----------------
torch.set_float32_matmul_precision("medium")
checkpoint_path = "transformer_checkpoint.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Initialize TensorBoard
tb_log_dir = "runs/transformer_training"
writer = SummaryWriter(tb_log_dir)
print(f"TensorBoard logging to: {tb_log_dir}")
print(f"View logs: tensorboard --logdir={tb_log_dir}")

# ---------------- CONFIG ----------------
data_dir = './Data'   # where your train_0000.bin ... train_0059.bin are stored
batch_size = 4
seq_len = 1024
vocab_size = 8192
embd_dim = 512
num_heads = 8
num_layers = 21
lr = 1e-4  # Increased base learning rate
epochs = 2
warmup_steps = 1000  # Longer warmup for stability

# Calculate max_steps based on dataset size
# 3B tokens ÷ (batch_size=4 × seq_len=1024) = ~732k steps per epoch
tokens_per_batch = batch_size * seq_len
total_tokens = 3_000_000_000
steps_per_epoch = total_tokens // tokens_per_batch
max_steps = steps_per_epoch * epochs  # Total steps for all epochs

print(f"Training Configuration:")
print(f"Tokens per batch: {tokens_per_batch:,} | Steps per epoch: {steps_per_epoch:,} | Total steps: {max_steps:,}")
# ----------------------------------------

# ---------------- MODEL -----------------
model = Transformer(vocab_size, embd_dim, seq_len, num_heads, num_layers).to(device)
params = count_parameters(model)
print(f"Total params: {params['Total']:,}")
print(f"Trainable params: {params['Trainable']:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler("cuda")

# Learning rate scheduler with warmup and cosine decay
def get_lr(step):
    if step < warmup_steps:
        # Linear warmup
        return lr * (step + 1) / warmup_steps
    elif step < max_steps:
        # Cosine decay to 10% of peak LR
        import math
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
    else:
        # Minimum learning rate
        return lr * 0.1

# ---------------- RESUME ----------------
global_step = 0  # Initialize global step counter
start_sample_idx = 0  # Start from beginning of dataset
total_tokens_trained = 0

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    # Resume from checkpoint
    global_step = checkpoint.get("global_step", 0)
    start_sample_idx = checkpoint.get("sample_idx", 0)
    total_tokens_trained = checkpoint.get("total_tokens_trained", 0)
    
    print(f"Resumed from step {global_step:,} | Progress: {(global_step / max_steps * 100):.2f}%")
else:
    print("Starting fresh training")

# ---------------- DATASET ----------------
dataset = ShardedDataset(data_dir, seq_len)
if start_sample_idx > 0:
    dataset = Subset(dataset, range(start_sample_idx, len(dataset)))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

print(f"Dataset loaded: {len(dataset):,} samples")

# ---------------- VALIDATION FUNCTION ----------------
def calculate_validation_loss(model, data_dir, max_batches=20, val_batch_size=2):
    """Calculate validation loss using a small portion of the dataset"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Create separate validation dataset (use same data but smaller batches for speed)
    val_dataset = ShardedDataset(data_dir, seq_len)
    # Use only first 1000 samples for quick validation
    val_subset = torch.utils.data.Subset(val_dataset, range(min(1000, len(val_dataset))))
    val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False)
    
    total_loss = 0.0
    total_batches = 0
    
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
    
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    model.train()  # Switch back to training mode
    return avg_loss, perplexity.item()

# ---------------- TRAIN LOOP -------------
# Calculate which epoch we're in based on global_step
current_epoch = global_step // steps_per_epoch
steps_in_current_epoch = global_step % steps_per_epoch

print(f"\nStarting training from epoch {current_epoch + 1}/{epochs}")

for epoch in range(current_epoch, epochs):
    model.train()

    for step, (x, y) in enumerate(loader):
        t0 = time.time()

        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # More aggressive clipping
        
        # Learning rate scheduling
        current_lr = get_lr(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        scaler.step(optimizer)
        scaler.update()

        # ---- metrics ----
        torch.cuda.synchronize() if device == "cuda" else None
        elapsed_s = time.time() - t0
        tokens = batch_size * seq_len
        throughput = tokens / elapsed_s
        elapsed_ms = elapsed_s * 1000
        total_tokens_trained += tokens

        global_step += 1
        
        # Calculate training perplexity
        train_perplexity = torch.exp(loss).item()
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', loss.item(), global_step)
        writer.add_scalar('Perplexity/Train', train_perplexity, global_step)
        writer.add_scalar('Learning_Rate', current_lr, global_step)
        writer.add_scalar('Gradient_Norm', grad_norm, global_step)
        writer.add_scalar('Throughput_tokens_per_sec', throughput, global_step)
        writer.add_scalar('Total_Tokens_Trained', total_tokens_trained, global_step)
        
        # Display training metrics every 10 steps
        if global_step % 10 == 0:
            progress_pct = (global_step / max_steps) * 100
            print(
                f"Step {global_step} | "
                f"Loss: {loss.item():.4f} | "
                f"PPL: {train_perplexity:.2f} | "
                f"LR: {current_lr:.2e} | "
                f"Progress: {progress_pct:.2f}% | "
                f"Throughput: {throughput:.0f} tok/s"
            )

        # ---- validation & checkpoint every 100 steps ----
        if global_step % 500 == 0:
            val_loss, val_perplexity = calculate_validation_loss(model, data_dir)
            print(f"Validation | Loss: {val_loss:.4f} | PPL: {val_perplexity:.2f}")
            
            # Log validation metrics to TensorBoard
            writer.add_scalar('Loss/Validation', val_loss, global_step)
            writer.add_scalar('Perplexity/Validation', val_perplexity, global_step)
            writer.add_scalars('Loss/Train_vs_Val', {
                'Train': loss.item(),
                'Validation': val_loss
            }, global_step)
            writer.add_scalars('Perplexity/Train_vs_Val', {
                'Train': train_perplexity,
                'Validation': val_perplexity
            }, global_step)
            
            # Save checkpoint
            current_sample_idx = start_sample_idx + (step + 1) * batch_size
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "sample_idx": current_sample_idx,
                "total_tokens_trained": total_tokens_trained,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity
            }, checkpoint_path)
            print(f"Checkpoint saved | Tokens: {total_tokens_trained:,}")
            writer.flush()
        
        # ---- log model parameters every 500 steps ----
        if global_step % 500 == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Parameters/{name}', param, global_step)
                    writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
            writer.flush()
        
        # Check if we've reached max steps
        if global_step >= max_steps:
            print(f"Training complete. Reached {max_steps:,} steps.")
            break
    
    # Break outer loop if max steps reached
    if global_step >= max_steps:
        break

print(f"\nTraining completed!")
print(f"Total steps: {global_step:,} | Tokens trained: {total_tokens_trained:,} | Epochs: {global_step / steps_per_epoch:.2f}")

# Close TensorBoard writer
writer.close()
print(f"TensorBoard logs: {tb_log_dir}")
