import matplotlib.pyplot as plt
import os
import glob

def log_metrics(log_path, epoch, avg_loss):
    """Appends the epoch loss to a text file."""
    mode = 'a' if os.path.exists(log_path) else 'w'
    with open(log_path, mode) as f:
        f.write(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}\n")
    print(f"📝 Logged metrics to {log_path}")

def plot_loss(loss_history, save_path):
    """Plots the loss history and saves the image."""
    epochs = range(1, len(loss_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history, marker='o', linestyle='-', color='#2c3e50', linewidth=2)
    
    plt.title('Training Loss per Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Contrastive Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs) # Force integer ticks for epochs
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # Close memory to prevent leaks
    print(f"📊 Updated loss plot at {save_path}")

def get_latest_checkpoint(base_output_dir="outputs/finetune"):
    """Finds the latest timestamp folder, then the latest lora_epoch folder inside it."""
    
    # 1. Find all timestamp folders
    runs = sorted(glob.glob(os.path.join(base_output_dir, "*")))
    runs = [r for r in runs if os.path.isdir(r)]
    
    if not runs:
        raise ValueError(f"No training runs found in {base_output_dir}")
    
    latest_run = runs[-1] # Last one is latest due to YYYYMMDD format
    
    # 2. Find all epoch folders inside the latest run
    epochs = glob.glob(os.path.join(latest_run, "lora_epoch_*"))
    
    if not epochs:
        raise ValueError(f"No checkpoint folders found in {latest_run}")
    
    # Sort by epoch number (handle 'lora_epoch_10' vs 'lora_epoch_2')
    epochs.sort(key=lambda x: int(x.split('_')[-1]))
    
    latest_checkpoint = epochs[-1]
    print(f"🔎 Auto-detected latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint