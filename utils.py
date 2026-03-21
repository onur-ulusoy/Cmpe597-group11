import matplotlib.pyplot as plt
import os

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
