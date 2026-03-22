import argparse
import torch
import sys
import os
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
from dataset import MemeCapTrainDataset
from tqdm import tqdm
import types

# Add root directory to path
sys.path.append(os.getcwd())
import utils 

def train(args):
    # 1. Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    log_file = os.path.join(run_dir, "training_log.txt")
    plot_file = os.path.join(run_dir, "loss_plot.png")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on {device} (Cloud Mode)")

    # 2. Load Large Model
    model_id = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    print(f"🏗️ Loading Model: {model_id}")
    
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)

    # --- PATCH: Fix for PEFT compatibility ---
    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding
    model.get_input_embeddings = types.MethodType(get_input_embeddings, model)

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.vision_model.embeddings.patch_embedding.register_forward_hook(make_inputs_require_grad)
    model.text_model.embeddings.token_embedding.register_forward_hook(make_inputs_require_grad)

    # Enable Gradient Checkpointing (Optional on Cloud if you have >24GB VRAM, but safe to keep)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False 

    # 3. Aggressive LoRA Configuration
    # We target ALL linear layers now, not just Attention.
    config = LoraConfig(
        r=64,               # Increased from 16 to 64 (More capacity)
        lora_alpha=128,     # Alpha usually 2x Rank
        target_modules=[
            "q_proj", "v_proj", "k_proj", "out_proj", # Attention
            "fc1", "fc2"                              # MLP Layers (Crucial for knowledge)
        ], 
        lora_dropout=0.05,
        bias="none"
    )
    
    model = get_peft_model(model, config)
    print(f"🧠 Trainable Parameters:")
    model.print_trainable_parameters()
    model.to(device)
    model.train()

    # 4. Data Loader (Optimized for Speed)
    dataset = MemeCapTrainDataset(
        json_path=args.train_json,
        image_root=args.image_root,
        processor=processor
    )
    
    # num_workers=4 is standard for Cloud GPUs
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    # Increased LR slightly for LoRA
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    loss_history = []
    
    print(f"📉 Batch Size: {args.batch_size}")

    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values,
                return_loss=True 
            )
            
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            current_loss = loss.item()
            total_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        utils.log_metrics(log_file, epoch, avg_loss)
        utils.plot_loss(loss_history, plot_file)
        
        # Save every epoch
        save_path = os.path.join(run_dir, f"lora_epoch_{epoch}")
        model.save_pretrained(save_path)
        print(f"✅ Saved adapter to {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune")
    
    # CLOUD SETTINGS
    parser.add_argument("--batch_size", type=int, default=64) # Try 64 or 128!
    parser.add_argument("--epochs", type=int, default=10)     # Train longer
    parser.add_argument("--lr", type=float, default=5e-5)     # Higher LR for LoRA
    
    args = parser.parse_args()
    train(args)
