import argparse
import torch
import sys
import os
import gc
import types
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
from dataset import MemeCapTrainDataset
from tqdm import tqdm

# Add root directory to path so we can import utils
sys.path.append(os.getcwd())
import utils 

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def train(args):
    # 1. Setup Directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    log_file = os.path.join(run_dir, "training_log.txt")
    plot_file = os.path.join(run_dir, "loss_plot.png")
    
    print(f"📂 Experiment Output Directory: {run_dir}")

    device = get_device()
    print(f"🚀 Training on {device}...")

    # 2. Load the LARGE Model (ViT-L)
    model_id = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    print(f"🏗️ Loading Model: {model_id}")
    
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)

    # --- 🔧 PATCH: Fix for 'NotImplementedError: get_input_embeddings' ---
    # PEFT needs this method to exist to enable gradient checkpointing.
    # We point it to the text embeddings to satisfy the check.
    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    model.get_input_embeddings = types.MethodType(get_input_embeddings, model)

    # --- 🔧 PATCH: Ensure Gradients Flow for Checkpointing ---
    # Gradient Checkpointing cuts the graph, so we must force inputs to require grad.
    # We do this for BOTH Vision and Text encoders.
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.vision_model.embeddings.patch_embedding.register_forward_hook(make_inputs_require_grad)
    model.text_model.embeddings.token_embedding.register_forward_hook(make_inputs_require_grad)

    # Enable Gradient Checkpointing
    model.gradient_checkpointing_enable()
    model.config.use_cache = False 

    # 3. Apply LoRA
    config = LoraConfig(
        r=16,              
        lora_alpha=32,     
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none"
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()

    # 4. Data Loader
    dataset = MemeCapTrainDataset(
        json_path=args.train_json,
        image_root=args.image_root,
        processor=processor
    )
    
    # num_workers=0 is required for Mac MPS stability
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 5. Training Loop with Gradient Accumulation
    loss_history = []
    accumulation_steps = args.grad_accum_steps
    
    print(f"📉 Batch Size: {args.batch_size} | Accumulation Steps: {accumulation_steps}")
    print(f"🔄 Effective Batch Size: {args.batch_size * accumulation_steps}")

    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for i, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values,
                return_loss=True 
            )
            
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            # Step optimizer only after accumulating enough gradients
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Record raw loss
            current_loss = loss.item() * accumulation_steps
            total_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
            
            # Memory Cleanup for MPS
            del pixel_values, input_ids, attention_mask, outputs, loss
            if i % 50 == 0:
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()
            
        # End of Epoch Stats
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # --- Save Logs & Plot ---
        utils.log_metrics(log_file, epoch, avg_loss)
        utils.plot_loss(loss_history, plot_file)
        
        # --- Save Checkpoint ---
        save_path = os.path.join(run_dir, f"lora_epoch_{epoch}")
        model.save_pretrained(save_path)
        print(f"✅ Saved adapter to {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune")
    
    # Keep batch size small for ViT-L on Mac
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5) 
    args = parser.parse_args()
    
    train(args)
