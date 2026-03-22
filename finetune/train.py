import argparse
import torch
import torch.nn as nn
import sys
import os
from datetime import datetime
from torch.utils.data import DataLoader
import open_clip 
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Add root directory to path
sys.path.append(os.getcwd())
import utils 
from finetune.dataset import MemeCapTrainDataset

# --- Adapter: REQUIRED for Training ---
class OpenClipAdapter:
    def __init__(self, preprocess_fn, tokenizer):
        self.preprocess = preprocess_fn
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
        data = {}
        
        # 1. Handle Images
        if images is not None:
            if isinstance(images, list):
                pixel_values = torch.stack([self.preprocess(img) for img in images])
            else:
                pixel_values = self.preprocess(images).unsqueeze(0)
            data["pixel_values"] = pixel_values
        
        # 2. Handle Text
        if text is not None:
            input_ids = self.tokenizer(text)
            data["input_ids"] = input_ids
            data["attention_mask"] = torch.ones_like(input_ids)
            
        return data

def train(args):
    # 1. Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # --- RESTORED LOGGING FILES ---
    log_file = os.path.join(run_dir, "training_log.txt")
    plot_file = os.path.join(run_dir, "loss_plot.png")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on {device} (OpenCLIP Native)")

    # 2. Load Model
    model_name = "ViT-L-14"
    pretrained = "laion2b_s32b_b82k"
    print(f"🏗️ Loading: {model_name} ({pretrained})")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained=pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    
    # Initialize Adapter
    processor = OpenClipAdapter(preprocess, tokenizer)

    # 3. Apply LoRA
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_fc", "c_proj", "out_proj"], # Target Attention & MLP
        lora_dropout=0.05,
        bias="none"
    )
    
    # Freeze base model to save memory
    for param in model.parameters():
        param.requires_grad = False 
        
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
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True 
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Loss Functions
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    
    # --- RESTORED HISTORY TRACKING ---
    loss_history = []

    print(f"📉 Batch Size: {args.batch_size}")

    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch in progress_bar:
            images = batch["pixel_values"].to(device)
            texts = batch["input_ids"].to(device)
            
            # Forward pass
            image_features, text_features, logit_scale = model(images, texts)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # Calculate Logits
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            # Labels (0, 1, 2... batch_size)
            labels = torch.arange(len(images), device=device, dtype=torch.long)

            # Symmetric Loss
            loss = (loss_img(logits_per_image, labels) + loss_txt(logits_per_text, labels)) / 2
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        
        # --- RESTORED LOGGING CALLS ---
        loss_history.append(avg_loss)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Save to TXT and PNG
        utils.log_metrics(log_file, epoch, avg_loss)
        utils.plot_loss(loss_history, plot_file)
        
        # Save Adapter Checkpoint
        save_path = os.path.join(run_dir, f"lora_epoch_{epoch}")
        model.save_pretrained(save_path)
        print(f"✅ Saved adapter to {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune")
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()
    train(args)
