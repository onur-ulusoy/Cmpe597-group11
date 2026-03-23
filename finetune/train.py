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

# --- Adapter Helper ---
class OpenClipAdapter:
    def __init__(self, preprocess_fn, tokenizer):
        self.preprocess = preprocess_fn
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
        data = {}
        if images is not None:
            if isinstance(images, list):
                pixel_values = torch.stack([self.preprocess(img) for img in images])
            else:
                pixel_values = self.preprocess(images).unsqueeze(0)
            data["pixel_values"] = pixel_values
        
        if text is not None:
            input_ids = self.tokenizer(text)
            data["input_ids"] = input_ids
            
        return data

def train(args):
    # --- 1. Automated Directory Setup ---
    # Base folder: outputs/finetune_type1 OR outputs/finetune_type2
    base_save_dir = os.path.join("outputs", f"finetune_{args.task}")
    
    # Run folder: outputs/finetune_type1/20260323_120000
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    log_file = os.path.join(run_dir, "training_log.txt")
    plot_file = os.path.join(run_dir, "loss_plot.png")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training Task: {args.task.upper()}")
    print(f"📂 Saving to: {run_dir}")

    # --- 2. Load Model ---
    model_name = "ViT-L-14"
    pretrained = "laion2b_s32b_b82k"
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    processor = OpenClipAdapter(preprocess, tokenizer)

    # --- 3. Apply LoRA ---
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_fc", "c_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False 
        
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()

    # --- 4. Data Loader ---
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
    
    loss_history = []

    # --- 5. Training Loop ---
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch in progress_bar:
            images = batch["pixel_values"].to(device)
            captions = batch["input_ids"].to(device)
            titles = batch["title_ids"].to(device)
            
            # A. Encode Features
            # We access the base CLIP model inside the LoRA wrapper
            # LoRA wraps the linear layers, so calling model.model.encode_... works
            img_feats = model.model.encode_image(images)
            cap_feats = model.model.encode_text(captions)
            title_feats = model.model.encode_text(titles)
            
            # B. Define Query based on Task
            if args.task == "type1":
                # Type 1: Query = Image
                query_feats = img_feats
            elif args.task == "type2":
                # Type 2: Query = Image + Title
                # Normalize first to give equal weight
                img_norm = img_feats / img_feats.norm(dim=1, keepdim=True)
                title_norm = title_feats / title_feats.norm(dim=1, keepdim=True)
                # Average them
                query_feats = (img_norm + title_norm) / 2.0
            
            # C. Normalize for Loss Calculation
            query_feats = query_feats / query_feats.norm(dim=1, keepdim=True)
            cap_feats = cap_feats / cap_feats.norm(dim=1, keepdim=True)
            
            # D. Compute Similarity & Loss
            logit_scale = model.model.logit_scale.exp()
            
            logits_per_query = logit_scale * query_feats @ cap_feats.t()
            logits_per_cap = logits_per_query.t()
            
            labels = torch.arange(len(images), device=device, dtype=torch.long)
            
            loss = (loss_img(logits_per_query, labels) + loss_txt(logits_per_cap, labels)) / 2
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Log and Save
        utils.log_metrics(log_file, epoch, avg_loss)
        utils.plot_loss(loss_history, plot_file)
        
        save_path = os.path.join(run_dir, f"lora_epoch_{epoch}")
        model.save_pretrained(save_path)
        print(f"✅ Saved adapter to {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required Task Argument
    parser.add_argument("--task", type=str, required=True, choices=["type1", "type2"], 
                        help="type1: Image only. type2: Image + Title fusion.")
    
    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    # output_dir argument REMOVED (Automated)
    
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    args = parser.parse_args()
    train(args)
