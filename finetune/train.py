import argparse
import torch
import torch.nn as nn
import sys
import os
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import open_clip 
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

sys.path.append(os.getcwd())
from utils import set_seed, log_metrics, plot_loss
from dataset import load_memecap_records

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

# --- Finetune Specific Dataset ---
class MemeCapFinetuneDataset(Dataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor
        
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sample = self.records[idx]
        
        try:
            image = Image.open(sample.image_path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        processed_image = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        processed_caption = self.processor(text=sample.caption, return_tensors="pt")["input_ids"].squeeze(0)
        processed_title = self.processor(text=sample.title, return_tensors="pt")["input_ids"].squeeze(0)

        return {
            "pixel_values": processed_image,
            "input_ids": processed_caption,
            "title_ids": processed_title 
        }

def train(args):
    set_seed(42)
    # --- 1. Automated Directory Setup ---
    base_save_dir = os.path.join("outputs", f"finetune_{args.task}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    log_file = os.path.join(run_dir, "training_log.txt")
    plot_file = os.path.join(run_dir, "loss_plot.png")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Task: {args.task.upper()}")
    print(f"Saving to: {run_dir}")

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
    # Call the universal data loader from the root
    records = load_memecap_records(args.train_json, args.image_root)
    dataset = MemeCapFinetuneDataset(records, processor)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True 
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
            
            # Access base CLIP model inside LoRA wrapper
            img_feats = model.model.encode_image(images)
            cap_feats = model.model.encode_text(captions)
            title_feats = model.model.encode_text(titles)
            
            if args.task == "type1":
                query_feats = img_feats
            elif args.task == "type2":
                img_norm = img_feats / img_feats.norm(dim=1, keepdim=True)
                title_norm = title_feats / title_feats.norm(dim=1, keepdim=True)
                query_feats = (img_norm + title_norm) / 2.0
            
            query_feats = query_feats / query_feats.norm(dim=1, keepdim=True)
            cap_feats = cap_feats / cap_feats.norm(dim=1, keepdim=True)
            
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
        
        # Using universal loggers
        log_metrics(log_file, epoch, avg_loss)
        plot_loss(loss_history, plot_file)
        
        save_path = os.path.join(run_dir, f"lora_epoch_{epoch}")
        model.save_pretrained(save_path)
        print(f"Saved adapter to {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["type1", "type2"])
    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    args = parser.parse_args()
    train(args)