import argparse
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model
from dataset import MemeCapTrainDataset
from tqdm import tqdm
import os

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps" # Use Mac GPU
    else:
        return "cpu"

def train(args):
    device = get_device()
    print(f"🚀 Training on {device}...")

    # 1. Load Pretrained CLIP
    model_id = "openai/clip-vit-base-patch32" 
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)

    # 2. Apply LoRA
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

    # 3. Data Loader
    print(f"Loading data from: {args.train_json}")
    dataset = MemeCapTrainDataset(
        json_path=args.train_json,
        image_root=args.image_root,
        processor=processor
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # num_workers=0 is safer on Mac

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # 5. Training Loop
    for epoch in range(args.epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward Pass
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values,
                return_loss=True 
            )
            
            loss = outputs.loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(args.output_dir, f"lora_epoch_{epoch+1}")
        model.save_pretrained(save_path)
        print(f"✅ Saved adapter to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default paths assume running from ROOT directory
    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3) # Reduced to 3 for quick testing
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
