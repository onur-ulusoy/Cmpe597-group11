import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
from tqdm import tqdm
import argparse

# Define the 7 emotion classes mapping
EMOTION_TO_ID = {
    "Anger": 0, "Disgust": 1, "Fear": 2, "Joy": 3, 
    "Neutral": 4, "Sadness": 5, "Surprise": 6
}
ID_TO_EMOTION = {v: k for k, v in EMOTION_TO_ID.items()}
NUM_CLASSES = len(EMOTION_TO_ID)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_clip_features(data, image_root, model, processor, device):
    """
    Passes all images and texts through frozen CLIP once before training.
    This saves massive amounts of time during the MLP training loop.
    """
    image_features_list = []
    text_features_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for item in tqdm(data, desc="Extracting CLIP Features"):
            # Extract Label
            label_str = item.get("vlm_sentiment_label", "Neutral")
            label_id = EMOTION_TO_ID.get(label_str, 4) # Default to Neutral if weird

            # Extract Text
            meme_caps = item.get("meme_captions", [])
            text = meme_caps[0] if isinstance(meme_caps, list) and len(meme_caps) > 0 else item.get("title", "")
            if not text:
                continue

            # Extract Image
            img_fname = item.get("img_fname", "")
            img_path = os.path.join(image_root, img_fname)
            if not os.path.exists(img_path):
                continue
            
            try:
                raw_image = Image.open(img_path).convert('RGB')
            except Exception:
                continue

            # Process through CLIP
            inputs = processor(text=[text], images=raw_image, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)            
            img_embeds = model.get_image_features(pixel_values=inputs.pixel_values)
            txt_embeds = model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

            # Normalize embeddings (standard practice for CLIP)
            img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
            txt_embeds = txt_embeds / txt_embeds.norm(p=2, dim=-1, keepdim=True)

            image_features_list.append(img_embeds.cpu())
            text_features_list.append(txt_embeds.cpu())
            labels_list.append(torch.tensor([label_id]))

    # Stack all tensors
    return (
        torch.cat(image_features_list, dim=0),
        torch.cat(text_features_list, dim=0),
        torch.cat(labels_list, dim=0)
    )

class UnimodalMLP(nn.Module):
    """
    A simple but robust MLP for Unimodal Classification (Image-Only or Text-Only).
    Using LayerNorm and GELU based on best practices from Task 2.2.
    """
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=7, dropout=0.3):
        super(UnimodalMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def train_and_evaluate(model, train_loader, test_loader, device, epochs=15, lr=1e-3, model_name="Unimodal Model"):
    """
    Standard training loop for PyTorch with Macro F1-Score evaluation.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_f1 = 0.0
    best_acc = 0.0
    
    print(f"\n{'='*40}")
    print(f"Training {model_name}")
    print(f"{'='*40}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluation Phase
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                outputs = model(features)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        
        # Save best metrics
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_acc = acc
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_train_loss:.4f} | Test Acc: {acc:.4f} | Test Macro F1: {macro_f1:.4f}")

    print(f"\n>>> BEST RESULTS FOR {model_name.upper()} <<<")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"Macro F1: {best_f1:.4f}\n")
    
    return best_acc, best_f1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the "Gold Standard" Qwen JSON files
    train_data = load_json(args.train_labels)
    test_data = load_json(args.test_labels)

    # 2. Load Frozen CLIP for Feature Extraction
    print("\nLoading OpenCLIP (ViT-L/14) to extract features...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # 3. Extract Features (This happens once and saves to RAM)
    print("\nExtracting Train Set Features...")
    train_img_feat, train_txt_feat, train_labels = extract_clip_features(train_data, args.image_root, clip_model, clip_processor, device)
    
    print("\nExtracting Test Set Features...")
    test_img_feat, test_txt_feat, test_labels = extract_clip_features(test_data, args.image_root, clip_model, clip_processor, device)

    # Free up GPU memory since we don't need CLIP anymore
    del clip_model
    torch.cuda.empty_cache()

    # 4. Create PyTorch DataLoaders
    batch_size = 64
    
    # Image Datasets
    train_img_dataset = TensorDataset(train_img_feat, train_labels)
    test_img_dataset = TensorDataset(test_img_feat, test_labels)
    
    train_img_loader = DataLoader(train_img_dataset, batch_size=batch_size, shuffle=True)
    test_img_loader = DataLoader(test_img_dataset, batch_size=batch_size, shuffle=False)

    # Text Datasets
    train_txt_dataset = TensorDataset(train_txt_feat, train_labels)
    test_txt_dataset = TensorDataset(test_txt_feat, test_labels)
    
    train_txt_loader = DataLoader(train_txt_dataset, batch_size=batch_size, shuffle=True)
    test_txt_loader = DataLoader(test_txt_dataset, batch_size=batch_size, shuffle=False)

    # 5. Train Image-Only Baseline
    img_model = UnimodalMLP(input_dim=768, num_classes=NUM_CLASSES).to(device)
    img_acc, img_f1 = train_and_evaluate(img_model, train_img_loader, test_img_loader, device, epochs=15, model_name="Image-Only Baseline")

    # 6. Train Text-Only Baseline
    txt_model = UnimodalMLP(input_dim=768, num_classes=NUM_CLASSES).to(device)
    txt_acc, txt_f1 = train_and_evaluate(txt_model, train_txt_loader, test_txt_loader, device, epochs=15, model_name="Text-Only Baseline")

    # 7. Print and Save Final Comparison Table
    result_text = (
        f"\n{'='*50}\n"
        f"TASK 2.3.b: UNIMODAL BASELINES FINAL COMPARISON\n"
        f"{'='*50}\n"
        f"{'Modality':<20} | {'Accuracy':<12} | {'Macro F1':<12}\n"
        f"-" * 50 + "\n"
        f"{'Image-Only MLP':<20} | {img_acc:.4f}{'':<6} | {img_f1:.4f}\n"
        f"{'Text-Only MLP':<20} | {txt_acc:.4f}{'':<6} | {txt_f1:.4f}\n"
        f"{'='*50}\n"
    )
    
    print(result_text)
    
    # Save the table to the outputs folder
    out_dir = "outputs/sentiment_classification"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "unimodal_results.txt")
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(result_text)
        
    print(f"Results successfully saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Ensure these paths point to your newly generated Qwen labels
    parser.add_argument("--train_labels", type=str, default="outputs/sentiment_classification/labels/Qwen_VL_Chat/train_qwen_labels.json")
    parser.add_argument("--test_labels", type=str, default="outputs/sentiment_classification/labels/Qwen_VL_Chat/test_qwen_labels.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    args = parser.parse_args()
    main(args)