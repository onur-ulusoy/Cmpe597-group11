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

# 7-class emotion mapping
EMOTION_TO_ID = {
    "Anger": 0, "Disgust": 1, "Fear": 2, "Joy": 3, 
    "Neutral": 4, "Sadness": 5, "Surprise": 6
}
NUM_CLASSES = len(EMOTION_TO_ID)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_clip_features(data, image_root, model, processor, device):
    """
    Extracts paired image and text features. 
    Includes truncation=True to prevent sequence length errors.
    """
    image_features_list = []
    text_features_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for item in tqdm(data, desc="Extracting CLIP Features"):
            label_str = item.get("vlm_sentiment_label", "Neutral")
            label_id = EMOTION_TO_ID.get(label_str, 4)

            meme_caps = item.get("meme_captions", [])
            text = meme_caps[0] if isinstance(meme_caps, list) and len(meme_caps) > 0 else item.get("title", "")
            if not text:
                continue

            img_fname = item.get("img_fname", "")
            img_path = os.path.join(image_root, img_fname)
            if not os.path.exists(img_path):
                continue
            
            try:
                raw_image = Image.open(img_path).convert('RGB')
            except Exception:
                continue

            # Standardize text length to 77 tokens
            inputs = processor(text=[text], images=raw_image, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            
            img_embeds = model.get_image_features(pixel_values=inputs.pixel_values)
            txt_embeds = model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

            # L2 Normalization
            img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
            txt_embeds = txt_embeds / txt_embeds.norm(p=2, dim=-1, keepdim=True)

            image_features_list.append(img_embeds.cpu())
            text_features_list.append(txt_embeds.cpu())
            labels_list.append(torch.tensor([label_id]))

    return (
        torch.cat(image_features_list, dim=0),
        torch.cat(text_features_list, dim=0),
        torch.cat(labels_list, dim=0)
    )

class LateFusionMLP(nn.Module):
    """
    Refined Architecture: Lower capacity (256 units) and higher Dropout (0.5) 
    to force generalization and prevent overfitting.
    """
    def __init__(self, input_dim=1536, hidden_dim=256, num_classes=7, dropout=0.5):
        super(LateFusionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Simplified to one main hidden layer to prevent memorizing small datasets
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, img_features, txt_features):
        fused_features = torch.cat([img_features, txt_features], dim=1)
        return self.network(fused_features)

def train_and_evaluate(model, train_loader, test_loader, device, epochs=20, lr=5e-4):
    """
    Training loop with Label Smoothing and high Weight Decay.
    """
    # Label smoothing 0.1 prevents the model from over-fitting to the "silver" labels
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Increased weight_decay to 1e-2 for stronger L2 regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    best_f1 = 0.0
    best_acc = 0.0
    
    print(f"\n{'='*45}")
    print("Training Regularized Multimodal Late Fusion")
    print(f"{'='*45}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for img_feat, txt_feat, labels in train_loader:
            img_feat, txt_feat, labels = img_feat.to(device), txt_feat.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(img_feat, txt_feat)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for img_feat, txt_feat, labels in test_loader:
                img_feat, txt_feat = img_feat.to(device), txt_feat.to(device)
                outputs = model(img_feat, txt_feat)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_acc = acc
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Acc: {acc:.4f} | Test Macro F1: {macro_f1:.4f}")

    return best_acc, best_f1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    train_data = load_json(args.train_labels)
    test_data = load_json(args.test_labels)

    # 2. Extract Features
    print("\nLoading CLIP and extracting features...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    train_img_feat, train_txt_feat, train_labels = extract_clip_features(train_data, args.image_root, clip_model, clip_processor, device)
    test_img_feat, test_txt_feat, test_labels = extract_clip_features(test_data, args.image_root, clip_model, clip_processor, device)

    del clip_model
    torch.cuda.empty_cache()

    # 3. DataLoaders
    batch_size = 64
    train_loader = DataLoader(TensorDataset(train_img_feat, train_txt_feat, train_labels), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_img_feat, test_txt_feat, test_labels), batch_size=batch_size, shuffle=False)

    # 4. Train
    model = LateFusionMLP(input_dim=1536).to(device)
    multi_acc, multi_f1 = train_and_evaluate(model, train_loader, test_loader, device)

    # 5. Save Results
    out_dir = "outputs/sentiment_classification"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "multimodal_results.txt")

    result_text = (
        f"\n{'='*60}\n"
        f"TASK 2.3.c: MULTIMODAL LATE FUSION\n"
        f"{'='*60}\n"
        f"Modality     : Concatenated CLIP (Img+Txt)\n"
        f"Dropout      : 0.5\n"
        f"Weight Decay : 0.01\n"
        f"Accuracy     : {multi_acc:.4f}\n"
        f"Macro F1     : {multi_f1:.4f}\n"
        f"{'='*60}\n"
    )

    print(result_text)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(result_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_labels", type=str, default="outputs/sentiment_classification/labels/Qwen_VL_Chat/train_qwen_labels.json")
    parser.add_argument("--test_labels", type=str, default="outputs/sentiment_classification/labels/Qwen_VL_Chat/test_qwen_labels.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    args = parser.parse_args()
    main(args)