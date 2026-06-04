import json
from collections import defaultdict

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return {}

def extract_data(json_data):
    """
    Extracts the label and the image filename for each meme.
    """
    extracted = {}
    
    if isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, dict):
                item_id = item.get('post_id') or item.get('id') or item.get('meme_id')
                item_label = item.get('vlm_sentiment_label') or item.get('label') or item.get('Qwen_VL_Chat') or item.get('llava_1.5_7b_hf')
                item_img = item.get('img_fname')
                
                if item_id is not None and item_label is not None:
                    # Store both the label and the image filename
                    extracted[str(item_id)] = {
                        "label": str(item_label),
                        # Fallback to constructing the name if img_fname is missing
                        "img_fname": str(item_img) if item_img else f"memes_{item_id}.png" 
                    }
    
    # Keeping the dict fallback just in case
    elif isinstance(json_data, dict):
        for key, value in json_data.items():
            if isinstance(value, dict):
                label = value.get('vlm_sentiment_label', value.get('label', value.get('Qwen_VL_Chat', value.get('llava_1.5_7b_hf'))))
                img = value.get('img_fname', f"memes_{key}.png")
                extracted[str(key)] = {"label": str(label), "img_fname": str(img)}
                
    return extracted

def main():
    # 1. Load Data
    qwen_train = load_json('Qwen_VL_Chat/train_qwen_labels.json')
    qwen_test = load_json('Qwen_VL_Chat/test_qwen_labels.json')
    llava_train = load_json('llava_1.5_7b_hf/train_vlm_labels.json')
    llava_test = load_json('llava_1.5_7b_hf/test_vlm_labels.json')

    # Merge train and test dictionaries
    qwen_all = {**extract_data(qwen_train), **extract_data(qwen_test)}
    llava_all = {**extract_data(llava_train), **extract_data(llava_test)}

    # Find common IDs to compare
    common_ids = set(qwen_all.keys()).intersection(set(llava_all.keys()))
    print(f"Found {len(common_ids)} common memes to compare.\n")

    # 2. Tracking Metrics
    agreement_counts = defaultdict(int)
    qwen_class_totals = defaultdict(int)
    mismatches = []

    for meme_id in common_ids:
        q_label = qwen_all[meme_id]["label"]
        l_label = llava_all[meme_id]["label"]
        img_fname = qwen_all[meme_id]["img_fname"]
        
        qwen_class_totals[q_label] += 1
        
        if q_label == l_label:
            agreement_counts[q_label] += 1
        else:
            mismatches.append({
                'id': meme_id,
                'qwen': q_label,
                'llava': l_label,
                'img_fname': img_fname
            })

    # 3. Print Console Report
    print("=== Model Consistency Report ===")
    print(f"{'Emotion Category':<18} | {'Total (Qwen)':<12} | {'Matches w/ LLaVA':<18} | {'Consistency %'}")
    print("-" * 70)
    
    for emotion, total in sorted(qwen_class_totals.items()):
        matches = agreement_counts[emotion]
        percentage = (matches / total) * 100 if total > 0 else 0
        print(f"{emotion:<18} | {total:<12} | {matches:<18} | {percentage:.2f}%")
        
    print("-" * 70)
    overall_match = sum(agreement_counts.values())
    overall_pct = (overall_match / len(common_ids)) * 100 if common_ids else 0
    print(f"OVERALL AGREEMENT: {overall_match} / {len(common_ids)} ({overall_pct:.2f}%)\n")

    # 4. Generate Markdown File for Manual Review
    md_filename = "mismatches_report.md"
    with open(md_filename, 'w', encoding='utf-8') as f:
        f.write("# VLM Label Discrepancy Report\n\n")
        f.write("Use the 'Ground Truth' column to write down the correct label.\n\n")
        f.write("| Meme Image | ID | Qwen-VL-Chat | LLaVA-1.5 | Manual Check (Ground Truth) |\n")
        f.write("|------------|----|--------------|-----------|-----------------------------|\n")
        
        # Sort mismatches by Qwen label, then LLaVA label to group similar disagreements together
        mismatches.sort(key=lambda x: (x['qwen'], x['llava']))
        
        for m in mismatches:
            # Construct relative path to the image
            img_rel_path = f"../../../data/memes/{m['img_fname']}"
            
            # Use HTML <img> tag instead of Markdown ![]() to constrain width to 250px 
            # so the table doesn't blow up and become unreadable.
            img_tag = f'<img src="{img_rel_path}" width="250" />'
            
            f.write(f"| {img_tag} | `{m['id']}` | **{m['qwen']}** | **{m['llava']}** |  |\n")
            
    print(f"Saved {len(mismatches)} discrepancies to '{md_filename}' for your manual review.")

if __name__ == "__main__":
    main()