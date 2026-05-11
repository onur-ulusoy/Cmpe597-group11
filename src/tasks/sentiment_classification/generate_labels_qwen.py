import json
import os
import argparse
import random
from collections import Counter
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import shutil

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def clean_vlm_output(text, valid_classes):
    """
    Cleans the raw VLM text generation to extract exactly one of the valid classes.
    """
    text = text.lower().strip()
    for valid_class in valid_classes:
        if valid_class.lower() in text:
            return valid_class
    return "Neutral"

def save_markdown_report(samples, image_root, output_path):
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    
    sample_img_dir = os.path.join(out_dir, "sampled_images")
    os.makedirs(sample_img_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Qwen-VL Manual Quality Check (Random Samples)\n\n")
        for i, item in enumerate(samples):
            img_fname = item.get("img_fname", "")
            meme_caps = item.get("meme_captions", [])
            text = meme_caps[0] if isinstance(meme_caps, list) and len(meme_caps) > 0 else item.get("title", "")
            
            label = item.get("vlm_sentiment_label", "Unknown")
            
            src_img_path = os.path.join(image_root, img_fname)
            dst_img_path = os.path.join(sample_img_dir, img_fname)
            
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            
            md_img_path = f"sampled_images/{img_fname}"
            
            f.write(f"### {i+1}. Predicted Qwen Emotion: **{label}**\n")
            f.write(f"- **Meme Text**: {text}\n\n")
            f.write(f"<img src='{md_img_path}' width='400'>\n\n")
            f.write("---\n\n")

def main(args):
    valid_classes = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
    
    print(f"Loading VLM Model: {args.model_name} (This might take a while...)")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # Qwen-VL requires trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        trust_remote_code=True
    )
    
    # REMOVE the quantization_config completely.
    # Load the model directly in bfloat16 and send it to CUDA.
    
    print(f"Loading VLM Model: {args.model_name} in bfloat16 (This might take a while...)")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 # Load natively in bfloat16
    ).cuda().eval() # Send to GPU and set to eval mode manually

    model_slug = args.model_name.split("/")[-1].replace("-", "_")
    output_base = Path(args.output_dir) / model_slug
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_processed_data = []
    imbalance_report = {}

    for split_name, input_path in [("test", args.test_input), ("train", args.train_input)]:
        print(f"\nProcessing {split_name} split using {model_slug}...")
        data = load_json(input_path)
        
        label_counter = Counter()
        processed_data = []

        for item in tqdm(data):
            meme_caps = item.get("meme_captions", [])
            text = meme_caps[0] if isinstance(meme_caps, list) and len(meme_caps) > 0 else item.get("title", "")
            img_fname = item.get("img_fname", "")
            
            if not text or not img_fname:
                continue

            img_path = os.path.join(args.image_root, img_fname)
            if not os.path.exists(img_path):
                continue

            # We use the few-shot prompt that teaches the model about meme irony
            prompt_text = (
                f"You are an AI that classifies the true emotional intent of internet memes. "
                f"Memes often use irony. Here are examples of how to classify them:\n\n"
                f"Example 1:\n"
                f"Caption: 'Me looking at my bank account after buying expensive coffee.'\n"
                f"Image: A crying person.\n"
                f"Intent: This is self-deprecating humor.\n"
                f"Emotion: Joy\n\n"
                f"Example 2:\n"
                f"Caption: 'When the teacher assigns homework on Friday.'\n"
                f"Image: A character smiling while everything is on fire.\n"
                f"Intent: This expresses frustration and sarcasm.\n"
                f"Emotion: Disgust\n\n"
                f"Example 3:\n"
                f"Caption: 'I actually failed my final exam today.'\n"
                f"Image: A genuinely sad cat.\n"
                f"Intent: This is literal and depressing.\n"
                f"Emotion: Sadness\n\n"
                f"Now, look at the provided meme image and read its caption: '{text}'. "
                f"Classify its true intended emotion into EXACTLY ONE of these categories: "
                f"Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise. "
                f"Output ONLY the category name."
            )

            # Qwen-VL specific formatting
            query = tokenizer.from_list_format([
                {'image': img_path},
                {'text': prompt_text}
            ])

            try:
                with torch.no_grad():
                    response, _ = model.chat(tokenizer, query=query, history=None)
                
                final_label = clean_vlm_output(response, valid_classes)
            except Exception as e:
                print(f"Error processing {img_fname}: {e}")
                final_label = "Neutral"

            item["vlm_sentiment_label"] = final_label
            processed_data.append(item)
            all_processed_data.append(item)
            label_counter[final_label] += 1

        out_name = f"{split_name}_qwen_labels.json"
        save_json(processed_data, output_base / out_name)
        
        total = sum(label_counter.values())
        dist = {lbl: {"count": cnt, "percentage": (cnt/total)*100} for lbl, cnt in label_counter.items()}
        imbalance_report[split_name] = dist

    report_name = "qwen_imbalance_report.json"
    save_json(imbalance_report, output_base / report_name)
    
    if len(all_processed_data) >= args.num_samples:
        random.seed(42)
        sampled_items = random.sample(all_processed_data, args.num_samples)
        md_name = "manual_check.md"
        save_markdown_report(sampled_items, args.image_root, output_base / md_name)

    print(f"\nGeneration complete for {model_slug}. Outputs saved to {output_base}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str, default="data/memes-trainval.json")
    parser.add_argument("--test_input", type=str, default="data/memes-test.json")
    parser.add_argument("--output_dir", type=str, default="outputs/sentiment_classification/labels")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-VL-Chat")
    args = parser.parse_args()
    main(args)