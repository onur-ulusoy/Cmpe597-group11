import json
import os
import argparse
import random
from collections import Counter
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
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
    # Default to Neutral if the model completely fails to follow instructions
    return "Neutral"

def save_markdown_report(samples, image_root, output_path):
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    
    # Create a subfolder to hold ONLY the sampled images for GitHub rendering
    sample_img_dir = os.path.join(out_dir, "sampled_images")
    os.makedirs(sample_img_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# VLM Manual Quality Check (Random Samples)\n\n")
        for i, item in enumerate(samples):
            img_fname = item.get("img_fname", "")
            meme_caps = item.get("meme_captions", [])
            text = meme_caps[0] if isinstance(meme_caps, list) and len(meme_caps) > 0 else item.get("title", "")
            
            # Fetch the new VLM label
            label = item.get("vlm_sentiment_label", "Unknown")
            
            # Paths for copying the image
            src_img_path = os.path.join(image_root, img_fname)
            dst_img_path = os.path.join(sample_img_dir, img_fname)
            
            # Copy the image from the ignored data/ folder to the tracked outputs/ folder
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
            
            # The path inside the Markdown file is now just relative to the subfolder
            md_img_path = f"sampled_images/{img_fname}"
            
            f.write(f"### {i+1}. Predicted VLM Emotion: **{label}**\n")
            f.write(f"- **Meme Text**: {text}\n\n")
            f.write(f"<img src='{md_img_path}' width='400'>\n\n")
            f.write("---\n\n")

def main(args):
    valid_classes = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
    
    print(f"Loading VLM Model: {args.model_name} (This might take a while...)")
    
    # Load model in 4-bit precision to save GPU memory
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )

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
                
            try:
                raw_image = Image.open(img_path).convert('RGB')
            except Exception as e:
                # Silently skip broken images to keep the progress bar clean
                continue

            prompt = (
                f"USER: <image>\nYou are an expert at understanding internet memes. "
                f"Look at this meme image and read its caption: '{text}'. "
                f"Classify the intended emotion of this meme into EXACTLY ONE of these categories: "
                f"Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise. "
                f"Output only the category name and nothing else.\nASSISTANT:"
            )

            inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=10, 
                    do_sample=False 
                )

            input_length = inputs["input_ids"].shape[1]
            generated_text = processor.decode(output_ids[0][input_length:], skip_special_tokens=True)
            
            final_label = clean_vlm_output(generated_text, valid_classes)

            item["vlm_sentiment_label"] = final_label
            processed_data.append(item)
            all_processed_data.append(item)
            label_counter[final_label] += 1

        out_name = f"{split_name}_vlm_labels.json"
        save_json(processed_data, output_base / out_name)
        
        total = sum(label_counter.values())
        dist = {lbl: {"count": cnt, "percentage": (cnt/total)*100} for lbl, cnt in label_counter.items()}
        imbalance_report[split_name] = dist

    report_name = "vlm_imbalance_report.json"
    save_json(imbalance_report, output_base / report_name)
    
    # --- NEW: Generate Markdown Report ---
    if len(all_processed_data) >= args.num_samples:
        random.seed(42) # Seeded so it grabs the exact same random images as before if you want to compare
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
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    args = parser.parse_args()
    main(args)