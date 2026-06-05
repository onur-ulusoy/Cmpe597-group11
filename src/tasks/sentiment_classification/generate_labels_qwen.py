#!/usr/bin/env python3
import json
import os
import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


VALID_CLASSES = [
    "Anger",
    "Disgust",
    "Fear",
    "Joy",
    "Neutral",
    "Sadness",
    "Surprise",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_meme_caption(item):
    """
    Use meme_captions[0], as required by the project.
    Fall back to title only if meme_captions is missing or empty.
    """
    meme_caps = item.get("meme_captions", [])

    if (
        isinstance(meme_caps, list)
        and len(meme_caps) > 0
        and isinstance(meme_caps[0], str)
        and meme_caps[0].strip()
    ):
        return meme_caps[0].strip()

    title = item.get("title", "")
    if isinstance(title, str) and title.strip():
        return title.strip()

    return ""


def clean_vlm_output(text, valid_classes=VALID_CLASSES):
    """
    Robustly extract exactly one valid class.

    Returns:
        one of VALID_CLASSES, or "INVALID"

    This avoids the dangerous behavior of returning Joy from:
        "not Joy, probably Sadness"
    """
    if not isinstance(text, str) or not text.strip():
        return "INVALID"

    text = text.strip()

    # Remove common prefixes.
    text = re.sub(r"^\s*(emotion|label|answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    lowered = text.lower().strip()

    # Exact label after removing non-letters.
    cleaned_alpha = re.sub(r"[^a-zA-Z]", "", lowered)

    for label in valid_classes:
        if cleaned_alpha == label.lower():
            return label

    # If exactly one class name appears as a full word, accept it.
    matches = []
    for label in valid_classes:
        pattern = rf"\b{re.escape(label.lower())}\b"
        if re.search(pattern, lowered):
            matches.append(label)

    unique_matches = list(dict.fromkeys(matches))

    if len(unique_matches) == 1:
        return unique_matches[0]

    return "INVALID"


def build_prompt(caption, annotation_mode):
    if annotation_mode == "caption_only":
        return f"""You are annotating the dominant intended emotion of a meme caption.

The caption is a human-written explanation of what the meme poster is trying to convey.
Classify the emotion or attitude expressed by the meme poster.

Important rules:
- Do not assume every humorous meme is Joy.
- Do not use Fear unless the poster is expressing anxiety, danger, threat, panic, or being scared.
- Do not use Disgust for ordinary disappointment. Use Disgust only for revulsion, contempt, or strong rejection.
- Use Anger for frustration, annoyance, blame, criticism, or complaint.
- Use Sadness for disappointment, heartbreak, loss, hopelessness, or something being ruined.
- Use Surprise for shock, disbelief, confusion, or unexpectedness.
- Use Neutral only when there is no clear emotional attitude.

Labels:
Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise

Caption:
{caption}

Return only one label.

Label:"""

    if annotation_mode == "image_caption":
        return f"""You are annotating the dominant intended emotion of an internet meme.

The caption below is a human-written explanation of the meme poster's intended meaning.
Treat the caption as the primary evidence.
Use the image only as supporting context to understand the joke, irony, or visual contrast.

Do NOT label based only on literal visual features such as:
- fire or destruction
- weapons
- angry-looking characters
- scared-looking characters
- dark colors
- dramatic scenes

Classify the emotion or attitude expressed by the meme poster.

Important rules:
- Do not assume every humorous meme is Joy.
- Do not use Fear unless the poster is expressing anxiety, danger, threat, panic, or being scared.
- Do not use Disgust for ordinary disappointment. Use Disgust only for revulsion, contempt, or strong rejection.
- Use Anger for frustration, annoyance, blame, criticism, or complaint.
- Use Sadness for disappointment, heartbreak, loss, hopelessness, or something being ruined.
- Use Surprise for shock, disbelief, confusion, or unexpectedness.
- Use Neutral only when there is no clear emotional attitude.

Labels:
Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise

Caption:
{caption}

Return only one label.

Label:"""

    raise ValueError(f"Unknown annotation_mode: {annotation_mode}")


def run_qwen(model, tokenizer, query):
    with torch.no_grad():
        response, _ = model.chat(
            tokenizer,
            query=query,
            history=None,
        )
    return response


def compute_distribution(items, label_key):
    counter = Counter(item.get(label_key, "MISSING") for item in items)
    total = sum(counter.values())

    valid_total = sum(counter[label] for label in VALID_CLASSES)
    invalid_or_missing_total = counter.get("INVALID", 0) + counter.get("MISSING", 0)

    all_label_distribution = {}
    for label, count in counter.most_common():
        all_label_distribution[label] = {
            "count": count,
            "percentage_among_all": round((count / total) * 100, 2) if total else 0.0,
        }

    valid_label_distribution = {}
    for label in VALID_CLASSES:
        count = counter.get(label, 0)
        valid_label_distribution[label] = {
            "count": count,
            "percentage_among_valid": round((count / valid_total) * 100, 2) if valid_total else 0.0,
            "percentage_among_all": round((count / total) * 100, 2) if total else 0.0,
        }

    return {
        "total_items": total,
        "valid_labeled_items": valid_total,
        "invalid_or_missing_items": invalid_or_missing_total,
        "all_label_distribution": all_label_distribution,
        "valid_label_distribution": valid_label_distribution,
    }


def stratified_sample(items, label_key, per_class=10, seed=42):
    random.seed(seed)

    by_label = defaultdict(list)
    for item in items:
        label = item.get(label_key, "MISSING")
        by_label[label].append(item)

    sampled = []

    for label in VALID_CLASSES + ["INVALID"]:
        group = by_label.get(label, [])
        if not group:
            continue

        k = min(per_class, len(group))
        sampled.extend(random.sample(group, k))

    return sampled


def save_markdown_report(samples, image_root, output_path, label_key, raw_key, annotation_mode):
    output_path = Path(output_path)
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_img_dir = out_dir / "sampled_images"
    sample_img_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Qwen-VL Manual Quality Check - {annotation_mode}\n\n")
        f.write("Use this file to inspect silver-label noise.\n\n")
        f.write("Suggested manual fields:\n")
        f.write("- Correct? yes/no/ambiguous\n")
        f.write("- Human label\n")
        f.write("- Notes\n\n")
        f.write("---\n\n")

        for i, item in enumerate(samples):
            img_fname = item.get("img_fname", "")
            caption = get_meme_caption(item)
            title = item.get("title", "")
            label = item.get(label_key, "MISSING")
            raw_response = item.get(raw_key, "")

            src_img_path = Path(image_root) / img_fname
            dst_img_path = sample_img_dir / img_fname

            if img_fname and src_img_path.exists():
                try:
                    shutil.copy2(src_img_path, dst_img_path)
                except Exception as e:
                    print(f"Warning: could not copy image {src_img_path}: {e}")

            f.write(f"## {i + 1}. Predicted label: **{label}**\n\n")

            if title:
                f.write(f"- **Title:** {title}\n")

            if caption:
                f.write(f"- **Meme caption:** {caption}\n")

            f.write(f"- **Raw model response:** `{raw_response}`\n")
            f.write("- **Manual judgment:** \n")
            f.write("- **Human label if different:** \n")
            f.write("- **Notes:** \n\n")

            if img_fname:
                md_img_path = f"sampled_images/{img_fname}"
                f.write(f"<img src='{md_img_path}' width='400'>\n\n")

            f.write("---\n\n")


def load_model_and_tokenizer(args):
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    print(f"Loading model: {args.model_name}")

    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    if args.device_map_auto:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).cuda().eval()

    return model, tokenizer


def process_split(split_name, input_path, model, tokenizer, args, output_base):
    print(f"\nProcessing {split_name} split with annotation_mode={args.annotation_mode}")

    data = load_json(input_path)

    label_key = f"qwen_{args.annotation_mode}_sentiment_label"
    raw_key = f"qwen_{args.annotation_mode}_raw_response"

    processed_data = []

    skipped_missing_caption = 0
    skipped_missing_image = 0
    exception_count = 0
    invalid_count = 0

    for item in tqdm(data, desc=f"{split_name} / {args.annotation_mode}"):
        caption = get_meme_caption(item)

        if not caption:
            skipped_missing_caption += 1
            continue

        img_fname = item.get("img_fname", "")
        img_path = os.path.join(args.image_root, img_fname) if img_fname else ""

        if args.annotation_mode == "image_caption":
            if not img_path or not os.path.exists(img_path):
                skipped_missing_image += 1
                continue

        raw_response = ""
        final_label = "INVALID"

        try:
            prompt_text = build_prompt(caption, args.annotation_mode)
            query = make_query(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                img_path=img_path,
                annotation_mode=args.annotation_mode,
            )

            raw_response = run_qwen(model, tokenizer, query)
            final_label = clean_vlm_output(raw_response)

            if final_label == "INVALID" and args.retry_invalid:
                retry_prompt = build_retry_prompt(caption, args.annotation_mode)
                retry_query = make_query(
                    tokenizer=tokenizer,
                    prompt_text=retry_prompt,
                    img_path=img_path,
                    annotation_mode=args.annotation_mode,
                )

                retry_response = run_qwen(model, tokenizer, retry_query)
                retry_label = clean_vlm_output(retry_response)

                item[f"qwen_{args.annotation_mode}_first_raw_response"] = raw_response
                item[f"qwen_{args.annotation_mode}_retry_raw_response"] = retry_response

                raw_response = retry_response
                final_label = retry_label

        except Exception as e:
            exception_count += 1
            raw_response = f"ERROR: {repr(e)}"
            final_label = "INVALID"

        if final_label == "INVALID":
            invalid_count += 1

        item[label_key] = final_label
        item[raw_key] = raw_response
        item[f"qwen_{args.annotation_mode}_model"] = args.model_name
        item[f"qwen_{args.annotation_mode}_annotation_mode"] = args.annotation_mode

        processed_data.append(item)

    output_json = output_base / f"{split_name}_qwen_{args.annotation_mode}_labels.json"
    save_json(processed_data, output_json)

    report = compute_distribution(processed_data, label_key)
    report["split"] = split_name
    report["annotation_mode"] = args.annotation_mode
    report["input_path"] = str(input_path)
    report["model_name"] = args.model_name
    report["skipped_missing_caption"] = skipped_missing_caption
    report["skipped_missing_image"] = skipped_missing_image
    report["exception_count"] = exception_count
    report["invalid_count"] = invalid_count

    report_json = output_base / f"{split_name}_qwen_{args.annotation_mode}_imbalance_report.json"
    save_json(report, report_json)

    print(f"Saved labels to: {output_json}")
    print(f"Saved report to: {report_json}")
    print(f"Valid labels: {report['valid_labeled_items']}")
    print(f"Invalid/missing labels: {report['invalid_or_missing_items']}")

    return processed_data, report


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_slug = args.model_name.split("/")[-1].replace("-", "_")

    if args.annotation_mode == "caption_only":
        run_slug = f"{model_slug}_caption_only"
    elif args.annotation_mode == "image_caption":
        run_slug = f"{model_slug}_image_caption"
    else:
        raise ValueError(f"Unknown annotation_mode: {args.annotation_mode}")

    output_base = Path(args.output_dir) / run_slug
    output_base.mkdir(parents=True, exist_ok=True)

    save_json(vars(args), output_base / "generation_args.json")

    model, tokenizer = load_model_and_tokenizer(args)

    splits = []
    if args.split in ["train", "both"]:
        splits.append(("train", args.train_input))
    if args.split in ["test", "both"]:
        splits.append(("test", args.test_input))

    all_processed_data = []
    combined_report = {}

    for split_name, input_path in splits:
        processed_data, report = process_split(
            split_name=split_name,
            input_path=input_path,
            model=model,
            tokenizer=tokenizer,
            args=args,
            output_base=output_base,
        )

        all_processed_data.extend(processed_data)
        combined_report[split_name] = report

    combined_report_path = output_base / f"qwen_{args.annotation_mode}_combined_imbalance_report.json"
    save_json(combined_report, combined_report_path)

    label_key = f"qwen_{args.annotation_mode}_sentiment_label"
    raw_key = f"qwen_{args.annotation_mode}_raw_response"

    if args.manual_per_class > 0 and all_processed_data:
        samples = stratified_sample(
            all_processed_data,
            label_key=label_key,
            per_class=args.manual_per_class,
            seed=args.seed,
        )

        manual_check_path = output_base / f"manual_check_{args.annotation_mode}_stratified.md"

        save_markdown_report(
            samples=samples,
            image_root=args.image_root,
            output_path=manual_check_path,
            label_key=label_key,
            raw_key=raw_key,
            annotation_mode=args.annotation_mode,
        )

        print(f"Saved manual check to: {manual_check_path}")

    print(f"\nGeneration complete. Outputs saved to: {output_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_input",
        type=str,
        default="data/memes-trainval.json",
        help="Path to MemeCap train/validation JSON.",
    )

    parser.add_argument(
        "--test_input",
        type=str,
        default="data/memes-test.json",
        help="Path to MemeCap test JSON.",
    )

    parser.add_argument(
        "--image_root",
        type=str,
        default="data/memes",
        help="Directory containing meme images.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/sentiment_classification/labels",
        help="Base output directory.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen-VL-Chat",
        help="Qwen model name.",
    )

    parser.add_argument(
        "--annotation_mode",
        type=str,
        choices=["caption_only", "image_caption"],
        default="caption_only",
        help=(
            "caption_only uses only meme captions; "
            "image_caption uses both meme image and meme caption."
        ),
    )

    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split to process.",
    )

    parser.add_argument(
        "--manual_per_class",
        type=int,
        default=10,
        help="Number of examples per predicted class for manual-check markdown.",
    )

    parser.add_argument(
        "--retry_invalid",
        action="store_true",
        help="Retry once with a stricter prompt if the first response is invalid.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Model dtype.",
    )

    parser.add_argument(
        "--device_map_auto",
        action="store_true",
        help="Use device_map='auto' instead of forcing .cuda().",
    )

    args = parser.parse_args()
    main(args)