import argparse
import os
import sys
import json
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.common.classification_dataset import load_classification_records
from src.models.pretrained.openclip import OpenCLIPBackend


def get_text_hash(text):
    text = "" if text is None else str(text).strip()
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, preprocess):
        self.paths = [Path(p) for p in paths]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        return self.preprocess(img), p.name


def collect_records(train_json, test_json, image_root, limit=None):
    train_records = load_classification_records(
        json_path=train_json,
        image_root=image_root,
        limit=limit,
    )

    test_records = load_classification_records(
        json_path=test_json,
        image_root=image_root,
        limit=limit,
    )

    return train_records + test_records


def extract_image_features(records, output_dir, backend, batch_size):
    img_save_dir = Path(output_dir) / "images"
    img_save_dir.mkdir(parents=True, exist_ok=True)

    # sample.image_path is a Path object in your dataset loader.
    unique_paths = sorted(set([str(sample.image_path) for sample in records]))

    dataset = ImageDataset(unique_paths, backend.preprocess)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
    )

    print(f"Extracting {len(unique_paths)} unique images...")

    for images, fnames in tqdm(loader, desc="Images"):
        missing_mask = [
            not (img_save_dir / f"{fname}.pt").exists()
            for fname in fnames
        ]

        if not any(missing_mask):
            continue

        images = images.to(backend.device)

        with torch.no_grad():
            feats = backend.model.encode_image(images)
            feats = F.normalize(feats, p=2, dim=-1)

        for fname, feat in zip(fnames, feats):
            save_path = img_save_dir / f"{fname}.pt"
            if not save_path.exists():
                torch.save(feat.cpu(), save_path)


def extract_text_features(records, output_dir, backend, batch_size):
    text_save_dir = Path(output_dir) / "texts"
    text_save_dir.mkdir(parents=True, exist_ok=True)

    # Critical: use exactly sample.text from load_classification_records().
    unique_texts = sorted(set([str(sample.text).strip() for sample in records if str(sample.text).strip()]))

    print(f"Extracting {len(unique_texts)} unique classification texts...")

    mapping = {}

    for i in tqdm(range(0, len(unique_texts), batch_size), desc="Texts"):
        batch_texts = unique_texts[i:i + batch_size]

        batch_to_encode = []
        batch_hashes = []

        for text in batch_texts:
            t_hash = get_text_hash(text)
            mapping[text] = t_hash

            save_path = text_save_dir / f"{t_hash}.pt"
            if not save_path.exists():
                batch_to_encode.append(text)
                batch_hashes.append(t_hash)

        if not batch_to_encode:
            continue

        feats = backend.encode_texts(batch_to_encode, batch_size=len(batch_to_encode))

        for t_hash, feat in zip(batch_hashes, feats):
            torch.save(feat.cpu(), text_save_dir / f"{t_hash}.pt")

    mapping_path = Path(output_dir) / "text_mapping.json"

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)

    print(f"Saved text mapping to {mapping_path}")


def verify_features(records, output_dir):
    output_dir = Path(output_dir)
    missing_images = []
    missing_texts = []

    with open(output_dir / "text_mapping.json", "r", encoding="utf-8") as f:
        text_mapping = json.load(f)

    for sample in records:
        img_path = output_dir / "images" / f"{sample.image_path.name}.pt"
        text = str(sample.text).strip()
        t_hash = text_mapping.get(text, get_text_hash(text))
        text_path = output_dir / "texts" / f"{t_hash}.pt"

        if not img_path.exists():
            missing_images.append(str(img_path))

        if not text_path.exists():
            missing_texts.append(str(text_path))

    print(f"Verification missing images: {len(missing_images)}")
    print(f"Verification missing texts : {len(missing_texts)}")

    if missing_images or missing_texts:
        if missing_images:
            print("Example missing image:", missing_images[0])
        if missing_texts:
            print("Example missing text:", missing_texts[0])
        raise RuntimeError("Feature extraction verification failed.")

    print("Feature verification OK.")


def main(args):
    from src.tasks.caption_classification.train.caption_type import get_device

    device = get_device(args.device)

    backend = OpenCLIPBackend(
        args.model_name,
        args.pretrained,
        device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(
        args.train_json,
        args.test_json,
        args.image_root,
        limit=args.limit,
    )

    print(f"Total classification records: {len(records)}")

    extract_image_features(records, output_dir, backend, args.batch_size)
    extract_text_features(records, output_dir, backend, args.batch_size)
    verify_features(records, output_dir)

    print("Extraction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="data/features/openclip_vit_l_14")

    parser.add_argument("--model_name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    main(args)