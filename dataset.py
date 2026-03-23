import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MemeSample:
    idx: int
    post_id: str
    image_path: str
    title: str
    caption: str
    img_fname: str

def first_nonempty_string(value) -> str:
    if not value: return ""
    if isinstance(value, str): return value.strip()
    if isinstance(value, list):
        for x in value:
            if isinstance(x, str) and x.strip(): return x.strip()
    return ""

def try_resolve_image_path(image_root: Path, img_fname: str) -> Optional[Path]:
    candidate = image_root / img_fname
    if candidate.exists(): return candidate.resolve()
    
    basename = Path(img_fname).name
    candidate2 = image_root / basename
    if candidate2.exists(): return candidate2.resolve()
    
    matches = list(image_root.rglob(basename))
    if matches: return matches[0].resolve()
    return None

def load_memecap_records(json_path: str, image_root: str, limit: int = None) -> List[MemeSample]:
    json_path = Path(json_path)
    image_root = Path(image_root)

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    if limit is not None:
        raw_data = raw_data[:limit]

    records = []
    skipped = 0

    for idx, item in enumerate(raw_data):
        img_fname = first_nonempty_string(item.get("img_fname") or item.get("img"))
        title = first_nonempty_string(item.get("title", ""))
        caption = first_nonempty_string(item.get("meme_captions") or item.get("caption"))
        post_id = str(item.get("post_id", idx))

        if not img_fname or not caption:
            skipped += 1
            continue

        image_path = try_resolve_image_path(image_root, img_fname)
        if image_path is None:
            skipped += 1
            continue

        records.append(MemeSample(
            idx=idx, post_id=post_id, image_path=str(image_path),
            title=title, caption=caption, img_fname=img_fname
        ))

    print(f"[Dataset] Loaded {len(records)} samples from {json_path} (Skipped: {skipped})")
    return records