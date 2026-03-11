import argparse
from pathlib import Path

import torch

from common import (
    MemeCapDataset,
    compute_metrics,
    l2_normalize,
    make_query_embeddings,
    print_metrics,
    save_json,
    set_seed,
)
from openclip_backend import OpenCLIPBackend
from siglip_backend import SigLIPBackend
from blip_reranker import BLIPReranker


def build_backend(args):
    if args.model_family == "openclip":
        return OpenCLIPBackend(
            model_name=args.openclip_model_name,
            pretrained=args.openclip_pretrained,
            device=args.device,
        )
    if args.model_family == "siglip":
        return SigLIPBackend(
            checkpoint=args.siglip_checkpoint,
            device=args.device,
        )
    raise ValueError(f"Unsupported model_family: {args.model_family}")


def make_run_name(args) -> str:
    base = f"{args.model_family}_{args.input_type}"
    if args.input_type == "type2":
        alpha_tag = f"a{args.alpha}".replace(".", "p")
        return f"{base}_{alpha_tag}"
    return base


def run(args):
    set_seed(args.seed)

    dataset = MemeCapDataset.from_paths(
        data_dir=args.data_dir,
        test_json=args.test_json,
        image_root=args.image_root,
        limit=args.limit,
    )

    print(f"Using device: {args.device}")
    print(f"Test JSON: {dataset.json_path}")

    image_paths = [s.image_path for s in dataset.samples]
    titles = [s.title for s in dataset.samples]
    captions = [s.caption for s in dataset.samples]

    backend = build_backend(args)

    image_emb = backend.encode_images(image_paths, batch_size=args.batch_size)
    caption_emb = backend.encode_texts(captions, batch_size=args.text_batch_size)

    title_emb = None
    if args.input_type == "type2":
        title_emb = backend.encode_texts(titles, batch_size=args.text_batch_size)

    query_emb = make_query_embeddings(
        image_emb=image_emb,
        title_emb=title_emb,
        input_type=args.input_type,
        alpha=args.alpha,
    )
    caption_emb = l2_normalize(caption_emb)

    score_matrix = query_emb @ caption_emb.T

    run_name = make_run_name(args)
    metrics = compute_metrics(score_matrix)
    print_metrics(run_name, metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        output_dir / f"{run_name}_metrics.json",
        {
            "run_name": run_name,
            "model_family": args.model_family,
            "input_type": args.input_type,
            "alpha": args.alpha if args.input_type == "type2" else None,
            "num_queries": len(dataset.samples),
            "test_json": str(dataset.json_path),
            "image_root": str(Path(args.image_root)),
            "metrics": metrics,
            "missing_image_examples": dataset.missing_examples[:20],
        },
    )

    if args.use_blip_reranker:
        reranker = BLIPReranker(checkpoint=args.blip_checkpoint, device=args.device)
        reranked_matrix = reranker.rerank_topk(
            image_paths=image_paths,
            captions=captions,
            base_scores=score_matrix,
            top_k=args.blip_top_k,
            batch_size=args.blip_batch_size,
            blend_lambda=args.blip_blend_lambda,
        )

        rerank_name = f"{run_name}_blip_rerank"
        rerank_metrics = compute_metrics(reranked_matrix)
        print_metrics(rerank_name, rerank_metrics)

        save_json(
            output_dir / f"{rerank_name}_metrics.json",
            {
                "run_name": rerank_name,
                "base_model_family": args.model_family,
                "input_type": args.input_type,
                "alpha": args.alpha if args.input_type == "type2" else None,
                "blip_checkpoint": args.blip_checkpoint,
                "blip_top_k": args.blip_top_k,
                "blip_blend_lambda": args.blip_blend_lambda,
                "num_queries": len(dataset.samples),
                "metrics": rerank_metrics,
            },
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Zero-shot meme caption retrieval evaluator")

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--test_json", type=str, default=None)
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="outputs/zero_shot")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--model_family", type=str, required=True, choices=["openclip", "siglip"])
    parser.add_argument("--input_type", type=str, required=True, choices=["type1", "type2"])
    parser.add_argument("--alpha", type=float, default=0.7)

    parser.add_argument("--openclip_model_name", type=str, default="ViT-L-14")
    parser.add_argument("--openclip_pretrained", type=str, default="laion2b_s32b_b82k")

    parser.add_argument("--siglip_checkpoint", type=str, default="google/siglip2-base-patch16-384")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--use_blip_reranker", action="store_true")
    parser.add_argument("--blip_checkpoint", type=str, default="Salesforce/blip-itm-base-coco")
    parser.add_argument("--blip_top_k", type=int, default=50)
    parser.add_argument("--blip_batch_size", type=int, default=16)
    parser.add_argument("--blip_blend_lambda", type=float, default=0.5)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)