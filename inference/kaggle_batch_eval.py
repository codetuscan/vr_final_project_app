#!/usr/bin/env python3
"""Batch evaluation script for visual product search (Condition C).

Runs YOLO crop -> BLIP-2 captions -> CLIP embeddings -> fusion -> HNSW search,
with optional BLIP-ITM reranking. Computes Recall@K, mAP@K, NDCG@K and
exports a qualitative top-5 grid for 3 queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torch.nn.functional as F


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CPROMPT = "Question: Describe this clothing item including color, type, and style. Answer:"


@dataclass
class CatalogRecord:
    item_id: str
    image_path: str
    cropped_path: str
    caption: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluation for visual search")
    parser.add_argument("--gallery-root", required=True, help="Folder with gallery images")
    parser.add_argument("--query-root", required=True, help="Folder with query images")
    parser.add_argument("--work-dir", default="", help="Working directory for caches and outputs")
    parser.add_argument("--catalog-json", default="", help="Optional catalog JSON with captions")
    parser.add_argument("--labels-file", default="", help="Optional labels file for query item_id")
    parser.add_argument("--require-labels", action="store_true", help="Fail if any query label is missing")

    parser.add_argument("--yolo-weights", default="data/yolo_best.pt", help="YOLO weights path")
    parser.add_argument("--disable-crop", action="store_true", help="Disable YOLO cropping")
    parser.add_argument("--crop-pad", type=int, default=10)

    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrain", default="openai")
    parser.add_argument("--clip-ft-weights", default="", help="Fine-tuned CLIP weights checkpoint")

    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--index-backend", choices=["hnsw", "brute"], default="hnsw")
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef", type=int, default=50)
    parser.add_argument("--hnsw-ef-construct", type=int, default=200)

    parser.add_argument("--caption-batch", type=int, default=64)
    parser.add_argument("--embed-batch", type=int, default=128)
    parser.add_argument("--yolo-batch", type=int, default=64)
    parser.add_argument("--caption-max-tokens", type=int, default=20)

    parser.add_argument("--disable-blip2", action="store_true", help="Disable BLIP-2 captions")
    parser.add_argument("--blip2-model", default="Salesforce/blip2-opt-2.7b")

    parser.add_argument("--disable-itm", action="store_true", help="Disable BLIP ITM reranker")
    parser.add_argument("--itm-model", default="Salesforce/blip-itm-base-coco")

    parser.add_argument("--seed", type=int, default=106)
    parser.add_argument("--device", default="", help="cuda or cpu")

    parser.add_argument("--results-json", default="results_metrics.json")
    parser.add_argument("--qual-out", default="qualitative_top5.png")
    parser.add_argument("--qual-num", type=int, default=3)
    parser.add_argument("--qual-top-k", type=int, default=5)
    parser.add_argument("--no-qual", action="store_true")

    return parser.parse_args()


def default_work_dir() -> Path:
    kaggle_dir = Path("/kaggle/working")
    if kaggle_dir.exists():
        return kaggle_dir / "vps_eval"
    return Path("./artifacts")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_images(root: Path) -> List[str]:
    if not root.exists():
        return []
    files = [str(p) for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def infer_item_id(path_str: str) -> str:
    parts = Path(path_str).parts
    for part in parts:
        if part.startswith("id_"):
            return part
    return Path(path_str).parent.name


def load_labels(label_file: str) -> Dict[str, str]:
    if not label_file:
        return {}
    path = Path(label_file)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {label_file}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        items = []
        for entry in data:
            image_name = str(entry.get("image_name", "")) or str(entry.get("image_path", ""))
            item_id = str(entry.get("item_id", ""))
            status = str(entry.get("evaluation_status", "")).lower()
            if status and status != "query":
                continue
            if image_name and item_id:
                items.append((image_name, item_id))
        return build_label_lookup(items)

    items = []
    with path.open("r", encoding="utf-8") as handle:
        lines = [ln.strip() for ln in handle.readlines() if ln.strip()]
    if not lines:
        return {}

    header = lines[0].split()
    has_header = "image_name" in header or "item_id" in header

    for idx, line in enumerate(lines):
        if idx == 0 and has_header:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        image_name = parts[0]
        item_id = parts[1]
        status = parts[2].lower() if len(parts) > 2 else ""
        if status and status != "query":
            continue
        items.append((image_name, item_id))

    return build_label_lookup(items)


def build_label_lookup(items: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for img_name, item_id in items:
        key = img_name.replace("\\", "/")
        lookup[key] = item_id
        lookup[Path(key).name] = item_id
    return lookup


def resolve_query_label(query_path: str, query_root: Path, lookup: Dict[str, str]) -> str:
    if not lookup:
        return ""
    norm = query_path.replace("\\", "/")
    if norm in lookup:
        return lookup[norm]
    try:
        rel = Path(norm).relative_to(query_root)
        key = str(rel).replace("\\", "/")
        if key in lookup:
            return lookup[key]
    except ValueError:
        pass
    base = Path(norm).name
    return lookup.get(base, "")


def crop_cache_path(image_path: str, crop_dir: Path) -> Path:
    h = hashlib.md5(image_path.encode("utf-8")).hexdigest()
    return crop_dir / f"{h}.jpg"


def load_image(path: str, size: Tuple[int, int] | None = None) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
    except OSError:
        img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    if size is None:
        return img
    return ImageOps.pad(img, size, color=(0, 0, 0))


def yolo_crop_batch(model, img_paths: List[str], crop_dir: Path, pad: int, device: str) -> List[str]:
    out_paths = [str(crop_cache_path(p, crop_dir)) for p in img_paths]
    to_process = [(p, out) for p, out in zip(img_paths, out_paths) if not Path(out).exists()]
    if not to_process:
        return out_paths

    if model is None:
        for img_path, out_path in to_process:
            img = load_image(img_path)
            img.save(out_path)
        return out_paths

    results = model.predict([p for p, _ in to_process], device=device, verbose=False)
    for (img_path, out_path), result in zip(to_process, results):
        img = load_image(img_path)
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            best_idx = int(areas.argmax())
            best = boxes[best_idx]
            x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
            width, height = img.size
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(width, x2 + pad)
            y2 = min(height, y2 + pad)
            crop = img.crop((x1, y1, x2, y2))
        else:
            crop = img
        crop.save(out_path)
    return out_paths


def load_yolo(weights: str, device: str):
    if not weights:
        return None
    if not Path(weights).exists():
        return None
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for YOLO cropping") from exc
    return YOLO(weights)


def load_blip2(model_name: str, device: str):
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("transformers is required for BLIP-2 captions") from exc

    processor = Blip2Processor.from_pretrained(model_name)
    if device.startswith("cuda"):
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        ).eval()
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return processor, model


def caption_batch(processor, model, img_paths: List[str], batch_size: int, device: str, max_tokens: int) -> List[str]:
    captions: List[str] = []
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i : i + batch_size]
        images = [load_image(p) for p in batch]
        prompts = [CPROMPT] * len(images)
        inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True)
        if device.startswith("cuda"):
            inputs = inputs.to(device, torch.float16)
        else:
            inputs = inputs.to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, num_beams=1)
        texts = processor.batch_decode(out, skip_special_tokens=True)
        texts = [t.split("Answer:")[-1].strip() if "Answer:" in t else t.strip() for t in texts]
        captions.extend(texts)
    return captions


def load_itm(model_name: str, device: str):
    try:
        from transformers import BlipProcessor, BlipForImageTextRetrieval
    except ImportError as exc:
        raise RuntimeError("transformers is required for BLIP ITM reranking") from exc

    proc = BlipProcessor.from_pretrained(model_name)
    if device.startswith("cuda"):
        model = BlipForImageTextRetrieval.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device).eval()
    else:
        model = BlipForImageTextRetrieval.from_pretrained(model_name).to(device).eval()
    return proc, model


def itm_rerank(proc, model, query_path: str, candidates: List[int], meta: List[CatalogRecord], device: str) -> List[int]:
    if not candidates:
        return candidates

    captions = []
    valid_pos = []
    for pos, idx in enumerate(candidates):
        cap = meta[idx].caption
        if cap:
            captions.append(cap)
            valid_pos.append(pos)

    if not captions:
        return candidates

    qimg = load_image(query_path)
    scores: List[float] = []
    for i in range(0, len(captions), 32):
        chunk = captions[i : i + 32]
        inputs = proc(images=[qimg] * len(chunk), text=chunk, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        if device.startswith("cuda"):
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            out = model(**inputs, use_itm_head=True)
            chunk_scores = F.softmax(out.itm_score, dim=1)[:, 1]
        scores.extend(chunk_scores.detach().cpu().numpy().tolist())

    final_scores = np.zeros(len(candidates), dtype=np.float32)
    for s, pos in zip(scores, valid_pos):
        final_scores[pos] = float(s)
    order = np.argsort(-final_scores, kind="stable")
    return [candidates[i] for i in order]


def load_clip(model_name: str, pretrained: str, device: str, weights_path: str):
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("open-clip-torch is required for CLIP embeddings") from exc

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    if weights_path:
        ckpt = torch.load(weights_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
    model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def encode_images_batched(model, preprocess, img_paths: List[str], device: str, batch_size: int) -> np.ndarray:
    vecs = []
    use_amp = device.startswith("cuda")
    for i in tqdm(range(0, len(img_paths), batch_size), desc="image embed", leave=False):
        batch = img_paths[i : i + batch_size]
        imgs = [preprocess(load_image(p)) for p in batch]
        tb = torch.stack(imgs).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    feat = model.encode_image(tb)
            else:
                feat = model.encode_image(tb)
        vecs.append(F.normalize(feat, dim=-1).cpu().float().numpy())
    return np.concatenate(vecs, axis=0)


def encode_texts_batched(model, tokenizer, captions: List[str], device: str, batch_size: int) -> np.ndarray:
    vecs = []
    use_amp = device.startswith("cuda")
    for i in tqdm(range(0, len(captions), batch_size), desc="text embed", leave=False):
        batch = captions[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    feat = model.encode_text(tokens)
            else:
                feat = model.encode_text(tokens)
        vecs.append(F.normalize(feat, dim=-1).cpu().float().numpy())
    return np.concatenate(vecs, axis=0)


def build_hnsw(vectors: np.ndarray, m: int, ef: int, ef_construct: int):
    import hnswlib
    idx = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    idx.init_index(max_elements=len(vectors), ef_construction=ef_construct, M=m)
    idx.add_items(vectors, np.arange(len(vectors)))
    idx.set_ef(ef)
    return idx


def hit_rate_at_k(ret: List[str], rel: set, k: int) -> int:
    return int(len(set(ret[:k]) & rel) > 0)


def recall_at_k(ret: List[str], rel: set, k: int, nr: int) -> float:
    return sum(1 for r in ret[:k] if r in rel) / max(1, nr)


def ap_at_k(ret: List[str], rel: set, k: int, nr: int) -> float:
    h = 0
    s = 0.0
    for rk, r in enumerate(ret[:k], 1):
        if r in rel:
            h += 1
            s += h / rk
    return s / max(1, min(k, nr))


def ndcg_at_k(ret: List[str], rel: set, k: int, nr: int) -> float:
    d = sum(1 / np.log2(i + 2) for i, r in enumerate(ret[:k]) if r in rel)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, nr)))
    return d / ideal if ideal > 0 else 0.0


def compute_metrics(all_ret: List[List[str]], all_rel: List[set], all_nr: List[int], top_k: List[int]) -> Dict[int, Dict[str, Tuple[float, float]]]:
    out: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for k in top_k:
        hr = [hit_rate_at_k(r, rel, k) for r, rel in zip(all_ret, all_rel)]
        rc = [recall_at_k(r, rel, k, nr) for r, rel, nr in zip(all_ret, all_rel, all_nr)]
        ap = [ap_at_k(r, rel, k, nr) for r, rel, nr in zip(all_ret, all_rel, all_nr)]
        nd = [ndcg_at_k(r, rel, k, nr) for r, rel, nr in zip(all_ret, all_rel, all_nr)]
        out[k] = {
            "HR": (float(np.mean(hr)), float(np.std(hr))),
            "Recall": (float(np.mean(rc)), float(np.std(rc))),
            "mAP": (float(np.mean(ap)), float(np.std(ap))),
            "NDCG": (float(np.mean(nd)), float(np.std(nd))),
        }
    return out


def save_qualitative_grid(
    query_paths: List[str],
    query_crops: List[str],
    results: List[List[int]],
    gallery_meta: List[CatalogRecord],
    out_path: Path,
    num_rows: int,
    top_k: int,
    seed: int,
) -> None:
    if not query_paths or not results:
        return

    rng = random.Random(seed)
    indices = list(range(len(query_paths)))
    rng.shuffle(indices)
    indices = indices[:num_rows]

    cell_size = (224, 224)
    cols = 1 + top_k
    canvas = Image.new("RGB", (cell_size[0] * cols, cell_size[1] * num_rows), color=(10, 10, 10))

    for row, qi in enumerate(indices):
        qimg = load_image(query_crops[qi], size=cell_size)
        qimg = ImageOps.expand(qimg, border=4, fill=(30, 144, 255))
        qimg = ImageOps.pad(qimg, cell_size)
        canvas.paste(qimg, (0, row * cell_size[1]))

        for col in range(top_k):
            if col >= len(results[qi]):
                continue
            idx = results[qi][col]
            gpath = gallery_meta[idx].image_path
            gimg = load_image(gpath, size=cell_size)
            canvas.paste(gimg, ((col + 1) * cell_size[0], row * cell_size[1]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_blip2 = not args.disable_blip2
    use_itm = not args.disable_itm

    if device == "cpu" and use_blip2:
        print("BLIP-2 on CPU is very slow; disabling captions.")
        use_blip2 = False

    work_dir = Path(args.work_dir) if args.work_dir else default_work_dir()
    work_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = work_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = Path(args.catalog_json) if args.catalog_json else (work_dir / "catalog.json")

    set_seed(args.seed)

    gallery_root = Path(args.gallery_root)
    query_root = Path(args.query_root)
    gallery_paths = list_images(gallery_root)
    query_paths = list_images(query_root)

    if not gallery_paths:
        raise SystemExit(f"No gallery images found under {gallery_root}")
    if not query_paths:
        raise SystemExit(f"No query images found under {query_root}")

    label_lookup = load_labels(args.labels_file)

    yolo_model = None
    if not args.disable_crop:
        yolo_model = load_yolo(args.yolo_weights, device)
        if yolo_model is None:
            print("YOLO weights not found; using full images.")

    catalog_map: Dict[str, CatalogRecord] = {}
    if catalog_path.exists():
        try:
            with catalog_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            for entry in raw:
                image_path = str(entry.get("image_path", ""))
                if not image_path:
                    continue
                if not Path(image_path).exists():
                    # try rebasing to gallery root
                    image_path = str(gallery_root / Path(image_path).name)
                if not Path(image_path).exists():
                    continue
                catalog_map[image_path] = CatalogRecord(
                    item_id=str(entry.get("item_id", "")) or infer_item_id(image_path),
                    image_path=image_path,
                    cropped_path=str(entry.get("cropped_path", "")),
                    caption=str(entry.get("caption", "")),
                )
        except (json.JSONDecodeError, OSError):
            catalog_map = {}

    todo_paths: List[str] = []
    for p in gallery_paths:
        rec = catalog_map.get(p)
        need_crop = True
        need_caption = use_blip2
        if rec:
            need_crop = not rec.cropped_path or not Path(rec.cropped_path).exists()
            need_caption = use_blip2 and not rec.caption
        if need_crop or need_caption:
            todo_paths.append(p)

    blip_proc = None
    blip_model = None
    if use_blip2:
        print("Loading BLIP-2 for captions...")
        blip_proc, blip_model = load_blip2(args.blip2_model, device)

    for i in tqdm(range(0, len(todo_paths), args.yolo_batch), desc="catalog build"):
        batch = todo_paths[i : i + args.yolo_batch]
        crops = yolo_crop_batch(yolo_model, batch, crop_dir, args.crop_pad, device)
        if use_blip2:
            captions = caption_batch(blip_proc, blip_model, crops, args.caption_batch, device, args.caption_max_tokens)
        else:
            captions = ["" for _ in crops]

        for orig, crop, cap in zip(batch, crops, captions):
            rec = catalog_map.get(orig)
            if rec is None:
                rec = CatalogRecord(item_id=infer_item_id(orig), image_path=orig, cropped_path=crop, caption="")
            rec.cropped_path = crop
            if use_blip2:
                rec.caption = cap
            if not rec.caption:
                rec.caption = ""
            catalog_map[orig] = rec

    catalog = list(catalog_map.values())
    with catalog_path.open("w", encoding="utf-8") as handle:
        json.dump([rec.__dict__ for rec in catalog], handle, indent=2)

    if blip_model is not None:
        del blip_model
        torch.cuda.empty_cache()

    print(f"Catalog ready: {len(catalog)} items")

    clip_model, clip_preprocess, clip_tokenizer = load_clip(
        args.clip_model, args.clip_pretrain, device, args.clip_ft_weights
    )

    cache_key = hashlib.md5(
        f"{args.clip_model}|{args.clip_pretrain}|{args.clip_ft_weights}|{len(catalog)}".encode("utf-8")
    ).hexdigest()[:8]
    img_cache = work_dir / f"gal_img_{cache_key}.npy"
    txt_cache = work_dir / f"gal_txt_{cache_key}.npy"
    q_cache = work_dir / f"q_img_{cache_key}.npy"

    gallery_crops = [rec.cropped_path for rec in catalog]
    gallery_captions = [rec.caption for rec in catalog]

    if img_cache.exists():
        gal_img = np.load(img_cache)
    else:
        gal_img = encode_images_batched(clip_model, clip_preprocess, gallery_crops, device, args.embed_batch)
        np.save(img_cache, gal_img)

    if any(c for c in gallery_captions):
        if txt_cache.exists():
            gal_txt = np.load(txt_cache)
        else:
            gal_txt = encode_texts_batched(clip_model, clip_tokenizer, gallery_captions, device, args.embed_batch)
            np.save(txt_cache, gal_txt)
    else:
        gal_txt = np.zeros_like(gal_img)

    query_crops: List[str] = []
    for i in tqdm(range(0, len(query_paths), args.yolo_batch), desc="query crops"):
        batch = query_paths[i : i + args.yolo_batch]
        query_crops.extend(yolo_crop_batch(yolo_model, batch, crop_dir, args.crop_pad, device))

    if q_cache.exists():
        q_img = np.load(q_cache)
    else:
        q_img = encode_images_batched(clip_model, clip_preprocess, query_crops, device, args.embed_batch)
        np.save(q_cache, q_img)

    fused = args.alpha * gal_img + (1.0 - args.alpha) * gal_txt
    norms = np.linalg.norm(fused, axis=1, keepdims=True).clip(min=1e-8)
    fused = (fused / norms).astype("float32")

    if args.index_backend == "hnsw":
        try:
            index = build_hnsw(fused, args.hnsw_m, args.hnsw_ef, args.hnsw_ef_construct)
            use_hnsw = True
        except Exception as exc:
            print(f"HNSW failed ({exc}); falling back to brute-force")
            use_hnsw = False
    else:
        use_hnsw = False

    if not use_hnsw:
        fused_t = fused.T

    itm_proc = None
    itm_model = None
    if use_itm:
        print("Loading BLIP ITM reranker...")
        itm_proc, itm_model = load_itm(args.itm_model, device)

    gallery_item_counts: Dict[str, int] = {}
    for rec in catalog:
        gallery_item_counts[rec.item_id] = gallery_item_counts.get(rec.item_id, 0) + 1

    all_ret: List[List[str]] = []
    all_rel: List[set] = []
    all_nr: List[int] = []
    ranked_indices: List[List[int]] = []

    k_search = max(args.top_k, args.qual_top_k)

    for qi in tqdm(range(len(query_paths)), desc="search"):
        qvec = q_img[qi].reshape(1, -1).astype("float32")
        if use_hnsw:
            labels, _ = index.knn_query(qvec, k=k_search)
            candidates = list(labels[0])
        else:
            scores = fused @ qvec.squeeze(0)
            candidates = list(np.argsort(-scores)[:k_search])

        if use_itm and itm_proc is not None and itm_model is not None:
            candidates = itm_rerank(itm_proc, itm_model, query_crops[qi], candidates, catalog, device)

        ranked_indices.append(candidates)
        all_ret.append([catalog[i].item_id for i in candidates])

        label = resolve_query_label(query_paths[qi], query_root, label_lookup)
        if not label:
            label = infer_item_id(query_paths[qi])
        if not label and args.require_labels:
            raise SystemExit(f"Missing label for query: {query_paths[qi]}")

        rel = {label} if label else set()
        all_rel.append(rel)
        all_nr.append(gallery_item_counts.get(label, 1))

    valid_idx = [i for i, rel in enumerate(all_rel) if rel]
    if not valid_idx:
        raise SystemExit("No labeled queries found; cannot compute metrics")

    all_ret_v = [all_ret[i] for i in valid_idx]
    all_rel_v = [all_rel[i] for i in valid_idx]
    all_nr_v = [all_nr[i] for i in valid_idx]

    metrics = compute_metrics(all_ret_v, all_rel_v, all_nr_v, [5, 10, 15])

    print("\nMetrics:")
    for k in [5, 10, 15]:
        hr, hrs = metrics[k]["HR"]
        rc, rcs = metrics[k]["Recall"]
        mp, mps = metrics[k]["mAP"]
        nd, nds = metrics[k]["NDCG"]
        print(f"  K={k:2d} HR={hr:.4f}±{hrs:.4f} Recall={rc:.4f}±{rcs:.4f} mAP={mp:.4f}±{mps:.4f} NDCG={nd:.4f}±{nds:.4f}")

    results_path = Path(args.results_json)
    if not results_path.is_absolute():
        results_path = work_dir / results_path
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": {
                    "gallery_root": str(gallery_root),
                    "query_root": str(query_root),
                    "alpha": args.alpha,
                    "top_k": args.top_k,
                    "clip_model": args.clip_model,
                    "clip_pretrain": args.clip_pretrain,
                    "clip_ft_weights": args.clip_ft_weights,
                    "use_blip2": use_blip2,
                    "use_itm": use_itm,
                    "seed": args.seed,
                },
                "metrics": {
                    str(k): {m: list(v) for m, v in metrics[k].items()} for k in metrics
                },
                "num_queries": len(query_paths),
                "num_labeled": len(valid_idx),
            },
            handle,
            indent=2,
        )

    if not args.no_qual:
        qual_path = Path(args.qual_out)
        if not qual_path.is_absolute():
            qual_path = work_dir / qual_path
        save_qualitative_grid(
            query_paths,
            query_crops,
            ranked_indices,
            catalog,
            qual_path,
            args.qual_num,
            args.qual_top_k,
            args.seed,
        )
        print(f"Qualitative grid saved to {qual_path}")


if __name__ == "__main__":
    main()
