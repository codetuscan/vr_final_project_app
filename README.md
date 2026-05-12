# Visual Product Search Demo (Streamlit)

This repository contains a modular Streamlit application for the Visual Product Search Engine project. It follows the problem statement flow: upload an image, run detection and crop, require user confirmation and category selection, then perform retrieval and display top-K results.

## What is implemented
- Streamlit demo app with the required UX gating (no retrieval before confirmation).
- Modular pipeline structure with adapter classes for detector, embedder, captioner, and reranker.
- Local retrieval index with HNSW (hnswlib) and a brute-force fallback.
- Catalog loader that rebases Kaggle-style paths to the local gallery directory.
- Clear warnings when real models or data are not available.

## Why the structure looks like this
- Each model stage is isolated behind an adapter so you can swap YOLO, CLIP, BLIP-2, or a reranker without changing the UI or pipeline code.
- Configuration is centralized in one place, so paths and parameters do not get hardcoded across the codebase.
- The app flow is in a single Streamlit entrypoint, while the retrieval logic lives in a pipeline module to keep UI logic separate from ML logic.

## Project structure
```
project_app/
  streamlit_app.py
  requirements.txt
  README.md
  src/visual_search/
    config.py
    utils.py
    data/
      catalog.py
      gallery.py
    models/
      base.py
      detector.py
      captioner.py
      embedder.py
      reranker.py
    pipelines/
      retrieval.py
    retrieval/
      index.py
```

## Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run streamlit_app.py
```

## Expected data layout
By default the app expects the local DeepFashion-style folders:
```
archive/
  gallery/
    img/
      WOMEN/...
      MEN/...
  query/
    img/
      WOMEN/...
      MEN/...
```

It also looks for `catalog.json` at the repository root. If `catalog.json` is missing or invalid, the app will scan the gallery folder and build a minimal catalog without captions.

## Configuration (environment variables)
You can override paths and parameters via environment variables:
- `VPS_DATA_ROOT` (default: `archive`)
- `VPS_GALLERY_ROOT` (default: `archive/gallery/img`)
- `VPS_QUERY_ROOT` (default: `archive/query/img`)
- `VPS_CATALOG_PATH` (default: `catalog.json`)
- `VPS_TOP_K` (default: `15`)
- `VPS_ALPHA` (default: `0.5`)
- `VPS_INDEX_BACKEND` (`hnsw` or `brute-force`)
- `VPS_USE_MOCK_MODELS` (`true` or `false`)
- `VPS_YOLO_WEIGHTS` (path to YOLO weights)
- `VPS_CLIP_MODEL` (default: `ViT-B-32`)
- `VPS_CLIP_PRETRAIN` (default: `openai`)
- `VPS_DEVICE` (default: `cpu`)

## Where to plug in real models later
The app currently uses mock adapters by default to keep the demo working end-to-end while models are still being trained. Replace or extend these files when your models are ready:
- `src/visual_search/models/detector.py` for YOLO detection
- `src/visual_search/models/embedder.py` for CLIP image/text embeddings
- `src/visual_search/models/captioner.py` for BLIP-2 captions
- `src/visual_search/models/reranker.py` for BLIP-2 ITM reranking

If you enable real models, set `VPS_USE_MOCK_MODELS=false` and point to the model weights. The pipeline will try to use the real adapters and fall back to mock ones if dependencies or weights are missing.
For fine-tuned CLIP weights, extend `ClipEmbedder` to load from `VPS_CLIP_WEIGHTS`.

Optional dependencies for real models (install only when needed):
- `ultralytics` for YOLO detection
- `torch` and `open-clip-torch` for CLIP embeddings
- `transformers` for BLIP-2 captioning or ITM reranking

If `hnswlib` is not available on your system, set `VPS_INDEX_BACKEND=brute-force` to use a pure numpy search backend.

## Input / output flow
1. Upload a query image.
2. The app runs detection and shows the proposed crop.
3. The user confirms the crop and selects the category (top wear, bottom wear, or full body).
4. The app runs retrieval and displays top-K results with similarity scores and metadata.

## Known limitations (current version)
- Mock adapters are used by default, so retrieval quality is not representative.
- BLIP-2 captioning and ITM reranking are placeholders.
- HNSW index is built in memory on startup and not persisted to disk.
- Category selection is used for confirmation flow, not as a retrieval filter yet.

## Notes for evaluation/demo
- The confirmation gate is enforced: retrieval does not start until the user confirms the crop.
- The demo is structured to match the problem statement requirements for the Streamlit application.
# vr_final_project_app
