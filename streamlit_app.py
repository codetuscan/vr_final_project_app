"""Streamlit demo for the visual product search project."""

from __future__ import annotations

import base64
import hashlib
import io
import sys
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "src"))

from visual_search.config import get_config
from visual_search.data.catalog import load_catalog
from visual_search.data.gallery import scan_gallery
from visual_search.models.detector import MockDetector, YoloDetector
from visual_search.models.embedder import MockEmbedder, ClipEmbedder
from visual_search.models.reranker import NoOpReranker
from visual_search.pipelines.retrieval import RetrievalPipeline
from visual_search.retrieval.index import HnswConfig, HnswIndex, BruteForceIndex
from visual_search.utils import ensure_rgb

CATEGORIES = ["Top wear", "Bottom wear", "Full body"]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.main > div { padding: 1rem 2.5rem 3rem; max-width: 1200px; margin: 0 auto; }

/* hide default header/footer */
header[data-testid="stHeader"] { background: rgba(15,15,26,.6); backdrop-filter: blur(12px); }
footer { display: none !important; }

/* ── Typography ── */
h1 { font-weight: 800 !important; font-size: 2.4rem !important; background: linear-gradient(135deg,#8B5CF6,#EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px; }
h2 { font-weight: 700 !important; font-size: 1.35rem !important; color: #C4B5FD !important; border: none !important; margin-top: 0.5rem !important; }
h3 { font-weight: 600 !important; font-size: 1.1rem !important; color: #A78BFA !important; }

/* ── Glass card ── */
.glass { background: rgba(255,255,255,0.04); border: 1px solid rgba(139,92,246,0.15); border-radius: 16px; padding: 1.25rem 1.5rem; backdrop-filter: blur(10px); margin-bottom: 1rem; }
.glass-highlight { background: linear-gradient(135deg,rgba(139,92,246,0.08),rgba(236,72,153,0.06)); border: 1px solid rgba(139,92,246,0.25); }

/* ── Step indicator ── */
.step-row { display:flex; gap:0.5rem; margin:1.2rem 0; }
.step { display:flex; align-items:center; gap:0.4rem; padding:0.4rem 1rem; border-radius:999px; font-size:0.8rem; font-weight:600; transition: all .3s; }
.step-active { background:rgba(139,92,246,0.2); color:#C4B5FD; border:1px solid rgba(139,92,246,0.4); }
.step-done { background:rgba(16,185,129,0.15); color:#6EE7B7; border:1px solid rgba(16,185,129,0.3); }
.step-pending { background:rgba(255,255,255,0.04); color:#64748B; border:1px solid rgba(255,255,255,0.06); }

/* ── Stat badges ── */
.stat-row { display:flex; gap:0.75rem; flex-wrap:wrap; margin:0.75rem 0 1rem; }
.stat { display:inline-flex; align-items:center; gap:0.35rem; padding:0.3rem 0.85rem; border-radius:999px; font-size:0.78rem; font-weight:600; }
.stat-purple { background:rgba(139,92,246,0.15); color:#C4B5FD; }
.stat-slate { background:rgba(255,255,255,0.06); color:#94A3B8; }
.stat-green { background:rgba(16,185,129,0.12); color:#6EE7B7; }

/* ── Upload area ── */
[data-testid="stFileUploader"] { background: rgba(139,92,246,0.04); border: 2px dashed rgba(139,92,246,0.25); border-radius: 16px; padding: 1.2rem; transition: all .3s; }
[data-testid="stFileUploader"]:hover { border-color: #8B5CF6; background: rgba(139,92,246,0.08); }

/* ── Buttons ── */
.stButton > button { border-radius: 12px !important; font-weight: 600 !important; padding: 0.5rem 1.5rem !important; border: 1px solid rgba(139,92,246,0.3) !important; background: rgba(139,92,246,0.08) !important; color: #C4B5FD !important; transition: all .25s !important; letter-spacing: 0.3px; }
.stButton > button:hover { background: rgba(139,92,246,0.2) !important; border-color: #8B5CF6 !important; transform: translateY(-1px); box-shadow: 0 4px 20px rgba(139,92,246,0.2); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg,#8B5CF6,#7C3AED) !important; color: #fff !important; border: none !important; }
.stButton > button[kind="primary"]:hover { background: linear-gradient(135deg,#7C3AED,#6D28D9) !important; box-shadow: 0 6px 24px rgba(139,92,246,0.35); }

/* ── Radio ── */
[data-testid="stRadio"] label { font-weight: 500; }
div[role="radiogroup"] > label { background: rgba(255,255,255,0.03); border: 1px solid rgba(139,92,246,0.12); border-radius: 10px; padding: 0.45rem 1rem !important; margin-right: 0.4rem; transition: all .2s; }
div[role="radiogroup"] > label:hover { border-color: rgba(139,92,246,0.4); background: rgba(139,92,246,0.06); }
div[role="radiogroup"] > label[data-checked="true"] { background: rgba(139,92,246,0.15) !important; border-color: #8B5CF6 !important; }

/* ── Preview images ── */
.preview-grid { display:flex; gap:1.25rem; margin:1rem 0; }
.preview-card { flex:1; background:rgba(255,255,255,0.03); border:1px solid rgba(139,92,246,0.12); border-radius:16px; overflow:hidden; transition: all .3s; }
.preview-card:hover { border-color:rgba(139,92,246,0.35); box-shadow:0 8px 30px rgba(139,92,246,0.1); transform:translateY(-3px); }
.preview-card img { width:100%; height:300px; object-fit:contain; background:rgba(0,0,0,0.25); display:block; }
.preview-label { padding:0.6rem 1rem; text-align:center; font-size:0.82rem; font-weight:600; color:#94A3B8; border-top:1px solid rgba(139,92,246,0.1); background:rgba(139,92,246,0.03); letter-spacing:0.3px; }

/* ── Result cards ── */
.result-card { background:rgba(255,255,255,0.03); border:1px solid rgba(139,92,246,0.1); border-radius:16px; overflow:hidden; transition: all .35s; margin-bottom:1rem; }
.result-card:hover { border-color:rgba(139,92,246,0.35); box-shadow:0 12px 40px rgba(139,92,246,0.12); transform:translateY(-4px); }
.result-card img { width:100%; height:260px; object-fit:cover; display:block; }
.result-body { padding:0.85rem 1rem 1rem; }
.result-rank { display:inline-block; padding:0.15rem 0.55rem; border-radius:6px; font-size:0.72rem; font-weight:700; background:linear-gradient(135deg,rgba(139,92,246,0.2),rgba(236,72,153,0.15)); color:#C4B5FD; margin-right:0.4rem; }
.result-id { font-weight:600; color:#E2E8F0; font-size:0.9rem; }
.result-meta { color:#64748B; font-size:0.8rem; margin-top:0.3rem; line-height:1.5; }
.result-score { display:inline-block; background:rgba(16,185,129,0.12); color:#6EE7B7; padding:0.1rem 0.5rem; border-radius:4px; font-size:0.72rem; font-weight:600; }
.result-caption { color:#94A3B8; font-size:0.78rem; font-style:italic; margin-top:0.35rem; }

/* ── Divider ── */
.divider { margin:1.5rem 0; border:0; height:1px; background:linear-gradient(to right,transparent,rgba(139,92,246,0.2),transparent); }

/* ── Alerts ── */
.stAlert { border-radius: 12px !important; }

/* ── Spinner ── */
.stSpinner > div { border-color: #8B5CF6 !important; }

/* ── Responsive ── */
@media (max-width:768px) {
  .preview-grid { flex-direction:column; }
  .preview-card img { height:200px; }
  .result-card img { height:200px; }
  .step-row { flex-wrap:wrap; }
  .main > div { padding: 0.5rem 1rem 2rem; }
}

/* ── Animations ── */
@keyframes fadeUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
.animate-in { animation: fadeUp 0.5s ease-out; }
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode()


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return _b64(buf.getvalue())


def _reset_flow_state() -> None:
    for key in ("crop_image", "crop_bbox", "crop_source", "crop_results", "selected_crop_idx", "last_threshold", "confirmed", "results", "selected_category", "crop_class"):
        st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def _load_gallery_records(catalog_path: str, gallery_root: str, allow_missing: bool) -> list:
    items = load_catalog(Path(catalog_path), Path(gallery_root), allow_missing_images=allow_missing)
    return items if items else scan_gallery(Path(gallery_root))


@st.cache_resource(show_spinner=False)
def _load_detector(sig: Tuple) -> object:
    use_mock, yolo_weights, device, pad, _ = sig
    if not use_mock and yolo_weights:
        try:
            return YoloDetector(yolo_weights, device=device, pad=pad)
        except RuntimeError:
            return MockDetector(pad=pad)
    return MockDetector(pad=pad)


@st.cache_resource(show_spinner=False)
def _load_embedder(sig: Tuple) -> object:
    use_mock, dim, clip_model, clip_pretrain, device = sig
    if not use_mock:
        try:
            return ClipEmbedder(model_name=clip_model, pretrained=clip_pretrain, device=device)
        except RuntimeError:
            return MockEmbedder(dim=dim)
    return MockEmbedder(dim=dim)


@st.cache_resource(show_spinner=False)
def _load_reranker() -> object:
    return NoOpReranker()


def _build_pipeline(config, records):
    embedder = _load_embedder((config.use_mock, config.embedding_dim, config.clip_model, config.clip_pretrain, config.device))
    reranker = _load_reranker()
    if config.index_backend == "hnsw":
        try:
            index = HnswIndex(dim=embedder.dim, config=HnswConfig(m=config.hnsw_m, ef=config.hnsw_ef))
        except RuntimeError:
            index = BruteForceIndex()
    else:
        index = BruteForceIndex()
    pipeline = RetrievalPipeline(embedder, index, reranker, alpha=config.alpha)
    pipeline.build_index(records)
    return pipeline


def _step_html(steps):
    """Render step indicators."""
    parts = []
    for label, status in steps:
        parts.append(f'<div class="step step-{status}">{label}</div>')
    return f'<div class="step-row">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Visual Product Search", page_icon="🔍", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

config = get_config()
detector = _load_detector((config.use_mock, config.yolo_weights, config.device, config.crop_pad, "v2"))

# ── Determine current step ──
has_upload = "original_image" in st.session_state
has_crop = "crop_image" in st.session_state and st.session_state["crop_image"] is not None
is_confirmed = st.session_state.get("confirmed", False)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="glass glass-highlight animate-in" style="text-align:center;padding:2rem 1.5rem 1.5rem;">
    <h1 style="margin:0 0 0.3rem;">🔍 Visual Product Search</h1>
    <p style="color:#94A3B8;font-size:0.95rem;margin:0;max-width:600px;margin:0 auto;">
        Upload a clothing image, review the detected crop, select a category, and discover visually similar products from the gallery.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Step indicators ──
steps = [
    ("① Upload", "done" if has_upload else "active"),
    ("② Detect & Crop", "done" if has_crop else ("active" if has_upload else "pending")),
    ("③ Category", "done" if is_confirmed else ("active" if has_crop else "pending")),
    ("④ Results", "active" if is_confirmed else "pending"),
]
st.markdown(_step_html(steps), unsafe_allow_html=True)

# ── Stat badges ──
if not Path(config.gallery_root).exists():
    st.error(f"Gallery folder not found: `{config.gallery_root}`")
    st.stop()

records = _load_gallery_records(str(config.catalog_path), str(config.gallery_root), config.allow_missing_images)
if not records:
    st.error("No gallery records found. Check catalog.json and gallery paths.")
    st.stop()

cat_name = Path(config.catalog_path).name if Path(config.catalog_path).exists() else "scan"
st.markdown(f"""
<div class="stat-row">
    <span class="stat stat-purple">📦 {len(records):,} gallery items</span>
    <span class="stat stat-slate">📁 {cat_name}</span>
    <span class="stat stat-green">⚡ {config.index_backend.upper()}</span>
</div>
""", unsafe_allow_html=True)

# ── Warnings ──
if config.use_mock:
    st.warning("Using **mock models**. Configure real YOLO / CLIP / BLIP-2 paths for the full pipeline.")
elif getattr(detector, "is_mock", False):
    st.warning("YOLO weights unavailable — using mock detector.")

# ---------------------------------------------------------------------------
# Section 1 — Upload
# ---------------------------------------------------------------------------
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("## ① Upload Query Image")

uploaded_file = st.file_uploader("Choose an image (JPG / PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
if uploaded_file is None:
    st.markdown("""
    <div class="glass" style="text-align:center;padding:2.5rem;">
        <p style="font-size:2.5rem;margin:0;">📤</p>
        <p style="color:#94A3B8;margin:0.5rem 0 0;">Drag & drop or click above to upload a query image</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

file_bytes = uploaded_file.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()

if st.session_state.get("upload_hash") != file_hash:
    st.session_state["upload_hash"] = file_hash
    try:
        st.session_state["original_image"] = ensure_rgb(Image.open(io.BytesIO(file_bytes)))
    except OSError:
        st.error("Invalid image. Please upload a JPG or PNG file.")
        st.stop()
    _reset_flow_state()

original_image = st.session_state.get("original_image")
if original_image is None:
    st.stop()

# ---------------------------------------------------------------------------
# Section 2 — Detection & Preview
# ---------------------------------------------------------------------------
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("## ② Detection & Preview")

conf_threshold = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

if "crop_results" not in st.session_state or st.session_state.get("last_threshold") != conf_threshold:
    with st.spinner("Running detection…"):
        results = detector.detect_and_crop(original_image, threshold=conf_threshold)
        st.session_state["crop_results"] = results
        st.session_state["selected_crop_idx"] = 0
        st.session_state["last_threshold"] = conf_threshold

crop_results = st.session_state.get("crop_results", [])
orig_b64 = _img_to_b64(original_image)

if crop_results:
    st.markdown("Select a detected crop to proceed:")
    
    crop_idx = st.radio(
        "Select Crop", 
        range(len(crop_results)), 
        format_func=lambda i: f"Crop {i+1} - {crop_results[i].class_name or 'unknown'} (Conf: {crop_results[i].confidence:.2f})", 
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state["selected_crop_idx"] = crop_idx
    
    selected_result = crop_results[crop_idx]
    st.session_state["crop_image"] = selected_result.crop
    st.session_state["crop_bbox"] = selected_result.bbox
    st.session_state["crop_source"] = selected_result.source
    st.session_state["crop_class"] = selected_result.class_name

    class_name_to_cat = {
        "clothes_type_1": "Top wear",
        "clothes_type_2": "Bottom wear",
        "clothes_type_3": "Full body"
    }
    auto_cat = class_name_to_cat.get(selected_result.class_name)
    if auto_cat and st.session_state.get("selected_category") != auto_cat:
        st.session_state["selected_category"] = auto_cat

    crop_image = selected_result.crop
    crop_b64 = _img_to_b64(crop_image)
    st.markdown(f"""
    <div class="preview-grid animate-in">
        <div class="preview-card">
            <img src="data:image/png;base64,{orig_b64}" alt="Original">
            <div class="preview-label">📷 Original Upload</div>
        </div>
        <div class="preview-card">
            <img src="data:image/png;base64,{crop_b64}" alt="Crop">
            <div class="preview-label">✂️ Detected Crop</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    bbox = st.session_state.get("crop_bbox", "")
    src = st.session_state.get("crop_source", "unknown")
    st.caption(f"Crop source: **{src}** · BBox: `{bbox}`")
else:
    st.session_state.pop("crop_image", None)
    st.markdown(f"""
    <div class="preview-grid animate-in">
        <div class="preview-card">
            <img src="data:image/png;base64,{orig_b64}" alt="Original">
            <div class="preview-label">📷 Original Upload (no crops above threshold)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Action buttons
c1, c2, _ = st.columns([1, 1, 3])
with c1:
    if st.button("🔄 Re-detect"):
        with st.spinner("Re-running…"):
            results = detector.detect_and_crop(original_image, threshold=conf_threshold)
            st.session_state["crop_results"] = results
            st.session_state["selected_crop_idx"] = 0
            st.session_state["confirmed"] = False
            st.session_state["results"] = None
        st.rerun()
with c2:
    if st.button("🗑️ Reset"):
        _reset_flow_state()
        st.rerun()

# ---------------------------------------------------------------------------
# Section 3 — Category & Confirm
# ---------------------------------------------------------------------------
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("## ③ Select Category & Search")

prev_cat = st.session_state.get("selected_category")
selected_category = st.radio("Clothing region / category", CATEGORIES, key="selected_category", horizontal=True)
if prev_cat and prev_cat != selected_category:
    st.session_state["confirmed"] = False
    st.session_state["results"] = None

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

bc, ic = st.columns([1, 2])
with bc:
    if st.button("🔍 Confirm & Search", type="primary", use_container_width=True):
        st.session_state["confirmed"] = True
with ic:
    st.info("Retrieval runs only after you confirm the crop and category.")

# ---------------------------------------------------------------------------
# Section 4 — Results
# ---------------------------------------------------------------------------
if not st.session_state.get("confirmed"):
    st.stop()

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("## ④ Search Results")

if "pipeline" not in st.session_state:
    with st.spinner("Building gallery index…"):
        st.session_state["pipeline"] = _build_pipeline(config, records)

pipeline: RetrievalPipeline = st.session_state["pipeline"]
if not config.use_mock and getattr(pipeline.embedder, "is_mock", False):
    st.warning("CLIP dependencies unavailable — using mock embedder.")

if "results" not in st.session_state or st.session_state.get("results") is None:
    with st.spinner("Searching for similar products…"):
        st.session_state["results"] = pipeline.search(crop_image, config.top_k)
        st.session_state["missing_images"] = pipeline.missing_images

results = st.session_state.get("results") or []

# Results header
st.markdown(f"""
<div class="glass glass-highlight animate-in" style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;">
    <span style="font-size:1.1rem;font-weight:700;color:#E2E8F0;">🎯 Top-{config.top_k} Results</span>
    <div style="display:flex;gap:0.5rem;">
        <span class="stat stat-purple">Category: {selected_category}</span>
        <span class="stat stat-slate">Index: {pipeline.index.name}</span>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.get("missing_images"):
    st.warning(f"{st.session_state['missing_images']} gallery images were missing and replaced with blanks.")

if not results:
    st.warning("No results returned. Try another query image.")
else:
    cols = st.columns(3)
    for idx, r in enumerate(results):
        with cols[idx % 3]:
            try:
                img = Image.open(r.image_path)
            except OSError:
                img = Image.new("RGB", (224, 224), color=(30, 30, 40))
            ib64 = _img_to_b64(img)
            caption_html = f'<div class="result-caption">{r.caption}</div>' if r.caption else ""
            st.markdown(f"""
            <div class="result-card animate-in">
                <img src="data:image/png;base64,{ib64}" alt="Result {r.rank}">
                <div class="result-body">
                    <div><span class="result-rank">#{r.rank}</span><span class="result-id">{r.item_id}</span></div>
                    <div class="result-meta">
                        <span class="result-score">Score: {r.score:.3f}</span> · Category: {r.category or 'n/a'}
                    </div>
                    {caption_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
