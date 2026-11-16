"""ShopTheLook demo (Gradio).

Upload an outfit photo -> detected garments (numbered boxes) -> top-5 similar
catalog products per garment, with cosine-similarity scores.

Run:  python app/app.py        (or: bash run_demo.sh)
"""
from __future__ import annotations

import base64
import html
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from src.pipeline import ShopTheLookPipeline, draw_detections  # noqa: E402
from src.viz import SERIES  # noqa: E402

EMBEDDER = os.environ.get("SHOPTHELOOK_EMBEDDER", "fashionclip")
GROUP_LABEL = {"top": "Tops", "outerwear": "Jackets & outerwear", "shorts": "Shorts", "trousers": "Trousers & jeans",
               "skirt": "Skirts", "dress": "Dresses"}

print(f"Loading pipeline (embedder={EMBEDDER}) ...")
PIPE = ShopTheLookPipeline.load(EMBEDDER)
_THUMBS: dict[str, str] = {}


def thumb(path: str, size: int = 180) -> str:
    """Product image as a small base64 data URI (cached)."""
    if path not in _THUMBS:
        im = Image.open(path).convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        _THUMBS[path] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return _THUMBS[path]


CSS = """
.stl-item {margin: 0 0 22px 0;}
.stl-head {display:flex; align-items:center; gap:10px; margin-bottom:8px; font-size:15px;}
.stl-badge {color:#fff; border-radius:6px; padding:2px 9px; font-weight:600;}
.stl-sub {color:#6b6a65; font-size:13px;}
.stl-row {display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:12px;}
.stl-card {border:1px solid #e4e3df; border-radius:10px; padding:8px; background:#fff;}
.stl-card img {width:100%; height:170px; object-fit:contain; background:#fafaf8; border-radius:6px;}
.stl-name {font-size:12.5px; line-height:1.3; margin-top:6px; height:33px; overflow:hidden; color:#1d1d1b;}
.stl-meta {display:flex; justify-content:space-between; font-size:12px; color:#6b6a65; margin-top:4px;}
.stl-score {font-variant-numeric: tabular-nums; font-weight:600; color:#1d1d1b;}
.stl-warn {color:#9a5b00; font-size:12.5px; margin:-2px 0 8px 0;}
"""


def render(result) -> str:
    if not result.items:
        return "<p>No garments detected. Try a clearer, full-body outfit photo.</p>"
    parts = []
    for i, item in enumerate(result.items):
        d = item.detection
        color = SERIES[i % len(SERIES)]
        cards = "".join(
            f"""<div class="stl-card"><img src="{thumb(m.image_path)}" alt="">
            <div class="stl-name">{html.escape(m.name)}</div>
            <div class="stl-meta"><span>{html.escape(m.colour)} · {html.escape(m.article_type)}</span>
            <span class="stl-score" title="cosine similarity">{m.score:.3f}</span></div></div>"""
            for m in item.matches)
        cross = [m for m in item.matches if m.group != d.group]
        warn = (f'<div class="stl-warn">Category filter off: {len(cross)} of {len(item.matches)} results '
                f'are not {GROUP_LABEL[d.group].lower()}.</div>') if cross else ""
        parts.append(
            f"""<div class="stl-item"><div class="stl-head">
            <span class="stl-badge" style="background:{color}">{i + 1}</span>
            <b>{d.category.replace('_', ' ').capitalize()}</b>
            <span class="stl-sub">detector confidence {d.confidence:.2f} · searching {GROUP_LABEL[d.group]}</span>
            </div>{warn}<div class="stl-row">{cards}</div></div>""")
    return "".join(parts)


def shop(image: Image.Image | None, filter_on: bool):
    if image is None:
        return None, "<p>Upload an outfit photo to start.</p>", ""
    result = PIPE.run(image, k=5, filter_by_category=filter_on)
    annotated = draw_detections(image, [it.detection for it in result.items])
    t = result.timings_ms
    timing = (f"{len(result.items)} garment(s) · {t['total']:.0f} ms total "
              f"(detect {t['detect']:.0f} · embed {t['embed']:.0f} · search {t['search']:.1f})")
    return annotated, render(result), timing


examples_dir = ROOT / "app" / "examples"
examples = sorted(str(p) for p in examples_dir.glob("*.jpg")) if examples_dir.exists() else []

with gr.Blocks(title="ShopTheLook") as demo:
    gr.Markdown("## ShopTheLook\nUpload an outfit photo. We detect each garment and show the most similar "
                "products from the catalog. *Research prototype: catalog = Kaggle Fashion Product Images.*")
    with gr.Row():
        with gr.Column(scale=4):
            inp = gr.Image(type="pil", label="Outfit photo", height=420)
            filt = gr.Checkbox(value=True, label="Only show products of the detected category")
            btn = gr.Button("Shop the look", variant="primary")
            if examples:
                gr.Examples(examples=examples, inputs=inp, label="Example photos (DeepFashion2, research use)")
        with gr.Column(scale=5):
            out_img = gr.Image(label="Detected garments", height=420, interactive=False)
            timing = gr.Markdown()
    results = gr.HTML()
    btn.click(shop, [inp, filt], [out_img, results, timing])
    inp.upload(shop, [inp, filt], [out_img, results, timing])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)), css=CSS)
