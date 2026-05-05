"""Generate per-layer dimmed copies of pipeline_only.png.

Each output keeps the focused bounding box(es) at full colour and
fades the rest with a semi-transparent white overlay. This lets the
deck show the same architecture diagram on every layer slide while
visually drawing the audience's eye to the layer being explained.

Bounding boxes are eyeballed from the existing 1394 x 1310 image.
If the image is regenerated at a different size, scale these
proportionally (e.g. multiply each (x1, y1, x2, y2) by new_w/1394
and new_h/1310).
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image

REPO = Path(__file__).resolve().parent
SRC = REPO / "figures" / "pipeline_only.png"

# Strength of the dimming overlay. 0.0 = no dim, 1.0 = pure white.
DIM_STRENGTH = 0.62

# Bounding boxes per layer slide. Each entry is a list of (x1, y1, x2, y2)
# rectangles in source-image pixel coordinates that stay full-colour.
# Coordinates are calibrated against pipeline_only.png at 1394 x 1310 and
# were derived by detecting each layer's title-bar fill colour and
# extending downward to cover the lighter body section.
BBOXES: dict[str, list[tuple[int, int, int, int]]] = {
    # Slide 5 - Layer 1: Voxtral STT, Scribe v2 STT, Whisper (1c)
    "layer1": [
        (165, 255, 1335, 415),
    ],
    # Slide 6 - Layer 2: country detection
    "layer2": [
        (425, 385, 770, 540),
    ],
    # Slide 7 - Layer 3: candidate-1 left + candidate-1 right (skip middle)
    "layer3": [
        (75,  570, 425, 760),
        (775, 570, 1120, 760),
    ],
    # Slide 8 - Layer 4: candidate-2 (middle)
    "layer4": [
        (425, 570, 770, 760),
    ],
    # Slide 9 - Layer 4.5: targeted re-listen (bottom left)
    "layer4_5": [
        (60, 915, 450, 1075),
    ],
    # Slide 10 - Layer 5: reconciler (bottom right)
    "layer5": [
        (760, 915, 1120, 1075),
    ],
    # Slide 11 - Layer 6: schema
    "layer6": [
        (420, 1065, 780, 1235),
    ],
}


def _dimmed(base: Image.Image, strength: float) -> Image.Image:
    """Return a copy of `base` blended toward white by `strength`."""
    overlay = Image.new("RGB", base.size, (255, 255, 255))
    return Image.blend(base, overlay, strength)


def render_variant(name: str,
                    bboxes: list[tuple[int, int, int, int]]) -> Path:
    base = Image.open(SRC).convert("RGB")
    out = _dimmed(base, DIM_STRENGTH)
    # Paste the focus regions back at full colour.
    for box in bboxes:
        crop = base.crop(box)
        out.paste(crop, box[:2])
    dest = REPO / "figures" / f"pipeline_{name}.png"
    out.save(dest)
    return dest


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing source image: {SRC}")
    for name, bboxes in BBOXES.items():
        path = render_variant(name, bboxes)
        print(f"  wrote {path.name}  bboxes={len(bboxes)}")


if __name__ == "__main__":
    main()
