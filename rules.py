"""Phase 1 image-matching rules: metadata (30), histogram (30), template (40).

Each rule returns its points and a short explanation. Image reads and
template scores are cached because each suspect is compared to 10 originals.
"""

import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image


RULES = ("rule1_metadata", "rule2_histogram", "rule3_template")
HISTOGRAM_THRESHOLD = 0.65
TEMPLATE_THRESHOLD = 0.80


@lru_cache(maxsize=256)
def _size(path):
    with Image.open(path) as image:
        return image.size


@lru_cache(maxsize=256)
def _signature(path):
    with open(path, "rb") as image:
        header = image.read(8)
    if header.startswith(b"\xff\xd8"):
        return "JPEG"
    if header == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    return "unknown"


@lru_cache(maxsize=256)
def _hist(path):
    image = cv2.imread(path)
    if image is None:
        return None
    channels = [cv2.calcHist([image], [i], None, [32], [0, 256]) for i in range(3)]
    histogram = np.concatenate(channels).ravel().astype(np.float32)
    cv2.normalize(histogram, histogram)
    return histogram


@lru_cache(maxsize=16)
def _gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


@lru_cache(maxsize=256)
def _template_score(source_path, input_path):
    source, suspect = _gray(source_path), _gray(input_path)
    if source is None or suspect is None:
        return 0.0
    if suspect.shape[0] > source.shape[0] or suspect.shape[1] > source.shape[1]:
        return 0.0

    # Use the SAME scale for both images so a crop still fits its original.
    scale = min(1.0, 512 / max(source.shape))
    if scale < 1.0:
        source = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        suspect = cv2.resize(suspect, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if min(suspect.shape) < 8 or source.std() < 1 or suspect.std() < 1:
        return 0.0  # Normalized correlation is unreliable on constant images.
    result = cv2.matchTemplate(source, suspect, cv2.TM_CCOEFF_NORMED)
    score = float(result.max())
    return float(np.clip(score, 0, 1)) if np.isfinite(score) else 0.0


def rule1_metadata(target, input_path):
    """Check dimensions, file sizes, and actual JPEG/PNG headers."""
    out = {"rule": 1, "name": "Metadata", "fired": False, "score": 0,
           "out_of": 30, "note": "Unreadable metadata", "metric": 0.0}
    try:
        source_path = target["path"]
        sw, sh = _size(source_path)
        iw, ih = _size(input_path)
        source_bytes, input_bytes = os.path.getsize(source_path), os.path.getsize(input_path)
        ratio = min(source_bytes, input_bytes) / max(source_bytes, input_bytes)
        source_format, input_format = _signature(source_path), _signature(input_path)
        compatible = iw <= sw and ih <= sh and abs((iw / ih) / (sw / sh) - 1) <= 0.02
        out["metric"] = round(ratio, 3)
        out["note"] = (f"{source_format} to {input_format}; size ratio {ratio:.2f}; "
                       f"dimensions {iw}x{ih} vs {sw}x{sh}")

        # Metadata is not unique: require Rule 3's visual evidence as support.
        if (compatible and "unknown" not in (source_format, input_format)
                and _template_score(source_path, input_path) >= TEMPLATE_THRESHOLD):
            out["fired"] = True
            out["score"] = 20 + round(10 * ratio)
            out["note"] += "; template confirmed"
        else:
            out["note"] += "; dimensions, header, or template check failed"
    except (OSError, ValueError, ZeroDivisionError, cv2.error):
        out["note"] = "Unable to read metadata or supporting template"
    return out


def rule2_histogram(target, input_path):
    """Compare color distributions; award points for strong correlation."""
    out = {"rule": 2, "name": "Histogram", "fired": False, "score": 0,
           "out_of": 30, "note": "Unreadable histogram", "metric": 0.0}
    try:
        source, suspect = _hist(target["path"]), _hist(input_path)
        if source is None or suspect is None:
            return out
        correlation = float(cv2.compareHist(source, suspect, cv2.HISTCMP_CORREL))
        if not np.isfinite(correlation):
            return out
        out["metric"] = round(float(np.clip(correlation, 0, 1)), 3)
        out["note"] = f"Correlation {out['metric']:.3f}"
        if correlation >= HISTOGRAM_THRESHOLD:
            out["fired"] = True
            out["score"] = round(30 * out["metric"])
    except (OSError, ValueError, cv2.error):
        out["note"] = "Unable to compare histograms"
    return out


def rule3_template(target, input_path):
    """Find the suspect inside the original at a shared scale."""
    out = {"rule": 3, "name": "Template", "fired": False, "score": 0,
           "out_of": 40, "note": "Unreadable template", "metric": 0.0}
    try:
        score = _template_score(target["path"], input_path)
        out["metric"] = round(score, 3)
        out["note"] = f"Match score {score:.3f}"
        if score >= TEMPLATE_THRESHOLD:
            out["fired"] = True
            out["score"] = round(40 * out["metric"])
    except (OSError, ValueError, cv2.error):
        out["note"] = "Unable to compare templates"
    return out
