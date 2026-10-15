"""V2 keeps Rules 1-3 and adds multi-scale matching for resized content.

Weights are 15 + 15 + 20 + 50 = 100. A strong multi-scale match receives
50 points, enough to match even when resizing defeats the original rules.
"""

from functools import lru_cache

import cv2
import numpy as np

import rules


def _reweight(evidence, weight):
    evidence["score"] = round(evidence["score"] * weight / evidence["out_of"])
    evidence["out_of"] = weight
    return evidence


def rule1_metadata(target, input_path):
    return _reweight(rules.rule1_metadata(target, input_path), 15)


def rule2_histogram(target, input_path):
    return _reweight(rules.rule2_histogram(target, input_path), 15)


def rule3_template(target, input_path):
    return _reweight(rules.rule3_template(target, input_path), 20)


@lru_cache(maxsize=128)
def _small_gray(path):
    image = rules._gray(path)
    if image is None:
        return None
    scale = min(1.0, 320 / max(image.shape))
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def rule4_multiscale(target, input_path):
    """Try several content scales; no rotation search or learned model."""
    out = {"rule": 4, "name": "Multiscale", "fired": False, "score": 0,
           "out_of": 50, "note": "Unreadable images", "metric": 0.0}
    try:
        # Keep strong V1 matches, including brightness edits with changed colors.
        same_scale = rules._template_score(target["path"], input_path)
        if same_scale >= rules.TEMPLATE_THRESHOLD:
            out.update(fired=True, score=50, metric=round(same_scale, 3),
                       note=f"Same-scale match {same_scale:.3f}; no scale search needed")
            return out
        # A broad scale search can match unrelated gradients. Require some
        # color agreement before trusting a newly discovered spatial match.
        if rules.rule2_histogram(target, input_path)["metric"] < 0.30:
            out["note"] = "Insufficient color support for scale search (below 0.30)"
            return out
        source, suspect = _small_gray(target["path"]), _small_gray(input_path)
        if source is None or suspect is None:
            return out
        best, best_scale, direction = 0.0, 1.0, "suspect inside original"
        # Search both directions: a resized original may be inside the suspect.
        for image, template, label in ((source, suspect, "suspect inside original"),
                                       (suspect, source, "original inside suspect")):
            if image.std() < 1 or template.std() < 1:
                continue
            for scale in np.linspace(0.25, 1.0, 31):
                resized = cv2.resize(template, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
                h, w = resized.shape
                if min(h, w) < 32 or h > image.shape[0] or w > image.shape[1]:
                    continue
                if resized.std() < 1:
                    continue
                score = float(cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED).max())
                if np.isfinite(score) and score > best:
                    best, best_scale, direction = score, scale, label
        out["metric"] = round(best, 3)
        out["note"] = (f"Match score {best:.3f}; thumbnail scale {best_scale:.3f}; "
                       f"{direction}; threshold 0.85")
        if best >= 0.85:
            out["fired"] = True
            out["score"] = 50
    except (OSError, ValueError, cv2.error):
        out["note"] = "Unable to compare multi-scale templates"
    return out


# Callables select these V2 wrappers rather than the V1 functions of the same name.
RULES = (rule1_metadata, rule2_histogram, rule3_template, rule4_multiscale)
