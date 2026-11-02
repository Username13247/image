"""
CENG 466 - THE1 - Question 2: Underwater image enhancement (UIEB dataset)

Student 1: Name Surname, ID
Student 2: Name Surname, ID

Usage:
    python q2.py --input <folder of raw underwater images> --output <output folder>
Examples:
    python q2.py --input data/uieb/raw --output outputs/q2          (the given images)
    python q2.py --input test/uieb/raw --output test_outputs/q2     (the test images, after the deadline)
Outputs keep the input file names and sizes.

Develop, tune and report on the given pairs (data/uieb/raw, data/uieb/reference):
    python evaluate.py outputs/q2 data/uieb/reference
The test images are shared only AFTER submission: you must then run this
file UNCHANGED on them. Make sure it runs on any folder of images.

Rules:
  * enhance_underwater(): spatial-domain techniques only (no Fourier/wavelet, no
    machine learning, no neural networks or pretrained models). It may use simple
    statistics computed from the given references (e.g. a target histogram).
    Same pipeline and parameters for every image. It is re-used unchanged in Question 3.

Pipeline (all spatial domain):
    1. Red-channel compensation   -> undo the strong absorption of red light in water
    2. Underwater dehazing (UDCP) -> remove the fog caused by scattering
    3. Histogram specification    -> match the L*a*b* histograms of the given references
"""

import argparse
import os

import cv2
import numpy as np

# the given references, e.g. for a target histogram
GIVEN_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uieb", "reference")

# ---- one fixed parameter set, used for every image (chosen with tune_q2.py) ----
RED_ALPHA = 1.0         # strength of red compensation
PATCH = 15              # dark-channel window size (pixels)
OMEGA = 0.60            # fraction of the haze removed (< 1 keeps a depth cue)
T_MIN = 0.45            # lower bound on transmission (limits noise amplification)
GF_R, GF_EPS = 30, 1e-3 # guided-filter radius / regularisation
HIST_BLEND = 0.6        # 0 = no specification, 1 = full specification

# ablation switch: put "red", "dehaze" and/or "hist" in here to disable a step
SKIP = set()


def _to_float(img):
    return img.astype(np.float32) / 255.0


def _to_uint8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Technique 1: red-channel compensation
# ---------------------------------------------------------------------------
def compensate_red(f, alpha=None):
    """
    Red light is absorbed fastest in water, green survives. Move some of the
    green signal into red:

        R_c = R + alpha * (mean_G - mean_R) * (1 - R) * G

    Controls against over-reddening:
      * proportional to G: red is added only where there is scene signal,
        and the noise in R is never multiplied;
      * (1 - R): pixels that are already red get almost nothing;
      * zero when red is not attenuated, and mean(R_c) can never exceed mean(G).
    """
    alpha = RED_ALPHA if alpha is None else alpha
    b, g, r = cv2.split(f)
    gap = float(g.mean() - r.mean())
    if gap <= 0:
        return f
    boost = gap * (1.0 - r) * g
    alpha = min(alpha, gap / (float(boost.mean()) + 1e-6))
    return cv2.merge([b, g, np.clip(r + alpha * boost, 0, 1)])


# ---------------------------------------------------------------------------
# Technique 2: dehazing with the underwater dark channel prior
# ---------------------------------------------------------------------------
def guided_filter(guide, p, r, eps):
    """Edge-preserving smoothing of p, steered by guide (box filters only)."""
    k = (2 * r + 1, 2 * r + 1)
    m = lambda x: cv2.boxFilter(x, -1, k)
    mI, mp = m(guide), m(p)
    a = (m(guide * p) - mI * mp) / (m(guide * guide) - mI * mI + eps)
    b = mp - a * mI
    return m(a) * guide + m(b)


def dehaze(f):
    """
    Scattering model:  I = J*t + A*(1 - t)
      J = clear scene, t = transmission, A = background (water) light.
    In a haze-free patch some channel is close to 0, so the patch minimum
    measures the haze. Red is dark everywhere underwater because of
    absorption, so only G and B are used (underwater dark channel prior).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (PATCH, PATCH))
    dark = cv2.erode(np.minimum(f[:, :, 0], f[:, :, 1]), kernel)      # min filter

    # background light A: mean colour of the 0.1% haziest pixels
    n = max(1, dark.size // 1000)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    A = np.clip(f.reshape(-1, 3)[idx].mean(axis=0), 0.05, 1.0)

    # transmission, refined by a guided filter to avoid blocky halos
    norm = np.minimum(f[:, :, 0] / A[0], f[:, :, 1] / A[1])
    t = 1.0 - OMEGA * cv2.erode(norm, kernel)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    t = np.clip(guided_filter(gray, t.astype(np.float32), GF_R, GF_EPS), T_MIN, 1.0)

    J = (f - A) / t[:, :, None] + A
    return np.clip(J, 0, 1)


# ---------------------------------------------------------------------------
# Technique 3: histogram specification to the given references
# ---------------------------------------------------------------------------
_TARGET = None   # cached in memory, computed once per run


def _lab_hist(img_u8):
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    return np.stack([np.bincount(lab[:, :, c].ravel(), minlength=256)
                     for c in range(3)]).astype(np.float64)


def load_target_hist():
    """Average normalised L*, a*, b* histograms of the GIVEN references
    (never the test references). Returns None if they are not available."""
    global _TARGET
    if _TARGET is None and os.path.isdir(GIVEN_REF):
        hists = []
        for name in sorted(os.listdir(GIVEN_REF)):
            ref = cv2.imread(os.path.join(GIVEN_REF, name), cv2.IMREAD_COLOR)
            if ref is not None:
                h = _lab_hist(ref)
                hists.append(h / h.sum(axis=1, keepdims=True))
        if hists:
            _TARGET = np.mean(hists, axis=0)
    return _TARGET


def _match_channel(ch, target_hist):
    """Textbook specification z = G^-1(T(r)), using the two CDFs."""
    src_cdf = np.cumsum(np.bincount(ch.ravel(), minlength=256)) / ch.size
    tgt_cdf = np.cumsum(target_hist) / target_hist.sum()
    lut = np.clip(np.searchsorted(tgt_cdf, src_cdf), 0, 255).astype(np.float32)
    return lut[ch]


def specify_histogram(f, target, beta=None):
    """
    L*     -> brightness / contrast like the references
    a*, b* -> colour distribution like the references (removes leftover cast)
    Blended with the input (beta) so each image keeps its own character.
    """
    beta = HIST_BLEND if beta is None else beta
    lab = cv2.cvtColor(_to_uint8(f), cv2.COLOR_BGR2LAB)
    out = lab.astype(np.float32)
    for c in range(3):
        out[:, :, c] = (1 - beta) * lab[:, :, c] + beta * _match_channel(lab[:, :, c], target[c])
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return _to_float(cv2.cvtColor(out, cv2.COLOR_LAB2BGR))


# ---------------------------------------------------------------------------
def enhance_underwater(img):
    """
    img : uint8 array of shape (H, W, 3), BGR (as read by cv2.imread)
    returns : uint8 array of shape (H, W, 3), BGR

    Combine at least THREE spatial-domain techniques. Write each technique
    as a separate helper function so that it is easy to follow.
    """
    f = _to_float(img)
    if "red" not in SKIP:
        f = compensate_red(f)          # 1. absorption
    if "dehaze" not in SKIP:
        f = dehaze(f)                  # 2. scattering
    if "hist" not in SKIP:
        target = load_target_hist()
        if target is not None:         # skipped only if data/uieb/reference is missing
            f = specify_histogram(f, target, beta= 0.7)
    return _to_uint8(f)


def run(fn, in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        img = cv2.imread(os.path.join(in_dir, name), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cv2.imwrite(os.path.join(out_dir, name), fn(img))
        print(f"saved {os.path.join(out_dir, name)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of raw underwater images")
    ap.add_argument("--output", required=True, help="output folder")
    args = ap.parse_args()
    run(enhance_underwater, args.input, args.output)


if __name__ == "__main__":
    main()
