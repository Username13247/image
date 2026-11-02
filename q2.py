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
"""

import argparse
import os

import cv2
import numpy as np

# the given references, e.g. for a target histogram
GIVEN_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uieb", "reference")


# ---------------------------------------------------------------------------
# Pipeline configuration. Fixed for ALL images (no per-image tuning).
# Each step can be switched off through `skip` -> used for the ablation study.
# ---------------------------------------------------------------------------
CFG = dict(
    alpha=1.0,          # red compensation strength
    gain_lo=0.5,        # gray-world gains are clipped to [gain_lo, gain_hi]
    gain_hi=2.0,
    clahe_clip=2.0,     # CLAHE clip limit (on L channel)
    clahe_tile=8,       # CLAHE tile grid (8x8)
    beta=0.6,           # blend weight of histogram specification (0 = off, 1 = full)
    usm_sigma=2.0,      # unsharp mask Gaussian sigma
    usm_amount=0.5,     # unsharp mask strength
)
STEPS = ("red", "wb", "clahe", "hist", "sharpen")

_TARGET_CDF = None  # cached mean CDF of the given references, shape (3, 256), BGR order


def _load_target_cdf():
    """Mean per-channel CDF of the given reference images (simple reference statistics)."""
    global _TARGET_CDF
    if _TARGET_CDF is not None:
        return _TARGET_CDF
    hist = np.zeros((3, 256), np.float64)
    n = 0
    if os.path.isdir(GIVEN_REF):
        for name in sorted(os.listdir(GIVEN_REF)):
            ref = cv2.imread(os.path.join(GIVEN_REF, name), cv2.IMREAD_COLOR)
            if ref is None:
                continue
            for c in range(3):
                h = np.bincount(ref[..., c].ravel(), minlength=256).astype(np.float64)
                hist[c] += h / h.sum()      # each image weighs the same
            n += 1
    if n == 0:
        _TARGET_CDF = False                 # no references available -> step is skipped
        return _TARGET_CDF
    cdf = np.cumsum(hist / n, axis=1)
    _TARGET_CDF = cdf / cdf[:, -1:]
    return _TARGET_CDF


# ----------------------------- technique 1: absorption -----------------------------
def red_compensation(f, alpha):
    """Absorption fix. f: float BGR in [0,1]. Red is attenuated most by water, so add back a
    fraction of the (mean G - mean R) gap:  R' = R + alpha*(mG - mR)*(1 - R)*G.
    (1-R): do not push pixels that already have red into saturation.
    G: boost follows scene structure and leaves dark/empty regions dark."""
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    gap = max(g.mean() - r.mean(), 0.0)     # only compensate when red is the weak channel
    out = f.copy()
    out[..., 2] = np.clip(r + alpha * gap * (1.0 - r) * g, 0, 1)
    return out


def gray_world(f, lo, hi):
    """Remaining colour cast (blue / green) fix: scale each channel so channel means are equal."""
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    gains = np.clip(means.mean() / means, lo, hi)
    return np.clip(f * gains, 0, 1)


# ----------------------------- technique 2: scattering -----------------------------
def clahe_lightness(img8, clip, tile):
    """Haze / low contrast fix: CLAHE on L of Lab only, so hues (and dE) are preserved."""
    lab = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[..., 0] = cl.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ----------------------------- technique 3: histogram specification ----------------
def hist_specification(img8, target_cdf, beta):
    """Per-channel histogram matching to the mean reference histogram, blended with the input:
    out = beta*matched + (1-beta)*input. Own implementation via CDF inversion."""
    out = np.empty_like(img8)
    levels = np.arange(256)
    for c in range(3):
        h = np.bincount(img8[..., c].ravel(), minlength=256).astype(np.float64)
        cdf = np.cumsum(h)
        cdf /= cdf[-1]
        lut = np.interp(cdf, target_cdf[c], levels)   # for each input level, level with equal CDF
        out[..., c] = np.clip(lut, 0, 255).astype(np.uint8)[img8[..., c]]
    mixed = beta * out.astype(np.float32) + (1 - beta) * img8.astype(np.float32)
    return np.clip(mixed + 0.5, 0, 255).astype(np.uint8)


# ----------------------------- technique 4: sharpening -----------------------------
def unsharp_mask(img8, sigma, amount):
    """out = img + amount*(img - Gaussian(img)); restores detail softened by scattering."""
    blur = cv2.GaussianBlur(img8, (0, 0), sigma)
    out = cv2.addWeighted(img8, 1 + amount, blur, -amount, 0)
    return out


def enhance_underwater(img, skip=()):
    """
    img : uint8 array of shape (H, W, 3), BGR (as read by cv2.imread)
    returns : uint8 array of shape (H, W, 3), BGR

    Pipeline: red compensation -> gray world -> CLAHE(L) -> histogram specification -> unsharp mask.
    `skip` (tuple of names in STEPS) is only used for the ablation study; default = full pipeline.
    """
    c = CFG
    f = img.astype(np.float32) / 255.0
    if "red" not in skip:
        f = red_compensation(f, c["alpha"])
    if "wb" not in skip:
        f = gray_world(f, c["gain_lo"], c["gain_hi"])
    out = np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if "clahe" not in skip:
        out = clahe_lightness(out, c["clahe_clip"], c["clahe_tile"])
    if "hist" not in skip:
        tcdf = _load_target_cdf()
        if tcdf is not False:
            out = hist_specification(out, tcdf, c["beta"])
    if "sharpen" not in skip:
        out = unsharp_mask(out, c["usm_sigma"], c["usm_amount"])
    return out


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