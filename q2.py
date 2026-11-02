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
    gain_lo=0.7,        # gray-world gains are clipped to [gain_lo, gain_hi]
    gain_hi=1.5,
    wb_strength=0.5,    # gains**wb_strength: 0 = no balancing, 1 = full gray world
    den_d=7,            # bilateral denoise: neighbourhood diameter
    den_sc=20,          # bilateral: colour sigma (gray levels)
    den_ss=5,           # bilateral: spatial sigma (pixels)
    stretch_strength=0.8,  # per-channel percentile stretch blend (0 = off)
    stretch_lo=1.0,        # percentiles mapped to 0 / 1
    stretch_hi=99.5,
    stretch_max_gain=3.0,  # cap so near-empty channels (red) are not blown into noise
    hist_mode="rgb",      # "L": match lightness only (keeps hues) | "rgb": match each channel
    chroma_gain=1.0,   # Lab a,b scale about neutral (restores saturation)
    max_slope=3.0,      # histogram-spec LUT: maximum slope (limits noise/banding amplification)
    clahe_clip=1.5,     # CLAHE clip limit (on L channel)
    clahe_tile=8,       # CLAHE tile grid (8x8)
    beta=0.4,           # blend weight of histogram specification (0 = off, 1 = full)
    usm_sigma=2.0,      # unsharp mask Gaussian sigma
    usm_amount=0.3,     # unsharp mask strength
)

STEPS = ("denoise", "red", "stretch", "wb", "clahe", "hist", "chroma", "sharpen")

_TARGET_CDF = None  # cached mean CDFs of the given references: {"rgb": (3,256) BGR, "L": (256,)}


def _load_target_cdf():
    """Mean CDFs of the given reference images (simple reference statistics):
    per-channel BGR CDFs and the CDF of the Lab lightness L. Each image weighs the same."""
    global _TARGET_CDF
    if _TARGET_CDF is not None:
        return _TARGET_CDF
    hist = np.zeros((4, 256), np.float64)   # rows: B, G, R, L
    n = 0
    if os.path.isdir(GIVEN_REF):
        for name in sorted(os.listdir(GIVEN_REF)):
            ref = cv2.imread(os.path.join(GIVEN_REF, name), cv2.IMREAD_COLOR)
            if ref is None:
                continue
            chans = [ref[..., c] for c in range(3)] + [cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)[..., 0]]
            for i, ch in enumerate(chans):
                h = np.bincount(ch.ravel(), minlength=256).astype(np.float64)
                hist[i] += h / h.sum()
            n += 1
    if n == 0:
        _TARGET_CDF = False                 # no references available -> step is skipped
        return _TARGET_CDF
    cdf = np.cumsum(hist / n, axis=1)
    cdf /= cdf[:, -1:]
    _TARGET_CDF = {"rgb": cdf[:3], "L": cdf[3]}
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


def channel_stretch(f, lo_p, hi_p, strength, max_gain):
    """Per-channel percentile contrast stretch (textbook contrast stretching, robust version).
    Absorption squeezes the R (and sometimes G/B) range; mapping the lo..hi percentiles of each
    channel to 0..1 re-expands it. The gain is capped and the result blended with the input."""
    out = np.empty_like(f)
    for ch in range(3):
        lo, hi = np.percentile(f[..., ch], (lo_p, hi_p))
        gain = min(1.0 / max(hi - lo, 1e-3), max_gain)
        st = np.clip((f[..., ch] - lo) * gain, 0, 1)
        out[..., ch] = strength * st + (1 - strength) * f[..., ch]
    return out


def gray_world(f, lo, hi, strength=1.0):
    """Remaining colour cast (blue / green) fix: scale each channel so channel means are equal."""
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    gains = np.clip((means.mean() / means) ** strength, lo, hi)
    return np.clip(f * gains, 0, 1)


# ----------------------------- technique 2: scattering -----------------------------
def clahe_lightness(img8, clip, tile):
    """Haze / low contrast fix: CLAHE on L of Lab only, so hues (and dE) are preserved."""
    lab = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[..., 0] = cl.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ----------------------------- technique 3: histogram specification ----------------
def _match_channel(ch, target_cdf, max_slope):
    """CDF-inversion matching of one uint8 channel to a target CDF, with a regularised LUT:
    smoothed and slope-capped so sparsely populated ranges are not stretched into noise / banding."""
    h = np.bincount(ch.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(h)
    cdf /= cdf[-1]
    lut = np.interp(cdf, target_cdf, np.arange(256))   # level with equal CDF in the target
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2); k /= k.sum()
    lut = np.convolve(np.pad(lut, 6, mode="edge"), k, mode="valid")
    lut = lut[0] + np.concatenate([[0.0], np.cumsum(np.clip(np.diff(lut), 0, max_slope))])
    return np.clip(lut, 0, 255).astype(np.uint8)[ch]


def hist_specification(img8, target, beta, max_slope=3.0, mode="L"):
    """Histogram matching to the mean reference histogram, blended with the input:
    out = beta*matched + (1-beta)*input.
    mode "L"  : only Lab lightness is matched -> hues / chroma untouched (no channel-wise hue shifts)
    mode "rgb": each BGR channel is matched independently (can shift hues)."""
    if mode == "L":
        lab = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB)
        L = lab[..., 0]
        m = _match_channel(L, target["L"], max_slope).astype(np.float32)
        lab[..., 0] = np.clip(beta * m + (1 - beta) * L.astype(np.float32) + 0.5, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out = np.empty_like(img8)
    for c in range(3):
        out[..., c] = _match_channel(img8[..., c], target["rgb"][c], max_slope)
    mixed = beta * out.astype(np.float32) + (1 - beta) * img8.astype(np.float32)
    return np.clip(mixed + 0.5, 0, 255).astype(np.uint8)


# ----------------------------- technique 4: saturation -----------------------------
def chroma_boost(img8, gain):
    """Scale the Lab chroma axes (a, b) about the neutral point (128): boosts saturation without
    touching lightness or hue. Counteracts the washed-out look left by scattering and the steps above."""
    lab = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1:] = 128.0 + gain * (lab[..., 1:] - 128.0)
    return cv2.cvtColor(np.clip(lab + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ----------------------------- technique 5: sharpening -----------------------------
def unsharp_mask(img8, sigma, amount):
    """out = img + amount*(img - Gaussian(img)); restores detail softened by scattering."""
    blur = cv2.GaussianBlur(img8, (0, 0), sigma)
    out = cv2.addWeighted(img8, 1 + amount, blur, -amount, 0)
    return out


def enhance_underwater(img, skip=()):
    """
    img : uint8 array of shape (H, W, 3), BGR (as read by cv2.imread)
    returns : uint8 array of shape (H, W, 3), BGR

    Pipeline: bilateral denoise -> red compensation -> channel stretch -> gray world -> CLAHE(L) -> histogram specification(L) -> chroma boost -> unsharp mask.
    `skip` (tuple of names in STEPS) is only used for the ablation study; default = full pipeline.
    """
    c = CFG
    if "denoise" not in skip:
        # placed FIRST: every later step boosts contrast and would amplify noise / JPEG blocks
        img = cv2.bilateralFilter(img, c["den_d"], c["den_sc"], c["den_ss"])
    f = img.astype(np.float32) / 255.0
    if "red" not in skip:
        f = red_compensation(f, c["alpha"])
    if "stretch" not in skip and c["stretch_strength"] > 0:
        f = channel_stretch(f, c["stretch_lo"], c["stretch_hi"], c["stretch_strength"], c["stretch_max_gain"])
    if "wb" not in skip:
        f = gray_world(f, c["gain_lo"], c["gain_hi"], c["wb_strength"])
    out = np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if "clahe" not in skip:
        out = clahe_lightness(out, c["clahe_clip"], c["clahe_tile"])
    if "hist" not in skip:
        tcdf = _load_target_cdf()
        if tcdf is not False:
            out = hist_specification(out, tcdf, c["beta"], c["max_slope"], c["hist_mode"])
    if "chroma" not in skip:
        out = chroma_boost(out, c["chroma_gain"])
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