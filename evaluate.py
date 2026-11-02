"""
CENG 466 - THE1 evaluation script.

Computes image quality metrics between your output images and the reference images.
Do NOT modify this file; all reported numbers must be produced with it.

Usage:
    python evaluate.py <output_dir> <reference_dir>

Images are matched by file name. Both images must have the same size.
Only NumPy and Pillow are needed.

FULL-REFERENCE metrics (output compared with the reference):
  PSNR    Peak Signal-to-Noise Ratio in dB, from the mean squared error.       higher = better
  SSIM    Structural Similarity: 11x11 Gaussian window (sigma 1.5), K1 = 0.01,
          K2 = 0.03, per RGB channel, averaged.                                higher = better (max 1)
  dE2000  Mean CIEDE2000 colour difference in CIELab (sRGB, D65).              lower = better
          (~1 invisible, ~2-3 just noticeable, >10 clearly different colours)
  GMSD    Gradient Magnitude Similarity Deviation (Xue et al., 2014, as in the
          authors' MATLAB code): how uniformly the edges/gradients of the
          output match the reference.                                         lower = better (0 = same)

NO-REFERENCE metrics (computed on one image alone). There is no "better"
direction: compare your output's value with the REFERENCE image's value.
  noise      noise standard deviation of the luminance (gray) image, in gray levels
             (Immerkaer, 1996); independent noise of std s in each RGB channel reads
             about 0.67 * s, because the gray conversion averages part of it away
  sharpness  mean gradient magnitude (Sobel); noise also raises it
  colourful  colourfulness (Hasler & Suesstrunk, 2003)
  entropy    Shannon entropy of the gray-level histogram, bits (max 8)
  UCIQE      underwater colour image quality (Yang & Sowmya, 2015), in CIELab:
             0.4680 * std(chroma) + 0.2745 * (99th - 1st percentile of L*)
             + 0.2576 * mean(chroma / sqrt(chroma^2 + L*^2)); implementations of
             UCIQE differ in these details, so compare only values from this script
"""

import os
import sys

import numpy as np
from PIL import Image


def read_image(path):
    """Reads an image as an RGB uint8 array of shape (H, W, 3)."""
    return np.asarray(Image.open(path).convert("RGB"))


def _check(output, reference):
    output = np.asarray(output, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if output.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {output.shape} vs {reference.shape}")
    return output, reference


# --------------------------------------------------------------------------- PSNR
def psnr(output, reference, data_range=255.0):
    output, reference = _check(output, reference)
    mse = np.mean((output - reference) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


# --------------------------------------------------------------------------- SSIM
def _gaussian_kernel(size=11, sigma=1.5):
    ax = np.arange(size) - (size - 1) / 2.0
    k = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _filter2(x, k):
    """Separable 'valid' correlation of a 2D array with a 1D kernel."""
    n = len(k)
    rows = sum(k[i] * x[i:x.shape[0] - n + 1 + i, :] for i in range(n))
    return sum(k[i] * rows[:, i:rows.shape[1] - n + 1 + i] for i in range(n))


def _ssim_channel(x, y, data_range):
    k = _gaussian_kernel()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x, mu_y = _filter2(x, k), _filter2(y, k)
    sxx = _filter2(x * x, k) - mu_x ** 2
    syy = _filter2(y * y, k) - mu_y ** 2
    sxy = _filter2(x * y, k) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return float(np.mean(num / den))


def ssim(output, reference, data_range=255.0):
    output, reference = _check(output, reference)
    if output.ndim == 2:
        return _ssim_channel(output, reference, data_range)
    return float(np.mean([_ssim_channel(output[..., c], reference[..., c], data_range)
                          for c in range(output.shape[2])]))


# --------------------------------------------------------------------------- colour
def rgb_to_lab(img):
    """sRGB uint8/float in [0, 255] -> CIELab (D65 white point)."""
    c = np.asarray(img, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e2000(output, reference):
    """Mean CIEDE2000 colour difference (Sharma, Wu & Dalal, 2005)."""
    output, reference = _check(output, reference)
    L1, a1, b1 = np.moveaxis(rgb_to_lab(reference), -1, 0)
    L2, a2, b2 = np.moveaxis(rgb_to_lab(output), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cm7 / (Cm7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    hsum = h1p + h2p
    hpm = np.where(np.abs(h1p - h2p) > 180, np.where(hsum < 360, hsum + 360, hsum - 360), hsum) / 2
    hpm = np.where(C1p * C2p == 0, hsum, hpm)
    T = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
         + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    d_theta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm ** 7 / (Cpm ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * d_theta)) * Rc
    de = np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))
    return float(de.mean())


# --------------------------------------------------------------------------- gradients
def _gray(img):
    img = np.asarray(img, dtype=np.float64)
    return img @ np.array([0.299, 0.587, 0.114]) if img.ndim == 3 else img


def _conv3(x, k):
    """3x3 correlation with edge replication (output has the same size)."""
    p = np.pad(x, 1, mode="edge")
    h, w = x.shape
    return sum(k[i, j] * p[i:i + h, j:j + w] for i in range(3) for j in range(3))


def _conv2_same(x, k):
    """2-D convolution with zero padding, cropped like MATLAB's conv2(x, k, 'same')."""
    kh, kw = k.shape
    h, w = x.shape
    full = np.zeros((h + kh - 1, w + kw - 1))
    for i in range(kh):
        for j in range(kw):
            full[i:i + h, j:j + w] += k[i, j] * x
    r0, c0 = kh // 2, kw // 2
    return full[r0:r0 + h, c0:c0 + w]


def gmsd(output, reference, c=170.0):
    """
    Gradient Magnitude Similarity Deviation, following the authors' MATLAB code
    (Xue, Zhang, Mou & Bovik, IEEE TIP 2014): luminance, 2x2 average + 2x downsampling,
    Prewitt gradients, GMS map with c = 170, standard deviation (N - 1).
    """
    output, reference = _check(output, reference)
    ave = np.full((2, 2), 0.25)
    dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3.0

    def grad_mag(img):
        g = _conv2_same(_gray(img), ave)[::2, ::2]
        return np.hypot(_conv2_same(g, dx), _conv2_same(g, dx.T))

    mo, mr = grad_mag(output), grad_mag(reference)
    gms = (2 * mo * mr + c) / (mo ** 2 + mr ** 2 + c)
    return float(gms.std(ddof=1))


# --------------------------------------------------------------------------- no-reference
def noise_sigma(img):
    """Noise standard deviation of the luminance image (Immerkaer, 1996)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    g = _gray(img)
    return float(np.sqrt(np.pi / 2) / 6 * np.abs(_conv3(g, k))[1:-1, 1:-1].mean())


def sharpness(img):
    """Mean Sobel gradient magnitude."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    g = _gray(img)
    return float(np.hypot(_conv3(g, kx), _conv3(g, kx.T)).mean())


def colourfulness(img):
    r, g, b = (np.asarray(img, dtype=np.float64)[..., i] for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def entropy(img):
    levels = np.clip(np.round(_gray(img)), 0, 255).astype(np.int64)
    p = np.bincount(levels.ravel(), minlength=256) / levels.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def uciqe(img):
    lab = rgb_to_lab(img)
    L, chroma = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    lo, hi = np.percentile(L, (1, 99))
    sat = chroma / np.maximum(np.hypot(chroma, L), 1e-6)
    return float(0.4680 * chroma.std() + 0.2745 * (hi - lo) + 0.2576 * sat.mean())


FULL_REFERENCE = {"PSNR": psnr, "SSIM": ssim, "dE2000": delta_e2000, "GMSD": gmsd}
NO_REFERENCE = {"noise": noise_sigma, "sharpness": sharpness, "colourful": colourfulness,
                "entropy": entropy, "UCIQE": uciqe}


def all_metrics(output, reference):
    """Dict with every full-reference metric of output vs reference and every no-reference metric of output."""
    m = {k: f(output, reference) for k, f in FULL_REFERENCE.items()}
    m.update({k: f(output) for k, f in NO_REFERENCE.items()})
    return m


def evaluate_dirs(output_dir, reference_dir):
    """Returns a list of (name, metrics dict) for every image present in both folders."""
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    names = sorted(n for n in os.listdir(reference_dir) if n.lower().endswith(exts))
    results = []
    for name in names:
        out_path = os.path.join(output_dir, name)
        if not os.path.exists(out_path):
            print(f"[missing] {name}")
            continue
        out, ref = read_image(out_path), read_image(os.path.join(reference_dir, name))
        if out.shape != ref.shape:
            print(f"[size mismatch] {name}: output {out.shape[1]}x{out.shape[0]}, "
                  f"reference {ref.shape[1]}x{ref.shape[0]} (skipped)")
            continue
        results.append((name, all_metrics(out, ref)))
    return results


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    output_dir, reference_dir = sys.argv[1], sys.argv[2]
    results = evaluate_dirs(output_dir, reference_dir)
    if not results:
        print("No matching images found.")
        sys.exit(1)
    keys = list(FULL_REFERENCE) + list(NO_REFERENCE)
    fmt = {"SSIM": "{:>10.4f}", "GMSD": "{:>10.4f}"}
    print(f"{'image':<24}" + "".join(f"{k:>10}" for k in keys))
    for name, m in results:
        print(f"{name:<24}" + "".join(fmt.get(k, "{:>10.2f}").format(m[k]) for k in keys))
    print(f"{'MEAN':<24}" + "".join(fmt.get(k, "{:>10.2f}").format(np.mean([m[k] for _, m in results]))
                                    for k in keys))
    refs = [read_image(os.path.join(reference_dir, n)) for n, _ in results]
    print(f"{'REFERENCE (no-ref only)':<24}" + " " * 40
          + "".join(f"{np.mean([f(r) for r in refs]):>10.2f}" for f in NO_REFERENCE.values()))


if __name__ == "__main__":
    main()
