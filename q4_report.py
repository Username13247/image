"""
Q4.2 report helper: metrics, zoomed texture views and pseudo-colored error maps of the round-trip experiment.
Run from inside THE1/ after  python q4.py  (needs outputs/q4/roundtrip/...).

Writes to q42_out/:
    q42_per_image.csv   PSNR, SSIM, GMSD for every image and every (down, up) combination
    q42_mean.csv        means per dataset and combination (table for the report)
    q42_zoom_<ds>.png   fine-texture crop (nearest-neighbour zoom) of the original and the 6 round trips
    q42_err_<ds>.png    error maps of the three upsamplers, one row per prefilter setting

pseudo_color() comes from q3.py (the Q3.2 mapping), so the error maps use the same colours as Q3.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import psnr, ssim, gmsd, read_image
from q3 import pseudo_color

DATASETS = {"lol": "data/lol/high", "uieb": "data/uieb/reference"}
DOWN, UP = ("decim", "gauss"), ("nearest", "bilinear", "bicubic")
RT = "outputs/q4/roundtrip"
OUT = "q42_out"
CROP, ZOOM = 48, 6
ERR_MAX = 64.0            # fixed maximum (gray levels) so all maps are comparable; larger errors saturate

os.makedirs(OUT, exist_ok=True)
combos = [(d, u) for d in DOWN for u in UP]


def rt_path(d, u, ds, name):
    return os.path.join(RT, f"{d}_{u}", ds, name)


# ---------------------------------------------------------------- metrics
per_rows, mean_rows = [], []
print(f"{'dataset':<6} {'combo':<18}{'PSNR':>8}{'SSIM':>9}{'GMSD':>9}")
for ds, folder in DATASETS.items():
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    refs = {n: read_image(os.path.join(folder, n)) for n in names}
    for d, u in combos:
        vals = []
        for n in names:
            out = read_image(rt_path(d, u, ds, n))
            v = (psnr(out, refs[n]), ssim(out, refs[n]), gmsd(out, refs[n]))
            vals.append(v)
            per_rows.append([ds, f"{d}_{u}", n] + [f"{x:.4f}" for x in v])
        m = np.mean(vals, axis=0)
        mean_rows.append([ds, f"{d}_{u}"] + [f"{x:.4f}" for x in m])
        print(f"{ds:<6} {d + '_' + u:<18}{m[0]:>8.2f}{m[1]:>9.4f}{m[2]:>9.4f}", flush=True)

with open(f"{OUT}/q42_per_image.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "combo", "image", "PSNR", "SSIM", "GMSD"])
    w.writerows(per_rows)
with open(f"{OUT}/q42_mean.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "combo", "PSNR", "SSIM", "GMSD"])
    w.writerows(mean_rows)


# ---------------------------------------------------------------- figures (first image of each dataset)
def texture_box(img):
    """Window with the largest gradient energy of the original: the finest texture."""
    g = img.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    e = np.abs(np.diff(g, axis=0, prepend=g[:1])) + np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    best, pos = -1.0, (0, 0)
    for y in range(0, g.shape[0] - CROP, CROP // 2):
        for x in range(0, g.shape[1] - CROP, CROP // 2):
            s = e[y:y + CROP, x:x + CROP].sum()
            if s > best:
                best, pos = s, (y, x)
    return pos


bar = pseudo_color(np.tile(np.linspace(0, 1, 256), (16, 1)))[..., ::-1]     # BGR -> RGB
for ds, folder in DATASETS.items():
    name = sorted(n for n in os.listdir(folder) if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))[0]
    ref = read_image(os.path.join(folder, name))

    y, x = texture_box(ref)
    tiles = [("original", ref)] + [(f"{d}+{u}", read_image(rt_path(d, u, ds, name))) for d, u in combos]
    fig, ax = plt.subplots(2, 4, figsize=(14, 7.4))
    for a in ax.ravel():
        a.axis("off")
    for a, (t, im) in zip(ax.ravel(), tiles):
        c = im[y:y + CROP, x:x + CROP]
        a.imshow(np.repeat(np.repeat(c, ZOOM, axis=0), ZOOM, axis=1))   # nearest-neighbour zoom: pixels stay visible
        a.set_title(t, fontsize=10)
    fig.suptitle(f"{ds}/{name}: {CROP}x{CROP} crop at ({x},{y}), shown x{ZOOM}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/q42_zoom_{ds}.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(2, 3, figsize=(13, 7.2))
    for i, d in enumerate(DOWN):
        for j, u in enumerate(UP):
            out = read_image(rt_path(d, u, ds, name)).astype(np.float64)
            err = np.abs(out - ref).mean(axis=2)             # mean absolute error over R, G, B (gray levels)
            ax[i, j].imshow(pseudo_color(np.clip(err / ERR_MAX, 0, 1))[..., ::-1])
            ax[i, j].set_title(f"{d} + {u}   (mean |err| = {err.mean():.2f})", fontsize=10)
            ax[i, j].axis("off")
    cax = fig.add_axes([0.35, 0.02, 0.3, 0.025])
    cax.imshow(bar, aspect="auto", extent=[0, ERR_MAX, 0, 1])
    cax.set_yticks([])
    cax.set_xlabel("absolute error (gray levels, saturates at %d)" % ERR_MAX, fontsize=9)
    fig.suptitle(f"{ds}/{name}: round-trip error maps", fontsize=11)
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(f"{OUT}/q42_err_{ds}.png", dpi=150)
    plt.close()

print(f"wrote files to {OUT}/")
