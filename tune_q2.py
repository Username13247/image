"""
Grid search for the q2.py parameters on the GIVEN UIEB images.
Run from inside THE1/:   python3 tune_q2.py
Writes tune_q2_results.csv and prints the best settings.

Metrics are computed with NumPy/OpenCV only (no extra packages);
always confirm the chosen setting afterwards with evaluate.py.
"""

import csv
import itertools
import os
import time

import cv2
import numpy as np

import q2

RAW_DIR = "data/uieb/raw"
REF_DIR = "data/uieb/reference"

# ---- the grid (current values marked with *) ----
GRID_ALPHA = [0.5, 0.75, 1.0]             # RED_ALPHA   (*1.0)
GRID_OMEGA = [0.6, 0.7, 0.8, 0.85, 0.9]   # OMEGA       (*0.85)
GRID_TMIN = [0.2, 0.3, 0.35, 0.45]        # T_MIN       (*0.35)
GRID_BETA = [0.3, 0.4, 0.5, 0.6, 0.7]     # HIST_BLEND  (*0.6)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    """SSIM (Wang et al. 2004), 11x11 Gaussian window, averaged over B, G, R."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    vals = []
    for c in range(3):
        x, y = a[:, :, c].astype(np.float64), b[:, :, c].astype(np.float64)
        mx, my = g(x), g(y)
        sxx, syy, sxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
        m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
        vals.append(m.mean())
    return float(np.mean(vals))


def _lab(img_bgr):
    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab).astype(np.float64)


def delta_e(a, b):
    """Mean CIEDE2000 colour difference (Sharma et al. 2005 formulation)."""
    L1, a1, b1 = np.moveaxis(_lab(a), 2, 0)
    L2, a2, b2 = np.moveaxis(_lab(b), 2, 0)
    Cbar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    zero = (C1p * C2p) == 0

    dLp, dCp = L2 - L1, C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where(zero, 0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hsum, hdiff = h1p + h2p, np.abs(h1p - h2p)
    hbp = np.where(hdiff <= 180, hsum / 2,
                   np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2))
    hbp = np.where(zero, hsum, hbp)

    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-((hbp - 275) / 25) ** 2)
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    dE = np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                 + Rt * (dCp / Sc) * (dHp / Sh))
    return float(dE.mean())


def main():
    names = sorted(f for f in os.listdir(RAW_DIR)
                   if cv2.imread(os.path.join(RAW_DIR, f)) is not None)
    raws = [cv2.imread(os.path.join(RAW_DIR, n)) for n in names]
    refs = [cv2.imread(os.path.join(REF_DIR, n)) for n in names]
    target = q2.load_target_hist()

    rows = []
    t0 = time.time()
    combos = list(itertools.product(GRID_ALPHA, GRID_OMEGA, GRID_TMIN))
    for k, (alpha, omega, tmin) in enumerate(combos, 1):
        q2.OMEGA, q2.T_MIN = omega, tmin          # dehaze() reads these globals
        # steps 1-2 do not depend on beta: compute once, reuse for every beta
        mids = [q2.dehaze(q2.compensate_red(q2._to_float(r), alpha=alpha)) for r in raws]
        for beta in GRID_BETA:
            p, s, d = [], [], []
            for mid, ref in zip(mids, refs):
                out = q2._to_uint8(q2.specify_histogram(mid, target, beta=beta))
                p.append(psnr(out, ref)); s.append(ssim(out, ref)); d.append(delta_e(out, ref))
            rows.append(dict(alpha=alpha, omega=omega, tmin=tmin, beta=beta,
                             psnr=np.mean(p), ssim=np.mean(s), dE=np.mean(d),
                             worst_dE=np.max(d), per_image_dE=d))
        print(f"[{k}/{len(combos)}] alpha={alpha} omega={omega} tmin={tmin}  "
              f"({time.time() - t0:.0f}s)")

    # rank each metric (PSNR, SSIM higher = better; dE lower = better) and average
    for key, rev in (("psnr", True), ("ssim", True), ("dE", False)):
        for rank, r in enumerate(sorted(rows, key=lambda r: r[key], reverse=rev)):
            r["rank_" + key] = rank
    for r in rows:
        r["avg_rank"] = (r["rank_psnr"] + r["rank_ssim"] + r["rank_dE"]) / 3
    rows.sort(key=lambda r: r["avg_rank"])

    cols = ["alpha", "omega", "tmin", "beta", "psnr", "ssim", "dE", "worst_dE", "avg_rank"]
    with open("tune_q2_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    def show(title, lst):
        print(f"\n{title}")
        print(" alpha omega  tmin  beta |  PSNR   SSIM     dE  worst_dE")
        for r in lst:
            print(f"  {r['alpha']:.2f}  {r['omega']:.2f}  {r['tmin']:.2f}  {r['beta']:.2f} |"
                  f" {r['psnr']:5.2f} {r['ssim']:.4f} {r['dE']:6.2f}  {r['worst_dE']:6.2f}")

    show("Top 10 by average rank of PSNR/SSIM/dE:", rows[:10])
    show("Top 5 by dE2000 alone:", sorted(rows, key=lambda r: r["dE"])[:5])

    # robustness: on how many images does the best setting beat the current one?
    cur = next((r for r in rows if (r["alpha"], r["omega"], r["tmin"], r["beta"])
                == (1.0, 0.85, 0.35, 0.6)), None)
    if cur:
        best = rows[0]
        wins = sum(b < c for b, c in zip(best["per_image_dE"], cur["per_image_dE"]))
        print(f"\nBest setting lowers dE on {wins}/{len(names)} images vs. the current one.")
    print("Full table: tune_q2_results.csv")


if __name__ == "__main__":
    main()
