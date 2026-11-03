"""
Q2 report helper: baselines, ablation study, parameter sensitivity and a visual ablation strip.
Run from inside THE1/ (next to q2.py, evaluate.py and data/):

    python3 q2_ablation.py                 # tables + CSV + visual strip   (a few minutes)
    python3 q2_ablation.py --sweep         # also one-at-a-time parameter sensitivity

Writes to ablation_out/:
    ablation_mean.csv      mean metrics of every variant            (table for the report)
    ablation_per_image.csv metrics of every variant for every image (find best / worst images)
    ablation_visual.png    one row per image: raw | only-X | w/o-X | full | reference

Nothing in q2.py is changed: the steps are switched off through q2.SKIP.
"""
import argparse
import csv
import os

import cv2
import numpy as np

import q2
from evaluate import all_metrics, read_image, FULL_REFERENCE, NO_REFERENCE

RAW, REF, OUT = "data/uieb/raw", "data/uieb/reference", "ablation_out"
KEYS = list(FULL_REFERENCE) + list(NO_REFERENCE)
STEPS = ("red", "dehaze", "hist")
LABEL = {"red": "red compensation", "dehaze": "dehazing (UDCP)", "hist": "histogram spec."}

NAMES = sorted(n for n in os.listdir(REF) if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
RAWS = {n: cv2.imread(os.path.join(RAW, n), cv2.IMREAD_COLOR) for n in NAMES}
REFS = {n: read_image(os.path.join(REF, n)) for n in NAMES}

_CACHE = {}


def run_variant(skip):
    """Enhanced images (BGR uint8) of every given image with the steps in `skip` switched off.
    skip=None returns the raw inputs."""
    key = None if skip is None else tuple(sorted(skip))
    if key not in _CACHE:
        if skip is None:
            _CACHE[key] = {n: RAWS[n] for n in NAMES}
        else:
            old = q2.SKIP
            q2.SKIP = set(skip)
            try:
                _CACHE[key] = {n: q2.enhance_underwater(RAWS[n]) for n in NAMES}
            finally:
                q2.SKIP = old
    return _CACHE[key]


def metrics_of(images):
    return {n: all_metrics(images[n][..., ::-1], REFS[n]) for n in NAMES}


def mean_row(per_image):
    return [float(np.mean([per_image[n][k] for n in NAMES])) for k in KEYS]


def variants():
    v = [("input (raw)", None), ("FULL pipeline", set())]
    v += [(f"baseline: only {LABEL[s]}", set(STEPS) - {s}) for s in STEPS]
    v += [(f"w/o {LABEL[s]}", {s}) for s in STEPS]
    return v


def print_header(title):
    print(f"\n{title}")
    print(f"{'variant':<34}" + "".join(f"{k:>10}" for k in KEYS))


def print_row(label, row):
    print(f"{label:<34}" + "".join(f"{x:>10.3f}" for x in row), flush=True)


# ------------------------------------------------------------------ visual strip
def tile(img, text, w=190):
    h = int(img.shape[0] * w / img.shape[1])
    t = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    for col, th in (((255, 255, 255), 2), ((0, 0, 0), 1)):
        cv2.putText(t, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, th, cv2.LINE_AA)
    return t


def visual_strip(path):
    cols = [("raw", None)]
    cols += [(f"only {s}", set(STEPS) - {s}) for s in STEPS]
    cols += [(f"w/o {s}", {s}) for s in STEPS]
    cols += [("FULL", set())]
    rows = []
    for n in NAMES:
        tiles = [tile(run_variant(sk)[n], lab) for lab, sk in cols]
        tiles.append(tile(REFS[n][..., ::-1].copy(), "reference"))
        h = min(t.shape[0] for t in tiles)
        rows.append(np.hstack([t[:h] for t in tiles]))
    w = min(r.shape[1] for r in rows)
    cv2.imwrite(path, np.vstack([r[:, :w] for r in rows]))


# ------------------------------------------------------------------ parameter sensitivity
SWEEPS = [("RED_ALPHA", (0.5, 1.0, 1.5)),
          ("OMEGA", (0.3, 0.6, 0.8)),
          ("T_MIN", (0.3, 0.45, 0.6)),
          ("HIST_BLEND", (0.4, 0.7, 1.0))]


def sweep():
    print_header("Parameter sensitivity (one parameter at a time, others at their q2.py value)")
    for name, values in SWEEPS:
        default = getattr(q2, name)
        for val in values:
            setattr(q2, name, val)
            _CACHE.clear()
            tag = "  <- current" if val == default else ""
            print_row(f"{name}={val}{tag}", mean_row(metrics_of(run_variant(set()))))
        setattr(q2, name, default)
    _CACHE.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="also run the parameter sensitivity")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    ref_nr = {k: float(np.mean([f(REFS[n]) for n in NAMES])) for k, f in NO_REFERENCE.items()}
    print("REFERENCE no-reference values (aim to be close): "
          + ", ".join(f"{k}={v:.2f}" for k, v in ref_nr.items()))

    print_header("Baselines and ablation")
    mean_rows, per_rows = [], []
    for label, skip in variants():
        per = metrics_of(run_variant(skip))
        row = mean_row(per)
        print_row(label, row)
        mean_rows.append([label] + row)
        per_rows += [[label, n] + [per[n][k] for k in KEYS] for n in NAMES]

    with open(os.path.join(OUT, "ablation_mean.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant"] + KEYS)
        w.writerows(mean_rows)
        w.writerow(["REFERENCE (no-ref only)"] + [""] * len(FULL_REFERENCE) + [ref_nr[k] for k in NO_REFERENCE])
    with open(os.path.join(OUT, "ablation_per_image.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "image"] + KEYS)
        w.writerows(per_rows)

    full = metrics_of(run_variant(set()))
    print("\nPer-image dE2000 of the FULL pipeline (worst first):")
    for n in sorted(NAMES, key=lambda n: -full[n]["dE2000"]):
        print(f"  {n:<22} dE2000={full[n]['dE2000']:.2f}  PSNR={full[n]['PSNR']:.2f}")

    visual_strip(os.path.join(OUT, "ablation_visual.png"))
    print(f"\nwrote {OUT}/ablation_mean.csv, ablation_per_image.csv, ablation_visual.png")
    if args.sweep:
        sweep()


if __name__ == "__main__":
    main()
