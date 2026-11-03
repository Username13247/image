"""
CENG 466 - THE1 - Question 3: Cross-dataset analysis (LOL <-> UIEB)

Student 1: Name Surname, ID
Student 2: Name Surname, ID

Usage:
    python q3.py --lol <folder of LOL low-light images> --uieb <folder of UIEB raw images> \\
                 --output <output folder>
Examples:
    python q3.py --lol data/lol/low --uieb data/uieb/raw --output outputs/q3          (the given images)
    python q3.py --lol test/lol/low --uieb test/uieb/raw --output test_outputs/q3     (the test images, after the deadline)
Writes, under <output>, the 2x2 grid of runs:
    lol_pipeline_on_lol/     Q1 pipeline on the LOL images
    lol_pipeline_on_uieb/    Q1 pipeline, UNCHANGED, on the UIEB images        (Q3.1)
    uieb_pipeline_on_uieb/   Q2 pipeline on the UIEB images
    uieb_pipeline_on_lol/    Q2 pipeline, UNCHANGED, on the LOL images         (Q3.3)
Evaluate each folder against data/lol/high or data/uieb/reference.
Q3.2 (histograms, pseudo-colored error maps) uses the given pairs, whose
references you have. The test images are shared only AFTER submission;
this file must then run on them UNCHANGED.
"""

import argparse
import os

import cv2
import numpy as np

from q1 import enhance_low_light   # must be used unchanged
from q2 import enhance_underwater  # must be used unchanged

LOL_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lol", "high")
UIEB_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uieb", "reference")


# gray level -> BGR color: blue -> cyan -> green -> yellow -> red
_STOPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
_COLORS = np.array([
    [255,   0,   0],   # blue   (BGR order!)
    [255, 255,   0],   # cyan
    [  0, 255,   0],   # green
    [  0, 255, 255],   # yellow
    [  0,   0, 255],   # red
], dtype=np.float32)
    

def pseudo_color(gray):
    """
    gray : float array of shape (H, W) with values in [0, 1]
    returns : uint8 BGR image of shape (H, W, 3)

    Implement a pseudo-coloring (intensity slicing or gray-level-to-color
    transformation) YOURSELF. Library colormaps (cv2.applyColorMap,
    matplotlib colormaps, ...) are NOT allowed.
    """
    g = np.clip(gray, 0.0, 1.0).astype(np.float32)
    
    out = np.empty(g.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(g, _STOPS, _COLORS[:, c])

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    
    g = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    return np.dstack([g, g, g])


ERR_MAX = 128.0   # fixed saturation level (intensity levels), same for every image


def error_map(output, reference):
    """
    Returns a pseudo-colored map of the per-pixel error between output and
    reference (both uint8 BGR). Choose and justify the error definition and
    the normalization (e.g. a fixed maximum, so maps can be compared).
    """
    diff = np.abs(output.astype(np.float32) - reference.astype(np.float32))
    err = diff.mean(axis=2)

    return pseudo_color(np.clip(err / ERR_MAX, 0.0, 1.0))
def save_error_maps(error_map, result_dir, ref_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(result_dir)):
        result_img = cv2.imread(os.path.join(result_dir, name), cv2.IMREAD_COLOR)
        ref_img = cv2.imread(os.path.join(ref_dir, name), cv2.IMREAD_COLOR)
        if result_img is None or ref_img is None:
            continue
        cv2.imwrite(os.path.join(out_dir, name), error_map(result_img,ref_img))


def run(pipeline, in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        img = cv2.imread(os.path.join(in_dir, name), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cv2.imwrite(os.path.join(out_dir, name), pipeline(img))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lol", required=True, help="folder of LOL low-light images")
    ap.add_argument("--uieb", required=True, help="folder of UIEB raw images")
    ap.add_argument("--output", required=True, help="output root folder")
    ap.add_argument("--ref-uieb", required=False, help="folder of reference directory for under water images")
    ap.add_argument("--ref-lol", required=False, help="folder of reference directory for low-light images")

    args = ap.parse_args()
    out = args.output
    lol_on_lol=os.path.join(out, "lol_pipeline_on_lol")
    lol_on_uieb=os.path.join(out, "lol_pipeline_on_uieb")
    uieb_on_uieb=os.path.join(out, "uieb_pipeline_on_uieb")
    uieb_on_lol= os.path.join(out, "uieb_pipeline_on_lol")
    run(enhance_low_light, args.lol, lol_on_lol)
    run(enhance_low_light, args.uieb, lol_on_uieb)
    run(enhance_underwater, args.uieb, uieb_on_uieb)
    run(enhance_underwater, args.lol, uieb_on_lol)
    # TODO: Q3.2 histograms and error maps (on the given pairs)

    ref_lol=LOL_REF_DIR 
    ref_uieb=UIEB_REF_DIR
    if args.ref_lol is not None:
        ref_lol=args.ref_lol
    if args.ref_uieb is not None:
        ref_uieb=args.ref_uieb
    

    save_error_maps(error_map, lol_on_lol, ref_lol, os.path.join(out, "error_maps_lol_lol"))
    save_error_maps(error_map, lol_on_uieb, ref_uieb, os.path.join(out, "error_maps_lol_uieb"))
    save_error_maps(error_map, uieb_on_uieb, ref_uieb, os.path.join(out, "error_maps_uieb_uieb"))
    save_error_maps(error_map, uieb_on_lol, ref_lol, os.path.join(out, "error_maps_uieb_lol"))


if __name__ == "__main__":
    main()
