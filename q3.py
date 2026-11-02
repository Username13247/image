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


def pseudo_color(gray):
    """
    gray : float array of shape (H, W) with values in [0, 1]
    returns : uint8 BGR image of shape (H, W, 3)

    Implement a pseudo-coloring (intensity slicing or gray-level-to-color
    transformation) YOURSELF. Library colormaps (cv2.applyColorMap,
    matplotlib colormaps, ...) are NOT allowed.
    """
    # TODO
    g = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def error_map(output, reference):
    """
    Returns a pseudo-colored map of the per-pixel error between output and
    reference (both uint8 BGR). Choose and justify the error definition and
    the normalization (e.g. a fixed maximum, so maps can be compared).
    """
    # TODO
    err = np.zeros(output.shape[:2])
    return pseudo_color(err)


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
    args = ap.parse_args()
    out = args.output
    run(enhance_low_light, args.lol, os.path.join(out, "lol_pipeline_on_lol"))
    run(enhance_low_light, args.uieb, os.path.join(out, "lol_pipeline_on_uieb"))
    run(enhance_underwater, args.uieb, os.path.join(out, "uieb_pipeline_on_uieb"))
    run(enhance_underwater, args.lol, os.path.join(out, "uieb_pipeline_on_lol"))
    # TODO: Q3.2 histograms and error maps (on the given pairs)


if __name__ == "__main__":
    main()
