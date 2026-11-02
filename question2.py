"""
CENG 466 - THE1 - Question 2: Underwater image enhancement (UIEB dataset)

Student 1: Ali Akın Duyar, 2633147
Student 2: Damla Paşaoğlu, ID

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


def enhance_underwater(img):
    """
    img : uint8 array of shape (H, W, 3), BGR (as read by cv2.imread)
    returns : uint8 array of shape (H, W, 3), BGR

    Combine at least THREE spatial-domain techniques. Write each technique
    as a separate helper function so that it is easy to follow.
    """
    # TODO: implement your pipeline


    
    return img


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
