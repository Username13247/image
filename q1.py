"""
CENG 466 - THE1 - Question 1: Low-light image enhancement (LOL dataset)

Student 1: Name Surname, ID
Student 2: Name Surname, ID

Usage:
    python q1.py --input <folder of low-light images> --output <output folder>
Examples:
    python q1.py --input data/lol/low --output outputs/q1          (the given images)
    python q1.py --input test/lol/low --output test_outputs/q1     (the test images, after the deadline)
Outputs keep the input file names and sizes.

Develop, tune and report on the given pairs (data/lol/low, data/lol/high):
    python evaluate.py outputs/q1 data/lol/high
The test images are shared only AFTER submission: you must then run this
file UNCHANGED on them. Make sure it runs on any folder of images.

Rules:
  * enhance_low_light(): spatial-domain techniques only (no Fourier/wavelet, no
    machine learning, no neural networks or pretrained models). It may use simple
    statistics computed from the given references (e.g. a target histogram).
    Same pipeline and parameters for every image.
    It is re-used unchanged in Questions 3 and 4.
"""

import argparse
import os

import cv2
import numpy as np

# the given references, e.g. for a target histogram
GIVEN_HIGH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lol", "high")


def enhance_low_light(img):
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
    ap.add_argument("--input", required=True, help="folder of low-light images")
    ap.add_argument("--output", required=True, help="output folder")
    args = ap.parse_args()
    run(enhance_low_light, args.input, args.output)


if __name__ == "__main__":
    main()
