"""
CENG 466 - THE1 - Question 4: Downsampling, upsampling and enhancement order

Student 1: Name Surname, ID
Student 2: Name Surname, ID

Usage:
    python q4.py
Reads   data/lol/high, data/uieb/reference   (Q4.2: the 10 + 10 given references)
        data/lol/low_res, data/lol/low       (Q4.3, optional: the 10 given LOL inputs, x4 downsampled / full size)
Writes
    outputs/q4/roundtrip/<down>_<up>/*.png      (Q4.2, <down> in {decim, gauss},
                                                 <up> in {nearest, bilinear, bicubic})
    outputs/q4/enhance_then_up/*.png            (Q4.3 order A, optional)
    outputs/q4/up_then_enhance/*.png            (Q4.3 order B, optional)

Rules:
  * Interpolation and downsampling must be implemented YOURSELF with NumPy.
    cv2.resize, skimage.transform.resize/rescale, PIL resize, scipy.ndimage.zoom
    and similar functions are NOT allowed in this question.
"""

import os

import cv2
import numpy as np

from q1 import enhance_low_light  # must be used unchanged

FACTOR = 4
LOL_HIGH = os.path.join("data", "lol", "high")
UIEB_REF = os.path.join("data", "uieb", "reference")
LOW_RES = os.path.join("data", "lol", "low_res")
LOL_LOW = os.path.join("data", "lol", "low")
OUT_ROOT = os.path.join("outputs", "q4")


def downsample(img, factor, prefilter):
    """
    img : uint8 (H, W, 3); H and W are divisible by factor.
    prefilter : False -> plain decimation (keep every factor-th pixel)
                True  -> spatial low-pass (smoothing) filter, then decimation
    returns : uint8 (H // factor, W // factor, 3)
    """
    # TODO
    return img[::factor, ::factor]


def upsample_nearest(img, factor):
    """uint8 (h, w, 3) -> uint8 (h * factor, w * factor, 3)"""
    # TODO
    return img


def upsample_bilinear(img, factor):
    """uint8 (h, w, 3) -> uint8 (h * factor, w * factor, 3)"""
    # TODO
    return img


def upsample_bicubic(img, factor):
    """uint8 (h, w, 3) -> uint8 (h * factor, w * factor, 3)"""
    # TODO
    return img


def main():
    # TODO: Q4.2 round-trip experiments on LOL_HIGH and UIEB_REF
    # TODO (optional): Q4.3 both orders on LOW_RES, compare with LOL_HIGH
    pass


if __name__ == "__main__":
    main()
