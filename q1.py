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


def exposure_scale(img):
    """Estimate a shared exposure gain from channel percentiles."""
    x = img.astype(np.float32) / 255.0

    highs = np.percentile(x, 99.5, axis=(0, 1))
    upper = float(
        0.5 * np.max(highs) + 0.5 * np.mean(highs)
    )

    gain = (1.0 / max(upper, 1.0 / 255.0)) ** 0.9

    return np.clip(x * gain, 0.0, 1.0).astype(np.float32)


def balance_colors(x):
    """Apply partial gray-world white balancing."""
    means = x.mean(axis=(0, 1))
    gains = means.mean() / np.maximum(means, 1.0 / 255.0)

    strength = 0.5
    result = x * (1.0 + strength * (gains - 1.0))

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def estimate_noise(y):
    """Estimate noise using the MAD of a spatial filter response."""
    kernel = np.array(
        [
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1],
        ],
        dtype=np.float32,
    )

    residual = cv2.filter2D(y, -1, kernel)

    # Exclude the border when the image is large enough.
    if min(y.shape) > 2:
        residual = residual[1:-1, 1:-1]

    median = np.median(residual)
    mad = np.median(np.abs(residual - median))

    # 6 is the square root of the sum of squared kernel weights.
    return float(mad / (0.6745 * 6.0))


def denoise(x):
    """Filter luminance adaptively and smooth chroma separately."""
    ycc = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    noise = estimate_noise(y)
    sigma_color = float(
        np.clip(3.0 * noise, 5.0 / 255.0, 25.0 / 255.0)
    )

    y = cv2.bilateralFilter(
        y,
        d=9,
        sigmaColor=sigma_color,
        sigmaSpace=3,
    )

    cr = cv2.GaussianBlur(cr, (9, 9), 2.0)
    cb = cv2.GaussianBlur(cb, (9, 9), 2.0)

    result = cv2.cvtColor(
        cv2.merge((y, cr, cb)),
        cv2.COLOR_YCrCb2BGR,
    )

    return np.clip(result, 0.0, 1.0)


def gamma_brighten(x):
    """Blend a fixed gamma of 0.5 with a statistics-based gamma."""
    brightness = (
        0.114 * x[:, :, 0]
        + 0.587 * x[:, :, 1]
        + 0.299 * x[:, :, 2]
    )
    mean = float(brightness.mean())

    if mean <= 0.0 or mean >= 1.0:
        return x.copy()

    adaptive = np.clip(
        np.log(0.45) / np.log(mean),
        0.4,
        1.0,
    )
    gamma = 0.25 + 0.5 * adaptive

    return np.power(x, gamma)


def enhance_low_light(img):
    """
    img : uint8 array of shape (H, W, 3), BGR (as read by cv2.imread)
    returns : uint8 array of shape (H, W, 3), BGR

    Combine at least THREE spatial-domain techniques. Write each technique
    as a separate helper function so that it is easy to follow.
    """
    # TODO: implement your pipeline
    
    img = exposure_scale(img)
    img = balance_colors(img)
    img = denoise(img)
    img = gamma_brighten(img)

    return np.clip(
        np.rint(img * 255.0), 0, 255
    ).astype(np.uint8)
    


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
