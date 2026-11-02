# CENG 466 THE1 – Question 2 (Underwater Enhancement): What We Did and Why

This file explains how our `q2.py` pipeline evolved, which operations it applies, which
parameters we changed along the way and why. It is written so you can defend every choice in
the oral exam.

---

## 1. Files

| File | Purpose |
|---|---|
| `q2.py` | The submission. `enhance_underwater()` + one helper function per technique. `CFG` holds every parameter. |
| `ablate_q2.py` | Prints mean metrics for: input, 2 baselines, full pipeline, and the pipeline with each step removed, plus parameter sweeps. Gives the report's baseline + ablation tables. |
| `tune_q2.py` | Random search over the global parameters on the 10 given images. Prints the best configurations. (Not part of the submission logic; it only helps us choose values for `CFG`.) |
| `evaluate.py` | Provided metrics script. Never modified. |

Run order: `python3 q2.py --input data/uieb/raw --output outputs/q2`, then
`python3 evaluate.py outputs/q2 data/uieb/reference`.

---

## 2. The physical problem (the "why" behind every step)

- **Absorption** (Beer–Lambert, `I(d) = I0 * exp(-a(λ) d)`): water's vibrational absorption bands reach
  into the red, while blue-green sits in a transparency window. So red decays fastest with depth →
  **blue/green color cast**. (Not "higher energy is absorbed more": blue photons have the most energy
  and are absorbed the least.)
- **Scattering** by particles veils the image like fog → **low contrast, haze, lost detail**.
- Noise is amplified by any contrast boost, and many UIEB images are JPEG-compressed (block artifacts).
- The **references are not ground truth**: volunteers picked the most pleasing output of several
  algorithms. They keep a blue/cyan water tone and strongly saturated warm objects. So "neutral
  gray" is *not* the target, and PSNR/SSIM can be misleading.

---

## 3. The current pipeline (v4), step by step

Order in `enhance_underwater()`:

1. **Bilateral denoise** (`cv2.bilateralFilter`) – cleans noise and JPEG blocks *first*, because every
   later step boosts contrast and would amplify them. Edge-preserving: weights depend on both
   spatial distance and intensity difference, so edges survive.
2. **Red compensation** (own) – targets absorption.
   `R' = R + alpha * (mean(G) - mean(R)) * (1 - R) * G` (values in [0,1]).
   `(1-R)` avoids saturating pixels that already have red; `G` makes the boost follow scene
   structure; only applied if mean(G) > mean(R).
3. **Per-channel percentile stretch** (own) – textbook contrast stretching, robust version.
   For each channel map the 1st..99.5th percentile to 0..1, gain capped at 3, then blend:
   `out = s * stretched + (1 - s) * original`. Re-expands the squeezed channel ranges and brightens dark scenes.
4. **Gray World** (own, partial) – scale each channel so the channel means equalize:
   `gain_c = (mean_all / mean_c) ** wb_strength`, clipped to [0.7, 1.5]. Partial because the
   references keep some blue tone.
5. **CLAHE on L of Lab** (`cv2.createCLAHE`) – targets scattering/haze. Local histogram equalization
   in tiles with a clip limit; doing it on L only keeps hues (protects ΔE).
6. **Histogram specification on L** (own CDF-matching) – maps the image's L histogram toward the *mean
   histogram of the given references* (allowed: "simple statistics of the given references").
   The LUT is smoothed and its slope capped (`max_slope`) so sparse ranges are not stretched into
   noise/banding. Result is blended: `out = beta * matched + (1 - beta) * input`.
7. **Chroma boost** (own) – in Lab, `a' = 128 + g (a - 128)`, same for b. Restores saturation lost
   by the previous steps without touching lightness or hue.
8. **Unsharp mask** – `out = img + amount * (img - Gaussian(img))`. Restores detail softened by scattering.

Why Lab? L (lightness) is separated from a, b (color), so contrast operations (CLAHE, histogram
matching) do not shift hues. ΔE2000 is also computed in Lab.

---

## 4. How we got here (version history and what each change fixed)

### Mean results on the 10 given images

| | PSNR | SSIM | ΔE2000 ↓ | GMSD ↓ | noise | sharpness | colourful | entropy | UCIQE |
|---|---|---|---|---|---|---|---|---|---|
| Raw input | 14.91 | 0.700 | 19.24 | 0.1025 | 1.64 | 33.4 | 64.9 | 6.49 | 18.6 |
| v1 | 18.75 | 0.767 | 12.61 | 0.0917 | 7.50 | 124.6 | 44.4 | 7.68 | 28.7 |
| v2 | 19.93 | 0.821 | **11.87** | 0.0553 | 3.76 | 68.5 | 44.4 | 7.38 | 24.8 |
| v3 | 19.68 | 0.818 | 12.88 | 0.0563 | 3.77 | 67.7 | 52.7 | 7.36 | 25.1 |
| Reference (no-ref only) | – | – | – | – | 3.57 | 71.9 | 63.7 | 7.42 | 28.8 |

v4 (adds channel stretch) has **not been evaluated yet** – fill in its row after running it.

### v1 – first implementation
Steps: red compensation → full Gray World → CLAHE(L) → per-channel RGB histogram matching → unsharp mask.

| Parameter | v1 |
|---|---|
| `alpha` | 1.0 |
| Gray World gains | full strength, clipped 0.5–2.0 |
| `clahe_clip` / tile | 2.0 / 8×8 |
| histogram matching | per-channel RGB, `beta = 0.6`, raw LUT |
| unsharp | sigma 2.0, amount 0.5 |

**Problems seen:** metrics improved over raw, but noise was 7.5 (ref 3.6) and sharpness 125 (ref 72):
over-enhanced, gritty. Visually: harsh HDR look (turtle), magenta blotches in dark background and
banding (statue), JPEG blocks amplified (starfish sky). Statue (488) got *worse* than its raw input.
Cause: too-large red/Gray World gains where red ≈ 0, steep per-channel LUTs, stacked contrast steps.

### v2 – fix noise and over-enhancement
| Change | Why |
|---|---|
| **Added bilateral denoise first** (d=7, σcolor=20, σspace=5) | noise/JPEG blocks were amplified by later steps |
| Gray World `wb_strength` 1.0→**0.5**, gains clipped **0.7–1.5** | avoid magenta; keep water tone like the references |
| Histogram LUT **smoothed + slope cap `max_slope=3`** | stops sparse ranges becoming noise/banding |
| `clahe_clip` 2.0→**1.5** | less contrast stacking |
| `beta` 0.6→**0.4** | gentler pull toward the reference histogram |
| `usm_amount` 0.5→**0.3** | sharpness was far above the reference |

**Result:** noise 3.76, sharpness 68.5, entropy 7.38 now ≈ reference; GMSD almost halved; ΔE 12.61→11.87.
**Remaining problem:** colourfulness 44 vs 64; blue fins turned lavender/pink, green fin turned lime (hue shifts).

### v3 – fix color
| Change | Why |
|---|---|
| `hist_mode` "rgb" → **"L"** | per-channel matching shifted hues; matching only lightness keeps them |
| **Added chroma boost**, `chroma_gain = 1.25` | colourfulness was low (44 vs 64) |

**Result:** colourful 44→52.7 (better visually), but ΔE 11.87→12.88 (worse) and PSNR/SSIM slightly lower.
We changed two things at once, so the ablation (`without hist` vs `hist_mode=rgb`, `chroma_gain` sweep)
must tell us which one cost ΔE. Starfish (387) was still gray with dark water; the reference is bright
blue water and orange starfish.

### v4 – current
| Change | Why |
|---|---|
| **Added per-channel percentile stretch** (`stretch_strength=0.5`, 1–99.5 %, max gain 3) | dark scenes (e.g. 387) have almost no red/mean-gap, so red compensation does ~nothing; stretching re-expands each channel and brightens |
| **Added `tune_q2.py`** | random search over `alpha, wb_strength, stretch_strength, beta, clahe_clip, chroma_gain, hist_mode`; instead of guessing |

Next: run `tune_q2.py`, pick ONE configuration, freeze it in `CFG`.

---

## 5. Full parameter table (current defaults in `CFG`)

| Parameter | Value | Meaning |
|---|---|---|
| `den_d, den_sc, den_ss` | 7, 20, 5 | bilateral: diameter, color sigma, spatial sigma |
| `alpha` | 1.0 | red compensation strength |
| `stretch_strength` | 0.5 | blend of percentile stretch (0 = off) |
| `stretch_lo, stretch_hi` | 1.0, 99.5 | percentiles mapped to 0 and 1 |
| `stretch_max_gain` | 3.0 | gain cap (protects near-empty channels) |
| `wb_strength` | 0.5 | exponent on Gray World gains (0 off, 1 full) |
| `gain_lo, gain_hi` | 0.7, 1.5 | clip range of Gray World gains |
| `clahe_clip, clahe_tile` | 1.5, 8 | CLAHE clip limit, 8×8 tiles |
| `hist_mode` | "L" | match Lab lightness only ("rgb" = per-channel) |
| `beta` | 0.4 | blend of histogram matching |
| `max_slope` | 3.0 | slope cap of the histogram LUT |
| `chroma_gain` | 1.25 | Lab a,b scale |
| `usm_sigma, usm_amount` | 2.0, 0.3 | unsharp mask |

All parameters are global: the same values for every image. They are chosen once on the given
images and frozen before running the test set.

---

## 6. Baselines and ablation (what to put in the report)

- **Baselines (single technique):** Gray World alone, and CLAHE(L) alone (both in `ablate_q2.py`).
- **Ablation:** the full pipeline vs. the pipeline with one step removed:
  `without denoise / red / stretch / wb / clahe / hist / chroma / sharpen`.
- For each row report all 9 metrics (mean + per image for the final one), and compare no-reference
  metrics with the **reference's** values (no "better" direction).
- Honest reporting matters: if a step does not help (e.g. red compensation overlapping with
  histogram matching), say so and justify keeping it by the physics (it targets absorption).

---

## 7. Likely oral-exam questions and short answers

**Why does red disappear first?** Water's vibrational absorption bands (overtones) reach the red; blue-green
photons fall in a transparency window; Beer–Lambert gives exponential decay with a much larger coefficient for red.

**Why denoise first?** Every later step (stretch, CLAHE, histogram matching, sharpening) boosts contrast and
would amplify noise and JPEG blocks. v1 without it had noise 7.5 vs reference 3.6.

**Why bilateral instead of Gaussian/median?** Edge-preserving: weights by intensity as well as distance, so it
smooths flat areas/blocks while keeping edges (less sharpness loss).

**Why Lab and L only for CLAHE/histogram matching?** Separates lightness from color, so contrast changes
do not shift hues; also the space in which ΔE2000 is measured. v1's per-channel matching produced lavender fins.

**Why not full Gray World?** Gray World assumes the average scene color is gray; underwater scenes
are legitimately blue/green, and the references keep a blue tone. Full strength gave magenta blotches in dark regions
(red gain too large where red ≈ 0). Hence exponent 0.5 and gains clipped to 0.7–1.5.

**Why a slope cap on the histogram LUT?** Where the input histogram is sparse (e.g. red), matching creates a
very steep LUT: it stretches noise and produces banding. Smoothing + capping prevents that.

**Is histogram specification allowed?** Yes: "Simple statistics of the given references (e.g. a target histogram)
may be used". It is our own implementation (CDF inversion), not a library function.

**Is using the references a form of overfitting?** The target histogram is a mean over the 10 given references and the
parameters are tuned on those 10 images, so given-image scores are slightly optimistic. The test images show real
generalization; report both. Mitigations: partial blending (`beta`), global parameters, few parameters.

**What does ablation show?** What each step contributes. Remove one step, re-measure; steps that don't help
should be justified or removed.

**Why do PSNR/SSIM mislead on UIEB?** The references are volunteers' picks among algorithm outputs, not captured ground
truth, so a visually pleasing result may differ strongly in pixels. ΔE, GMSD and visual inspection agreed better with
our impression (Q3.4 asks this).

**Which metrics are no-reference and how to use them?** noise, sharpness, colourful, entropy, UCIQE: compare with the
reference's value. UCIQE rewards contrast/chroma spread, so over-sharpened images can score well: don't use it alone.

**What failed along the way?** v1: gritty, magenta blotches, banding, JPEG blocks, and the statue (488) got worse than
its raw input. v3: colourfulness improved, but ΔE worsened (~+1). Be ready to explain each, with the numbers above.

---

## 8. Known caveats / things to double-check

- **CLAHE:** implemented with `cv2.createCLAHE`. The rules allow histogram functions but forbid "ready-made complete
  enhancement algorithms". CLAHE is listed as a technique in the textbook (Section 5.2.3), so it is probably fine, but
  be ready to explain it; we can write our own tile-based version if needed.
- **`GIVEN_REF` must exist at test time** (`data/uieb/reference` next to `q2.py`). If it is missing, the histogram step
  silently skips itself. Check that the test run actually finds the folder.
- Only spatial-domain methods, NumPy/OpenCV/Matplotlib are used: no Fourier, wavelet or learning-based method.
- v4 has not been evaluated on the real data yet; run `evaluate.py`, `ablate_q2.py`, `tune_q2.py` and update the tables above.
- Q3 reuses `enhance_underwater()` unchanged: do not edit `CFG` after Q1/Q3 results are produced.
- Fill in names and student IDs at the top of `q2.py`.
