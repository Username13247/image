"""Random search over the GLOBAL parameters of q2.py. Run from THE1/:  python tune_q2.py [n_trials]
Prints the best configurations; copy ONE of them into CFG in q2.py (same for all images).
Score = dE2000 + 10*(1-SSIM) + 0.5*|noise-ref| + 0.02*|sharp-ref| + 0.02*|colourful-ref|  (lower = better).
Edit W below to change the trade-off. Note: the target histogram in q2.py is built from the same
10 references, so scores are slightly optimistic; the test images tell the real story."""
import os, sys, copy, random
import cv2, numpy as np
import evaluate as ev
import q2

RAW, REF = "data/uieb/raw", "data/uieb/reference"
W = dict(ssim=10.0, noise=0.5, sharp=0.02, colour=0.02)
SPACE = dict(
    alpha=[0.0, 0.5, 1.0, 1.5],
    wb_strength=[0.0, 0.3, 0.5, 0.8, 1.0],
    stretch_strength=[0.0, 0.3, 0.5, 0.8, 1.0],
    beta=[0.2, 0.4, 0.6, 0.8],
    clahe_clip=[1.0, 1.5, 2.0],
    chroma_gain=[1.0, 1.15, 1.25, 1.4],
    hist_mode=["L", "rgb"],
)
names = sorted(n for n in os.listdir(REF) if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
raws = [cv2.imread(os.path.join(RAW, n)) for n in names]
refs = [cv2.cvtColor(cv2.imread(os.path.join(REF, n)), cv2.COLOR_BGR2RGB) for n in names]
ref_nr = {k: np.mean([f(r) for r in refs]) for k, f in ev.NO_REFERENCE.items()}
base = copy.deepcopy(q2.CFG)

def evaluate_cfg(cfg):
    q2.CFG.clear(); q2.CFG.update(cfg)
    ms = [ev.all_metrics(cv2.cvtColor(q2.enhance_underwater(im), cv2.COLOR_BGR2RGB), rf)
          for im, rf in zip(raws, refs)]
    m = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
    m["score"] = (m["dE2000"] + W["ssim"] * (1 - m["SSIM"]) + W["noise"] * abs(m["noise"] - ref_nr["noise"])
                  + W["sharp"] * abs(m["sharpness"] - ref_nr["sharpness"])
                  + W["colour"] * abs(m["colourful"] - ref_nr["colourful"]))
    return m

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
random.seed(0)
results = [(evaluate_cfg(base), {})]     # current defaults first
for t in range(n):
    over = {k: random.choice(v) for k, v in SPACE.items()}
    cfg = copy.deepcopy(base); cfg.update(over)
    results.append((evaluate_cfg(cfg), over))
    print(f"trial {t + 1}/{n}  score={results[-1][0]['score']:.3f}", flush=True)
results.sort(key=lambda r: r[0]["score"])
print("\n=== best 5 (the empty {} is the current default) ===")
for m, over in results[:5]:
    print(f"score={m['score']:.3f} PSNR={m['PSNR']:.2f} SSIM={m['SSIM']:.3f} dE={m['dE2000']:.2f} "
          f"GMSD={m['GMSD']:.4f} noise={m['noise']:.2f} sharp={m['sharpness']:.1f} colourful={m['colourful']:.1f}")
    print("   ", over)
