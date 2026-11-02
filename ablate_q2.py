"""Ablation / baseline / parameter sweep for Q2. Run from THE1/:  python ablate_q2.py
Writes nothing to disk except nothing; prints mean metrics per variant (copy into report tables)."""
import os, cv2, numpy as np
import evaluate as ev
import q2

RAW, REF = "data/uieb/raw", "data/uieb/reference"
names = sorted(n for n in os.listdir(REF) if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
raws = [cv2.imread(os.path.join(RAW, n)) for n in names]
refs = [cv2.cvtColor(cv2.imread(os.path.join(REF, n)), cv2.COLOR_BGR2RGB) for n in names]

def score(fn):
    ms = []
    for im, rf in zip(raws, refs):
        out = cv2.cvtColor(fn(im), cv2.COLOR_BGR2RGB)
        ms.append(ev.all_metrics(out, rf))
    return {k: np.mean([m[k] for m in ms]) for k in ms[0]}

def show(label, m):
    print(f"{label:<22}" + "".join(f"{k}={v:8.3f} " for k, v in m.items()))

show("input (no enhance)", score(lambda x: x))
show("baseline GrayWorld", score(lambda x: (q2.gray_world(x.astype(np.float32)/255, .5, 2)*255).astype(np.uint8)))
show("baseline CLAHE(L)", score(lambda x: q2.clahe_lightness(x, 2.0, 8)))
show("FULL pipeline", score(q2.enhance_underwater))
for s in q2.STEPS:
    show(f"without {s}", score(lambda x, s=s: q2.enhance_underwater(x, skip=(s,))))

# restore after sweeps
print("\n--- sweeps (fixed globally, pick ONE value for all images) ---")
for beta in (0.0, 0.2, 0.4, 0.6, 1.0):
    q2.CFG["beta"] = beta; show(f"beta={beta}", score(q2.enhance_underwater))
q2.CFG["beta"] = 0.4
for clip in (1.0, 1.5, 2.0, 3.0):
    q2.CFG["clahe_clip"] = clip; show(f"clahe_clip={clip}", score(q2.enhance_underwater))
q2.CFG["clahe_clip"] = 1.5
for a in (0.0, 0.5, 1.0, 1.5):
    q2.CFG["alpha"] = a; show(f"alpha={a}", score(q2.enhance_underwater))
q2.CFG["alpha"] = 1.0
for w in (0.0, 0.5, 1.0):
    q2.CFG["wb_strength"] = w; show(f"wb_strength={w}", score(q2.enhance_underwater))
q2.CFG["wb_strength"] = 0.5
for sl in (1.5, 3.0, 6.0):
    q2.CFG["max_slope"] = sl; show(f"max_slope={sl}", score(q2.enhance_underwater))
q2.CFG["max_slope"] = 3.0
for g in (1.0, 1.25, 1.5, 1.75):
    q2.CFG["chroma_gain"] = g; show(f"chroma_gain={g}", score(q2.enhance_underwater))
q2.CFG["chroma_gain"] = 1.25
for md in ("L", "rgb"):
    q2.CFG["hist_mode"] = md; show(f"hist_mode={md}", score(q2.enhance_underwater))
q2.CFG["hist_mode"] = "L"