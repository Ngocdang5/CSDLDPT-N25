"""
make_demo_data.py - Sinh am thanh TONG HOP de thu pipeline (KHONG phai du lieu that cua do an).

    python make_demo_data.py --out data/audio --n 60
Tao N file wav 30 giay (ten so) + labels.csv (6 'ho' am thanh) de kiem tra duong ong tu dau den cuoi.
"""
import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf

SR = 22050
FAMILIES = {  # ten: (f0 co so, so hoa am, bpm co so, muc nhieu)
    "tram_cham": (110, 8, 65, 0.002),
    "tram_nhanh": (110, 8, 130, 0.002),
    "trung_cham": (330, 4, 70, 0.004),
    "trung_nhanh": (330, 4, 140, 0.004),
    "cao_cham": (880, 2, 75, 0.01),
    "cao_nhanh": (880, 2, 150, 0.01),
}


def synth(f0, harm, bpm, noise, rng, dur=30):
    t = np.arange(SR * dur) / SR
    beat = 60.0 / bpm
    env = np.exp(-((t % beat) / (beat * 0.4)))
    notes = f0 * 2 ** (rng.choice([0, 2, 4, 7], size=int(dur / beat) + 1) / 12)
    f = notes[(t // beat).astype(int)]
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = sum((1.0 / h) * np.sin(h * ph) for h in range(1, harm + 1)) * env
    return (y + noise * rng.standard_normal(len(t))).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/audio")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default="labels.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    fams = list(FAMILIES)
    rows = []
    for i in range(a.n):
        fam = fams[i % len(fams)]
        f0, harm, bpm, noise = FAMILIES[fam]
        y = synth(f0 * rng.uniform(0.95, 1.05), harm, bpm * rng.uniform(0.95, 1.05),
                  noise, rng)
        name = f"{100000 + i * 37:06d}.wav"
        sf.write(os.path.join(a.out, name), y / np.abs(y).max() * 0.9, SR)
        rows.append((name, fam))
    pd.DataFrame(rows, columns=["file_name", "genre"]).to_csv(a.labels, index=False)
    print(f"Da tao {a.n} file trong {a.out} va {a.labels}")
