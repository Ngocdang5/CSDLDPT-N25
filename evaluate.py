"""
evaluate.py - Danh gia ket qua (yeu cau 4c).

    python evaluate.py --source csv --n 100

1) Tu truy hoi: cat doan 20 giay (giay 5 den 25) cua mot file trong CSDL lam truy van; do file goc
   co nam trong top 1 / top 5 khong (hit@1, hit@5, MRR). Day la phep thu "doan khac cua cung bai".
2) So sanh trong so cac bo: bang nhau, uu tien A/B/C, chi dung tung bo, Euclid 71 chieu.
3) Chi so tham chieu theo the loai (neu co nhan that, khong phai dap an tuyet doi):
   ty le top 5 cung the loai, leave-one-out tren tap database.
4) Thoi gian trich xuat va tim kiem moi truy van.
"""
import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audio_features import extract_features, load_audio, preprocess_array
from config import AUDIO_DIR, RESULTS_DIR, SR
from search import SearchEngine, combine, group_distances

VARIANTS = {
    "bang nhau (1/3, 1/3, 1/3)": (1, 1, 1),
    "uu tien A (0.5, 0.25, 0.25)": (2, 1, 1),
    "uu tien B (0.25, 0.5, 0.25)": (1, 2, 1),
    "uu tien C (0.25, 0.25, 0.5)": (1, 1, 2),
    "chi bo A": (1, 0, 0),
    "chi bo B": (0, 1, 0),
    "chi bo C": (0, 0, 1),
    "Euclid 71 chieu": None,
}


def dist(zq, Z, w):
    if w is None:
        return np.sqrt(((Z - zq) ** 2).sum(axis=1))
    return combine(group_distances(zq, Z), w)


def make_query_clip(y):
    """Doan 20 giay (giay 5 den 25); neu file ngan hon thi lay 2/3 o giua."""
    if len(y) >= 25 * SR:
        return y[5 * SR:25 * SR]
    n = len(y)
    return y[n // 6: n // 6 + (2 * n) // 3]


def self_retrieval(eng, n, seed, audio_dir):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eng.meta), size=min(n, len(eng.meta)), replace=False)
    qs, t_ext = [], []
    for i in idx:
        row = eng.meta.iloc[i]
        y = load_audio(os.path.join(audio_dir, row.file_path))
        t0 = time.time()
        _, zq = eng.embed(preprocess_array(make_query_clip(y)))
        t_ext.append(time.time() - t0)
        qs.append((i, zq))
    rows = []
    for name, w in VARIANTS.items():
        ranks = []
        for i, zq in qs:
            D = dist(zq, eng.Z, w)
            ranks.append(int((D < D[i]).sum()) + 1)
        r = np.array(ranks)
        rows.append({"phuong_an": name, "hit@1": (r <= 1).mean(), "hit@5": (r <= 5).mean(),
                     "MRR": (1.0 / r).mean(), "hang_trung_vi": float(np.median(r))})
    t_search = []
    for _, zq in qs[:50]:
        t0 = time.time()
        eng.search_vector(zq)
        t_search.append(time.time() - t0)
    return pd.DataFrame(rows), float(np.mean(t_ext)), float(np.mean(t_search)), len(qs)


def genre_precision(eng, w=(1, 1, 1)):
    g = eng.meta["genre"].fillna("unknown").to_numpy()
    known = np.array([x not in ("unknown", "") for x in g])
    if len(set(g[known])) < 2:
        return None
    hits, counts = [], []
    for i in np.where(known)[0]:
        D = combine(group_distances(eng.Z[i], eng.Z), w)
        D[i] = np.inf
        top = np.argsort(D)[:5]
        hits.append((g[top] == g[i]).mean())
    p = pd.Series(g[known]).value_counts(normalize=True)
    return float(np.mean(hits)), float((p ** 2).sum()), int(known.sum()), int(p.size)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["db", "csv"], default="db")
    ap.add_argument("--audio_dir", default=str(AUDIO_DIR))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    eng = SearchEngine.load(a.source)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tab, t_ext, t_search, nq = self_retrieval(eng, a.n, a.seed, a.audio_dir)
    tab.to_csv(RESULTS_DIR / "eval_self_retrieval.csv", index=False, float_format="%.4f")

    lines = [f"CSDL: {len(eng.meta)} file (tap database)",
             f"Tu truy hoi: {nq} truy van la doan 20 giay cua file trong CSDL",
             tab.round(3).to_string(index=False), "",
             f"Thoi gian trich xuat moi truy van : {t_ext:.2f}s",
             f"Thoi gian tim kiem moi truy van   : {t_search * 1000:.2f}ms"]
    gp = genre_precision(eng)
    if gp:
        lines += ["", f"Ty le top 5 cung the loai (tham chieu, {gp[2]} file, {gp[3]} the loai): "
                      f"{gp[0]:.3f} so voi ngau nhien {gp[1]:.3f}"]
    else:
        lines += ["", "Khong co nhan the loai (>= 2 nhan) nen bo qua chi so theo the loai."]
    txt = "\n".join(lines)
    (RESULTS_DIR / "eval_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    x = np.arange(len(tab))
    ax.bar(x - 0.2, tab["hit@1"], 0.4, color="#2a78d6", label="hit@1")
    ax.bar(x + 0.2, tab["hit@5"], 0.4, color="#eb6834", label="hit@5")
    ax.set_xticks(x, tab["phuong_an"], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Tu truy hoi theo phuong an trong so", loc="left")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eval_weights.png")
    print("\nDa luu results/eval_summary.txt, eval_self_retrieval.csv, eval_weights.png")
