"""
explain_query.py - Ket qua trung gian cua mot truy van (yeu cau 4b).

    python explain_query.py query.wav [--source csv] [--weights 1,1,1]

Ghi vao results/explain_<ten_file>/:
  vector_query.csv     71 gia tri: goc, mean/std cua CSDL, z-score, bo A/B/C
  top5.csv             5 file giong nhat + D_A, D_B, D_C, D, do tuong dong
  group_distances.png  khoang cach tung bo cua 5 ket qua
  z_profile.png        z-score cua truy van so voi file giong nhat
"""
import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audio_features import FEATURE_NAMES, GROUP_INDEX, GROUP_OF, GROUP_TITLES, GROUPS
from config import RESULTS_DIR, TOP_K
from search import SearchEngine, parse_weights

BLUE, ORANGE, GREEN, INK, GRID = "#2a78d6", "#eb6834", "#2f9e6b", "#222222", "#dddddd"


def explain(engine, query, weights=(1, 1, 1), k=TOP_K, out_dir=None):
    res, raw, zq = engine.search_file(query, k, weights)
    out_dir = Path(out_dir or RESULTS_DIR / f"explain_{Path(query).stem}")
    os.makedirs(out_dir, exist_ok=True)

    vec = pd.DataFrame({"feature": FEATURE_NAMES, "group": GROUP_OF, "raw": raw,
                        "db_mean": engine.mean, "db_std": engine.std, "z": zq})
    vec.to_csv(out_dir / "vector_query.csv", index=False)
    res.to_csv(out_dir / "top5.csv", index=False)

    # 1) khoang cach tung bo
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=150)
    x = np.arange(len(res))
    for j, (g, c) in enumerate(zip(GROUPS, (BLUE, ORANGE, GREEN))):
        ax.bar(x + (j - 1) * 0.26, res[f"D_{g}"], 0.26, color=c, label=f"Bo {g}: {GROUP_TITLES[g]}")
    ax.set_xticks(x, [f"#{r}\n{n}" for r, n in zip(res["rank"], res["file_name"])], fontsize=8)
    ax.set_ylabel("Khoang cach D_g")
    ax.set_ylim(0, max(res[["D_A", "D_B", "D_C"]].to_numpy().max() * 1.35, 0.1))
    ax.set_title("Khoang cach theo tung bo dac trung cua top " + str(len(res)), loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "group_distances.png")
    plt.close(fig)

    # 2) z-score truy van va file giong nhat
    top1 = engine.Z[engine.meta["audio_id"] == res.loc[0, "audio_id"]][0]
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    ax.plot(zq, color=BLUE, lw=1.2, label="Truy van")
    ax.plot(top1, color=ORANGE, lw=1.2, label=f"Top 1 ({res.loc[0, 'file_name']})")
    for g in GROUPS:
        ix = GROUP_INDEX[g]
        ax.axvspan(ix.min() - 0.5, ix.max() + 0.5, color=GRID, alpha=0.35 if g == "B" else 0.0)
        ax.text(ix.mean(), ax.get_ylim()[1], f"Bo {g}", ha="center", va="top", fontsize=8)
    ax.set_xlabel("Chi so chieu (0 den 70)")
    ax.set_ylabel("z-score")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "z_profile.png")
    plt.close(fig)
    return res, vec, out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--source", choices=["db", "csv"], default="db")
    ap.add_argument("--weights", type=parse_weights, default=(1, 1, 1))
    a = ap.parse_args()
    eng = SearchEngine.load(a.source)
    res, vec, out = explain(eng, a.query, a.weights)
    pd.set_option("display.width", 200)
    print("== Vector truy van, 8 chieu dau moi bo (goc va z-score) ==")
    for g in GROUPS:
        print(f"-- Bo {g} ({GROUP_TITLES[g]})")
        print(vec[vec["group"] == g].head(8).round(4).to_string(index=False))
    print("\n== Top ket qua ==")
    print(res.round(4).to_string(index=False))
    print("\nDa luu vao", out)
