"""
evaluate.py - Danh gia ket qua (yeu cau 4c).

    python evaluate.py --source csv --n 100
    python evaluate.py --source csv --n 100 --query_dir data/test_queries/

Hai che do danh gia:
A) Tu truy hoi (mac dinh):
   Cat doan 20 giay (giay 5-25) cua file trong CSDL lam truy van; do file goc co nam trong
   top 1 / top 5 khong (hit@1, hit@5, MRR). So sanh 8 phuong an trong so.

B) Truy van moi ngoai dataset (--query_dir hoac split == 'test' tu metadata.csv):
   - Trich xuat vector cho file truy van moi (do thoi gian).
   - Tim top 5 trong CSDL.
   - Phan biet khoang cach D (mean, std, min, max) va tuong dong.
   - Neu co nhan genre: do ty le top-5 cung the loai (genre-precision@5).
"""
import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audio_features import FEATURE_NAMES, extract_features, load_audio, preprocess_array
from config import AUDIO_DIR, METADATA_CSV, RESULTS_DIR, SR
from search import SearchEngine, combine, group_distances, normalize_weights

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

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


# ---------------------------------------------------------------- utilities
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


def _find_audio_files(directory):
    out = []
    for dp, _, fns in os.walk(directory):
        for fn in sorted(fns):
            if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                out.append(os.path.join(dp, fn))
    return out


# ===================================================================== A: tu truy hoi
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


def genre_precision_db(eng, w=(1, 1, 1)):
    """Leave-one-out genre precision tren tap database (chi so tham chieu)."""
    g = eng.meta["genre"].fillna("unknown").to_numpy()
    known = np.array([x not in ("unknown", "") for x in g])
    if len(set(g[known])) < 2:
        return None
    hits = []
    for i in np.where(known)[0]:
        D = combine(group_distances(eng.Z[i], eng.Z), w)
        D[i] = np.inf
        top = np.argsort(D)[:5]
        hits.append((g[top] == g[i]).mean())
    p = pd.Series(g[known]).value_counts(normalize=True)
    return float(np.mean(hits)), float((p ** 2).sum()), int(known.sum()), int(p.size)


# ===================================================================== B: truy van moi
def _embed_file(eng, path):
    """Trich xuat + chuan hoa mot file. Tra ve (zq, t_extract) hoac nem loi."""
    t0 = time.time()
    raw = eng.embed(preprocess_array(load_audio(path)))[1]
    return raw, time.time() - t0


def _collect_external_queries(query_dir, metadata_csv, audio_dir):
    """
    Thu thap danh sach (duong_dan, genre | None) tu:
      - query_dir neu duoc truyen vao; hoac
      - split == 'test' trong metadata_csv.
    Tra ve list cua (path, genre_or_None).
    """
    if query_dir:
        paths = _find_audio_files(query_dir)
        if not paths:
            raise SystemExit(f"Khong tim thay file am thanh nao trong {query_dir}")
        return [(p, None) for p in paths]

    # Dung tap test trong metadata.csv
    if not os.path.exists(metadata_csv):
        raise SystemExit(f"Khong tim thay {metadata_csv}; chay build_metadata.py truoc")
    meta = pd.read_csv(metadata_csv)
    test = meta[meta["split"] == "test"]
    if test.empty:
        raise SystemExit("metadata.csv khong co file nao voi split='test'. "
                         "Chay lai build_metadata.py --test_ratio > 0, "
                         "hoac truyen --query_dir de chi dinh thu muc khac.")
    rows = []
    for r in test.itertuples(index=False):
        p = os.path.join(audio_dir, r.file_path)
        genre = r.genre if hasattr(r, "genre") and r.genre not in ("unknown", "", None) else None
        rows.append((p, genre))
    return rows


def external_query_eval(eng, query_dir, metadata_csv, audio_dir, n=None, seed=42):
    """
    Danh gia tren tap truy van moi ngoai CSDL.

    Returns
    -------
    summary_lines : list[str]
    detail_df     : DataFrame (mot dong moi truy van)
    t_ext_mean    : float (giay)
    t_search_mean : float (giay)
    n_ok          : int
    """
    queries = _collect_external_queries(query_dir, metadata_csv, audio_dir)
    if n and n < len(queries):
        rng = np.random.default_rng(seed)
        queries = [queries[i] for i in rng.choice(len(queries), n, replace=False)]

    db_genres = eng.meta["genre"].fillna("unknown").to_numpy()

    rows, t_exts, t_searches, errors = [], [], [], []
    for path, q_genre in queries:
        try:
            zq, t_ext = _embed_file(eng, path)
        except Exception as e:
            errors.append(f"{os.path.basename(path)}: {e}")
            continue

        t0 = time.time()
        res = eng.search_vector(zq, k=5)
        t_search = time.time() - t0

        D_all = dist(zq, eng.Z, (1, 1, 1))

        top_genres = db_genres[res.index.map(lambda r: r)]  # dung iloc order
        # Note: res.index da reset (0..4), cot actual la loc qua iloc
        top_genres_arr = eng.meta["genre"].fillna("unknown").iloc[
            [eng.meta.index.get_loc(i) if i in eng.meta.index else i
             for i in range(len(eng.meta))]
        ].to_numpy()
        top_genres_res = top_genres_arr[
            np.argsort(dist(zq, eng.Z, (1, 1, 1)), kind="stable")[:5]
        ]

        gp5 = None
        if q_genre and q_genre not in ("unknown", ""):
            gp5 = float((top_genres_res == q_genre).mean())

        rows.append({
            "file": os.path.basename(path),
            "genre_query": q_genre or "unknown",
            "top1_file": res.iloc[0]["file_name"],
            "top1_dist": float(res.iloc[0]["distance"]),
            "top1_sim": float(res.iloc[0]["similarity"]),
            "mean_D_top5": float(res["distance"].mean()),
            "mean_D_all": float(D_all.mean()),
            "genre_precision@5": gp5,
            "t_extract_s": t_ext,
            "t_search_ms": t_search * 1000,
        })
        t_exts.append(t_ext)
        t_searches.append(t_search)

    if not rows:
        return ["[external_query] Tat ca file deu gap loi khi trich xuat."], pd.DataFrame(), 0, 0, 0

    df = pd.DataFrame(rows)

    lines = [
        "=== B. Danh gia tren tap truy van moi ngoai CSDL ===",
        f"So file truy van : {len(df)} / {len(queries)} thanh cong",
    ]
    if errors:
        lines.append(f"File loi ({len(errors)}): " + "; ".join(errors[:5])
                     + ("..." if len(errors) > 5 else ""))

    lines += [
        f"Khoang cach top-1 : mean={df['top1_dist'].mean():.3f}  "
        f"std={df['top1_dist'].std():.3f}  "
        f"min={df['top1_dist'].min():.3f}  max={df['top1_dist'].max():.3f}",
        f"Tuong dong top-1  : mean={df['top1_sim'].mean():.3f}  "
        f"std={df['top1_sim'].std():.3f}",
        f"Thoi gian trich xuat : {np.mean(t_exts):.2f}s/file",
        f"Thoi gian tim kiem   : {np.mean(t_searches) * 1000:.2f}ms/file",
    ]

    gp_vals = df["genre_precision@5"].dropna()
    if len(gp_vals) > 0:
        lines.append(
            f"Genre-precision@5 (tren {len(gp_vals)} file co nhan): "
            f"mean={gp_vals.mean():.3f}  std={gp_vals.std():.3f}"
        )
    else:
        lines.append("Khong co nhan genre cho file truy van -> bo qua genre-precision@5")

    return lines, df, float(np.mean(t_exts)), float(np.mean(t_searches)), len(df)


# ===================================================================== plots
def _save_weights_chart(tab, out_path):
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
    fig.savefig(out_path)
    plt.close(fig)


def _save_distance_hist(df, out_path, title="Phan bo khoang cach top-1 (truy van moi)"):
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
    bins = min(30, max(5, int(np.sqrt(len(df)) * 1.5)))
    ax.hist(df["top1_dist"], bins=bins, color="#2a78d6", edgecolor="#fcfcfb", linewidth=1.5)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("Khoang cach D top-1")
    ax.set_ylabel("So file truy van")
    ax.yaxis.grid(True, color="#e4e3df")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ===================================================================== main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Danh gia he thong tim kiem am nhac. "
                    "Mac dinh: tu truy hoi. "
                    "Truyen --query_dir hoac --external de danh gia tren file moi."
    )
    ap.add_argument("--source", choices=["db", "csv"], default="db")
    ap.add_argument("--audio_dir", default=str(AUDIO_DIR))
    ap.add_argument("--metadata", default=str(METADATA_CSV))
    ap.add_argument("--n", type=int, default=100,
                    help="So file lay mau (tu truy hoi va external)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query_dir", default=None,
                    help="Thu muc chua file am thanh truy van moi (ngoai dataset). "
                         "Neu bo qua, dung split='test' tu metadata.csv (neu co).")
    ap.add_argument("--external", action="store_true",
                    help="Bat danh gia tren file truy van ngoai (tu --query_dir hoac split=test)")
    ap.add_argument("--self", dest="run_self", action="store_true", default=None,
                    help="Chi chay danh gia tu truy hoi (mac dinh: chay ca hai neu --external)")
    a = ap.parse_args()

    # Neu khong chi dinh gi, mac dinh chi chay tu truy hoi
    run_self = True
    run_ext = a.external or (a.query_dir is not None)

    eng = SearchEngine.load(a.source)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_lines = [f"CSDL: {len(eng.meta)} file (tap database)", ""]

    # --- A: tu truy hoi ---
    if run_self:
        tab, t_ext, t_search, nq = self_retrieval(eng, a.n, a.seed, a.audio_dir)
        tab.to_csv(RESULTS_DIR / "eval_self_retrieval.csv", index=False, float_format="%.4f")

        self_lines = [
            "=== A. Tu truy hoi (doan 20 giay cua file trong CSDL) ===",
            f"So truy van: {nq}",
            tab.round(3).to_string(index=False), "",
            f"Thoi gian trich xuat moi truy van : {t_ext:.2f}s",
            f"Thoi gian tim kiem moi truy van   : {t_search * 1000:.2f}ms",
        ]
        gp = genre_precision_db(eng)
        if gp:
            self_lines += ["",
                           f"Leave-one-out genre-precision@5 (tren {gp[2]} file, {gp[3]} the loai): "
                           f"{gp[0]:.3f}  (baseline ngau nhien {gp[1]:.3f})"]
        else:
            self_lines += ["", "Khong co nhan the loai (>= 2 nhan) nen bo qua chi so genre."]

        all_lines += self_lines
        _save_weights_chart(tab, RESULTS_DIR / "eval_weights.png")

    # --- B: file truy van moi ngoai dataset ---
    if run_ext:
        all_lines.append("")
        ext_lines, ext_df, t_ext2, t_srch2, n_ok = external_query_eval(
            eng, a.query_dir, a.metadata, a.audio_dir, a.n, a.seed
        )
        all_lines += ext_lines
        if not ext_df.empty:
            ext_df.to_csv(RESULTS_DIR / "eval_external_queries.csv", index=False, float_format="%.4f")
            _save_distance_hist(ext_df, RESULTS_DIR / "eval_external_dist.png")

    if not run_self and not run_ext:
        all_lines.append("(Khong chay danh gia nao. Them --self hoac --external)")

    txt = "\n".join(all_lines)
    (RESULTS_DIR / "eval_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print("\nDa luu results/eval_summary.txt")
    if run_self:
        print("  eval_self_retrieval.csv, eval_weights.png")
    if run_ext:
        print("  eval_external_queries.csv, eval_external_dist.png")
