"""
evaluate.py - Danh gia he thong tim kiem am nhac (yeu cau 4c).

Ba che do chay:
  python evaluate.py --n 50                          # Chi tu truy hoi (mac dinh)
  python evaluate.py --query_dir data/queries/       # Chi danh gia file moi ngoai CSDL
  python evaluate.py --self --query_dir data/queries # Chay ca hai

Bon nhom thuc nghiem:
  E1. Tu truy hoi       : Hit@1, Hit@5, MRR (kiem tra tinh dung dan cua he thong)
  E2. Truy van nhac moi : Top-5 day du, khoang cach, tuong dong va thoi gian xu ly
  E3. So sanh trong so  : 8 phuong an A/B/C, so sanh tren cung tap truy van
  E4. Hieu nang         : Thoi gian tung cong doan (perf_counter)
"""
import argparse
import os
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audio_features import FEATURE_NAMES, GROUPS, extract_features, load_audio, preprocess_array
from config import AUDIO_DIR, METADATA_CSV, SR
from scaling import transform
from search import SearchEngine, combine, group_distances, normalize_weights

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}

# 8 phuong an trong so cho E1 / E3
VARIANTS = {
    "bang nhau (1/3,1/3,1/3)": (1, 1, 1),
    "uu tien A (0.6,0.2,0.2)": (3, 1, 1),
    "uu tien B (0.2,0.6,0.2)": (1, 3, 1),
    "uu tien C (0.2,0.2,0.6)": (1, 1, 3),
    "chi bo A":                 (1, 0, 0),
    "chi bo B":                 (0, 1, 0),
    "chi bo C":                 (0, 0, 1),
    "Euclid 71 chieu":          None,
}


# ================================================================== utilities
def _dist(zq, Z, w):
    """Tinh vector khoang cach D cho moi hang trong Z."""
    if w is None:
        return np.sqrt(((Z - zq) ** 2).sum(axis=1))
    return combine(group_distances(zq, Z), w)


def _rank_of(zq, Z, i, w):
    """
    Hang thuc te cua file thu i khi dung trong so w.
    Dua vao thu tu argsort that su (tranh lech khi dong hang).
    """
    D = _dist(zq, Z, w)
    order = np.argsort(D, kind="stable")
    return int(np.where(order == i)[0][0]) + 1


def _make_clip(y):
    """Doan 20 giay (5-25s); file ngan hon lay 2/3 o giua."""
    if len(y) >= 25 * SR:
        return y[5 * SR:25 * SR]
    n = len(y)
    return y[n // 6: n // 6 + (2 * n) // 3]


def _find_audio(directory):
    out = []
    for dp, _, fns in os.walk(directory):
        for fn in sorted(fns):
            if os.path.splitext(fn)[1].lower() in AUDIO_EXTS:
                out.append(os.path.join(dp, fn))
    return out


def _profile_query(eng, y_raw):
    """Do thoi gian tung cong doan (ms) cho mot truy van da load."""
    t0 = perf_counter()
    y_proc = preprocess_array(y_raw)
    t_prep = (perf_counter() - t0) * 1000

    t0 = perf_counter()
    raw = extract_features(y_proc)
    t_ext = (perf_counter() - t0) * 1000

    t0 = perf_counter()
    zq = transform(raw, eng.mean, eng.std)
    t_scale = (perf_counter() - t0) * 1000

    t0 = perf_counter()
    eng.search_vector(zq)
    t_search = (perf_counter() - t0) * 1000

    return zq, {
        "preprocess_ms": t_prep,
        "extract_ms":    t_ext,
        "scale_ms":      t_scale,
        "search_ms":     t_search,
        "total_ms":      t_prep + t_ext + t_scale + t_search,
    }


def summarize_performance(perf_rows, db_load_ms):
    """
    E4: Tong hop thoi gian tung cong doan (ms): mean, median, std, p95.
    Tach rieng truy van dau tien (warm-up) neu co >= 2 truy van.
    """
    if not perf_rows:
        return [], pd.DataFrame()

    df = pd.DataFrame(perf_rows)
    first_ms = float(df.iloc[0]["total_ms"])
    steady = df.iloc[1:] if len(df) > 1 else df

    stages = [
        ("load_preprocess_ms", "Doc + tien xu ly"),
        ("extract_ms",         "Trich xuat 71 dac trung"),
        ("scale_ms",           "Chuan hoa Z-score"),
        ("search_ms",          "Tinh khoang cach + xep hang"),
        ("total_ms",           "Tong thoi gian truy van"),
    ]
    rows = []
    for col, label in stages:
        if col not in steady.columns:
            continue
        vals = steady[col].to_numpy(dtype=float)
        rows.append({
            "cong_doan": label,
            "mean_ms":   float(np.mean(vals)),
            "median_ms": float(np.median(vals)),
            "std_ms":    float(np.std(vals)),
            "p95_ms":    float(np.percentile(vals, 95)),
        })

    perf_df = pd.DataFrame(rows)
    lines = [
        "=== E4. Hieu nang tung cong doan (ms) ===",
        f"Khoi tao CSDL (do 1 lan)    : {db_load_ms:.2f} ms",
        f"Truy van dau tien (warm-up) : {first_ms:.1f} ms | Do on dinh tren {len(steady)} truy van:",
        perf_df.round(2).to_string(index=False),
    ]
    return lines, perf_df


# ================================================================== E1: tu truy hoi
def _run_variants(eng, qs):
    """
    Tinh Hit@1, Hit@5, MRR, hang trung vi cho 8 phuong an trong so.
    qs: list[(db_index, zq_vector)]
    """
    rows = []
    for name, w in VARIANTS.items():
        ranks = [_rank_of(zq, eng.Z, i, w) for i, zq in qs]
        r = np.array(ranks)
        rows.append({
            "phuong_an":     name,
            "hit@1":         float((r <= 1).mean()),
            "hit@5":         float((r <= 5).mean()),
            "MRR":           float((1.0 / r).mean()),
            "hang_trung_vi": float(np.median(r)),
        })
    return pd.DataFrame(rows)


def exact_self_retrieval(eng, n, seed, audio_dir):
    """
    E1.1 — File nguyen ban: tien xu ly chuan (cat 30s giua, chuan hoa dinh),
    trich xuat dac trung roi truy van lai chinh minh trong CSDL.
    Hit@1 ky vong ~100% neu pipeline nhat quan.

    Returns: (DataFrame variants, t_ext_mean_s, t_search_mean_ms, n_queries)
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eng.meta), size=min(n, len(eng.meta)), replace=False)

    qs, perf_rows = [], []
    for i in idx:
        row = eng.meta.iloc[i]
        t0 = perf_counter()
        y = load_audio(os.path.join(audio_dir, row.file_path))
        t_load = (perf_counter() - t0) * 1000
        zq, timing = _profile_query(eng, y)
        timing["load_preprocess_ms"] = t_load + timing.pop("preprocess_ms")
        timing["total_ms"] += t_load
        perf_rows.append(timing)
        qs.append((int(i), zq))

    t_ext_s = float(np.mean([r["extract_ms"] for r in perf_rows])) / 1000.0
    t_srch_ms = float(np.mean([r["search_ms"] for r in perf_rows]))

    return (
        _run_variants(eng, qs),
        t_ext_s,
        t_srch_ms,
        len(qs),
        perf_rows,
    )


def segment_self_retrieval(eng, n, seed, audio_dir):
    """
    E1.2 — Doan cat 20 giay (giay 5-25): truy van bang doan am thanh ngan hon file goc.
    Kiem tra do on dinh cua dac trung khi khong co day du noi dung.

    Returns: (DataFrame variants, t_ext_mean_s, t_search_mean_ms, n_queries)
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eng.meta), size=min(n, len(eng.meta)), replace=False)

    qs, t_exts = [], []
    for i in idx:
        row = eng.meta.iloc[i]
        y = load_audio(os.path.join(audio_dir, row.file_path))
        t0 = perf_counter()
        _, zq = eng.embed(preprocess_array(_make_clip(y)))
        t_exts.append(perf_counter() - t0)
        qs.append((int(i), zq))

    t_searches = []
    for _, zq in qs[:50]:
        t0 = perf_counter()
        eng.search_vector(zq)
        t_searches.append(perf_counter() - t0)

    return (
        _run_variants(eng, qs),
        float(np.mean(t_exts)),
        float(np.mean(t_searches)) * 1000,
        len(qs),
    )


# ================================================================== E2: file nhac moi ngoai CSDL
def _collect_queries(query_dir, metadata_csv, audio_dir):
    """
    Lay danh sach (path, genre | None):
      - Neu query_dir: quet thu muc do.
      - Neu khong:     dung split='test' tu metadata_csv.
    """
    if query_dir:
        paths = _find_audio(query_dir)
        if not paths:
            raise SystemExit(f"Khong tim thay file am thanh nao trong {query_dir}")
        return [(p, None) for p in paths]

    if not os.path.exists(metadata_csv):
        raise SystemExit(f"Khong tim thay {metadata_csv}; chay build_metadata.py truoc")
    meta = pd.read_csv(metadata_csv)
    test_rows = meta[meta["split"] == "test"]
    if test_rows.empty:
        raise SystemExit(
            "metadata.csv khong co file nao voi split='test'.\n"
            "  Chay lai build_metadata.py --test_ratio > 0, hoac\n"
            "  truyen --query_dir de chi dinh thu muc chua file truy van moi."
        )
    result = []
    for r in test_rows.itertuples(index=False):
        p = os.path.join(audio_dir, r.file_path)
        g = getattr(r, "genre", None)
        genre = g if g and g not in ("unknown", "") else None
        result.append((p, genre))
    return result


def external_query_eval(eng, query_dir, metadata_csv, audio_dir, n=None, seed=42):
    """
    E2: Danh gia tren tap truy van moi ngoai CSDL.

    Luu y:
      - Moi truy van tra ve top-5 day du (eval_external_top5.csv).
      - Khong tinh lai argsort/khoang cach du thua: lay truc tiep tu res.

    Returns
    -------
    summary_lines : list[str]
    summary_df    : DataFrame (1 dong / truy van, tom tat)
    top5_df       : DataFrame (5 dong / truy van, day du)
    t_ext_mean_s  : float
    t_search_mean_ms : float
    n_ok          : int
    """
    queries = _collect_queries(query_dir, metadata_csv, audio_dir)
    if n and n < len(queries):
        rng = np.random.default_rng(seed)
        queries = [queries[i] for i in rng.choice(len(queries), n, replace=False)]

    summary_rows, top5_rows, t_exts, t_searches, errors = [], [], [], [], []

    for path, q_genre in queries:
        fname = os.path.basename(path)
        # Doc, tien xu ly, trich xuat, chuan hoa
        try:
            t_all0 = perf_counter()
            y_proc = preprocess_array(load_audio(path))
            t_ext0 = perf_counter()
            raw = extract_features(y_proc)
            t_ext = perf_counter() - t_ext0
            zq = transform(raw, eng.mean, eng.std)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fname}: {e}")
            continue

        # Tim kiem top 5
        t0 = perf_counter()
        res = eng.search_vector(zq, k=5)
        t_search = perf_counter() - t0
        t_total_ms = (perf_counter() - t_all0) * 1000

        # Genre ket qua: lay truc tiep tu res (khong argsort lai)
        top_genres = res["genre"].fillna("unknown").to_numpy()
        gp5 = None
        if q_genre and q_genre not in ("unknown", ""):
            gp5 = float((top_genres == q_genre).mean())

        # Tom tat moi truy van
        summary_rows.append({
            "query_file":        fname,
            "genre_query":       q_genre or "unknown",
            "top1_file":         res.iloc[0]["file_name"],
            "top1_dist":         float(res.iloc[0]["distance"]),
            "top1_sim":          float(res.iloc[0]["similarity"]),
            "mean_D_top5":       float(res["distance"].mean()),
            "mean_sim_top5":     float(res["similarity"].mean()),
            "genre_precision@5": gp5,
            "t_extract_s":       round(t_ext, 4),
            "t_search_ms":       round(t_search * 1000, 3),
            "t_total_ms":        round(t_total_ms, 2),
        })

        # Top-5 day du
        for _, row in res.iterrows():
            top5_rows.append({
                "query_file":  fname,
                "rank":        int(row["rank"]),
                "result_file": row["file_name"],
                "file_path":   row["file_path"],
                "distance":    float(row["distance"]),
                "similarity":  float(row["similarity"]),
                "D_A":         float(row.get("D_A", np.nan)),
                "D_B":         float(row.get("D_B", np.nan)),
                "D_C":         float(row.get("D_C", np.nan)),
                "genre":       row.get("genre", "unknown") or "unknown",
            })

        t_exts.append(t_ext)
        t_searches.append(t_search)

    if not summary_rows:
        return (["[E2] Tat ca file deu gap loi khi trich xuat."],
                pd.DataFrame(), pd.DataFrame(), 0.0, 0.0, 0)

    summary_df = pd.DataFrame(summary_rows)
    top5_df = pd.DataFrame(top5_rows)

    lines = [
        "=== E2. Danh gia tren tap truy van moi ngoai CSDL ===",
        f"So file truy van : {len(summary_df)} / {len(queries)} thanh cong",
    ]
    if errors:
        lines.append("File loi: " + "; ".join(errors[:5]) + ("..." if len(errors) > 5 else ""))

    t_ext_mean = float(np.mean(t_exts))
    t_srch_mean = float(np.mean(t_searches)) * 1000  # ms
    totals = summary_df["t_total_ms"].to_numpy(dtype=float)
    lines += [
        f"Khoang cach top-1 : mean={summary_df['top1_dist'].mean():.3f}"
        f"  std={summary_df['top1_dist'].std():.3f}"
        f"  min={summary_df['top1_dist'].min():.3f}"
        f"  max={summary_df['top1_dist'].max():.3f}",
        f"Tuong dong top-1   : mean={summary_df['top1_sim'].mean():.3f}"
        f"  std={summary_df['top1_sim'].std():.3f}",
        f"Thoi gian trich xuat    : {t_ext_mean:.2f} s/file",
        f"Thoi gian tim kiem      : {t_srch_mean:.2f} ms/file",
        f"Tong thoi gian/truy van : mean={np.mean(totals):.1f} ms"
        f"  median={np.median(totals):.1f} ms"
        f"  P95={np.percentile(totals, 95):.1f} ms",
    ]
    gp_vals = summary_df["genre_precision@5"].dropna()
    if len(gp_vals) > 0:
        lines.append(
            f"Genre-precision@5 ({len(gp_vals)} file co nhan): "
            f"mean={gp_vals.mean():.3f}  std={gp_vals.std():.3f}"
        )
    else:
        lines.append("(Khong co nhan genre -> bo qua genre-precision@5)")

    return lines, summary_df, top5_df, t_ext_mean, t_srch_mean, len(summary_df)


# ================================================================== E1 genre (tuy chon --genre)
def genre_precision_db(eng, w=(1, 1, 1)):
    """Leave-one-out genre-precision@5 (chi so phu, chay khi --genre)."""
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


# ================================================================== charts
def _chart_weights(tab, out_path):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    x = np.arange(len(tab))
    ax.bar(x - 0.2, tab["hit@1"], 0.4, color="#2a78d6", label="hit@1")
    ax.bar(x + 0.2, tab["hit@5"], 0.4, color="#eb6834", label="hit@5")
    ax.set_xticks(x, tab["phuong_an"], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("E1/E3: Tu truy hoi theo phuong an trong so", loc="left", fontsize=10)
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#dddddd")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _chart_dist_hist(summary_df, out_path):
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
    bins = min(30, max(5, int(np.sqrt(len(summary_df)) * 1.5)))
    ax.hist(summary_df["top1_dist"], bins=bins,
            color="#2a78d6", edgecolor="#fcfcfb", linewidth=1.5)
    ax.set_title("E2: Phan bo khoang cach top-1 (file truy van moi)", loc="left", fontsize=10)
    ax.set_xlabel("Khoang cach D top-1")
    ax.set_ylabel("So file truy van")
    ax.yaxis.grid(True, color="#e4e3df")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ================================================================== main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Danh gia he thong tim kiem am nhac (4 nhom E1-E4).\n"
            "  Mac dinh (khong co --query_dir / --external): chi chay E1 tu truy hoi.\n"
            "  --query_dir / --external: chi chay E2 file moi.\n"
            "  --self --query_dir ...: chay ca E1 lan E2."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source", choices=["db", "csv"], default="db",
                    help="Nguon du lieu: db (PostgreSQL) hoac csv")
    ap.add_argument("--audio_dir", default=str(AUDIO_DIR),
                    help="Thu muc chua file am thanh cua CSDL")
    ap.add_argument("--metadata", default=str(METADATA_CSV),
                    help="Duong dan toi metadata.csv")
    ap.add_argument("--n", type=int, default=50,
                    help="So file lay mau cho E1 va E2 (mac dinh 50)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query_dir", default=None,
                    help="[E2] Thu muc chua file nhac moi ngoai CSDL (vi du data/queries/)")
    ap.add_argument("--external", action="store_true",
                    help="[E2] Bat danh gia file moi (tu --query_dir hoac split='test')")
    ap.add_argument("--self", dest="run_self", action="store_true", default=False,
                    help="Ep chay E1 tu truy hoi du co --query_dir hay khong")
    ap.add_argument("--genre", action="store_true",
                    help="[Tuy chon] Tinh leave-one-out genre-precision@5 sau E1")
    a = ap.parse_args()
    source_dir = "postgresql" if a.source == "db" else "csv"
    results_dir = Path("results") / source_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Xac dinh che do chay ---
    run_ext  = a.external or (a.query_dir is not None)
    run_self = bool(a.run_self) or not run_ext   # mac dinh chi tu truy hoi khi khong co ext

    t0 = perf_counter()
    eng = SearchEngine.load(a.source)
    db_load_ms = (perf_counter() - t0) * 1000

    all_lines = [
        f"CSDL      : {len(eng.meta)} file (tap database)",
        f"Nguon     : {'PostgreSQL' if a.source == 'db' else 'CSV'}",
    ]
    if run_self:
        all_lines.append(f"n_sample  : {a.n}  seed={a.seed}")
    all_lines.append("")

    perf_rows_all = []

    # ---- E1: Tu truy hoi ------------------------------------------------
    if run_self:
        # E1.1: file nguyen ban (tien xu ly chuan 30s)
        tab_exact, t_ext_exact, t_srch_exact, nq_exact, perf_rows_exact = exact_self_retrieval(
            eng, a.n, a.seed, a.audio_dir)
        perf_rows_all.extend(perf_rows_exact)
        tab_exact.to_csv(results_dir / "eval_exact_retrieval.csv",
                         index=False, float_format="%.4f")

        # E1.2: doan cat 20 giay (giay 5-25)
        tab_seg, t_ext_seg, t_srch_seg, nq_seg = segment_self_retrieval(
            eng, a.n, a.seed, a.audio_dir)
        tab_seg.to_csv(results_dir / "eval_segment_retrieval.csv",
                       index=False, float_format="%.4f")
        tab_seg.to_csv(results_dir / "eval_weights.csv",
                       index=False, float_format="%.4f")

        # Bieu do dung doan cat (the hien do on dinh dac trung)
        _chart_weights(tab_seg, results_dir / "eval_weights.png")

        all_lines += [
            "=== E1. Tu truy hoi (kiem tra tinh dung dan cua he thong) ===",
            "",
            "--- E1.1 File nguyen ban (tien xu ly chuan 30s trung tam) ---",
            f"So truy van : {nq_exact}",
            tab_exact.round(4).to_string(index=False),
            f"Thoi gian trich xuat TB : {t_ext_exact:.3f} s/file",
            f"Thoi gian tim kiem TB   : {t_srch_exact:.2f} ms/file",
            "",
            "--- E1.2 Doan cat 20 giay (giay 5-25) ---",
            f"So truy van : {nq_seg}",
            tab_seg.round(4).to_string(index=False),
            f"Thoi gian trich xuat TB : {t_ext_seg:.3f} s/file",
            f"Thoi gian tim kiem TB   : {t_srch_seg:.2f} ms/file",
            "",
            "Luu y: Hit@1 cua tu truy hoi KHONG phan anh do chinh xac chung cua he thong.",
        ]

        if a.genre:
            gp = genre_precision_db(eng)
            if gp:
                all_lines += [
                    "",
                    "[Phu --genre] Leave-one-out genre-precision@5 "
                    f"({gp[2]} file, {gp[3]} the loai): "
                    f"{gp[0]:.3f}  (baseline ngau nhien {gp[1]:.3f})",
                    "Luu y: 577/596 file la Instrumental, chi so nay khong du dai dien.",
                ]
            else:
                all_lines += ["", "[Phu --genre] Khong du nhan genre (>= 2 the loai)."]
        all_lines.append("")

    # ---- E2: File nhac moi ngoai CSDL -----------------------------------
    if run_ext:
        ext_lines, summary_df, top5_df, t_ext2, t_srch2, n_ok = external_query_eval(
            eng, a.query_dir, a.metadata, a.audio_dir, a.n, a.seed
        )
        all_lines += ext_lines + [""]

        if not summary_df.empty:
            summary_df.to_csv(results_dir / "eval_external_queries.csv",
                              index=False, float_format="%.4f")
            top5_df.to_csv(results_dir / "eval_external_top5.csv",
                           index=False, float_format="%.4f")
            _chart_dist_hist(summary_df, results_dir / "eval_external_dist.png")

    # ---- E4: Hieu nang --------------------------------------------------
    if perf_rows_all:
        perf_lines, perf_df = summarize_performance(perf_rows_all, db_load_ms)
        if not perf_df.empty:
            perf_df.to_csv(results_dir / "eval_performance.csv", index=False, float_format="%.4f")
            all_lines += perf_lines + [""]

    # ---- Xuat bao cao ---------------------------------------------------
    if run_self and run_ext:
        summary_name = "eval_summary.txt"
    elif run_self:
        summary_name = "eval_summary_self.txt"
    else:
        summary_name = "eval_summary_external.txt"

    txt = "\n".join(all_lines)
    (results_dir / summary_name).write_text(txt + "\n", encoding="utf-8")
    print(txt)

    saved = [summary_name]
    if run_self:
        saved += ["eval_exact_retrieval.csv", "eval_segment_retrieval.csv",
                  "eval_weights.csv", "eval_weights.png"]
    if perf_rows_all:
        saved += ["eval_performance.csv"]
    if run_ext and not summary_df.empty:
        saved += ["eval_external_queries.csv", "eval_external_top5.csv",
                  "eval_external_dist.png"]
    print("\nDa luu:", ", ".join(str(results_dir / name) for name in saved))
