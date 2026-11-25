"""
evaluate_full.py - Danh gia day du cho muc 4c (Nguoi B).

Dat file nay canh evaluate.py (cung thu muc voi search.py, audio_features.py). Khong sua file nao co san.

    python evaluate_full.py --source csv                      # danh gia tren features.csv, khong can PostgreSQL
    python evaluate_full.py --source db                       # doc tu PostgreSQL (kem dung luong bang)
    python evaluate_full.py --manual_template 10              # tao file cham tay cho 10 truy van
    python evaluate_full.py --manual_score results/eval_manual.csv   # tinh P@5 tu file cham tay

A. HIEU QUA (effectiveness): truy van = cac file thuoc tap 'test' (khong nam trong tap tim kiem).
   "Dung" = cung the loai voi truy van (nhan trong metadata.csv). Can >= 2 the loai that.
   Chi so: Precision@K, Recall@K, mAP, R-precision, MRR, hit@1, hit@5, duong Precision-Recall (K = 1..kmax),
   so voi muc ngau nhien. Them phep thu leave-one-out tren tap database (nhieu truy van hon, on dinh hon).
   So sanh 9 phuong an: trong so cac bo A/B/C, tung bo rieng, Euclid 71 chieu, log2(tempo).
B. DO HIEU QUA (efficiency): thoi gian trich xuat / file (neu co file am thanh), thoi gian tim kiem / truy van,
   thoi gian tim kiem theo kich thuoc CSDL (nhan ban vector), dung luong vector so voi file WAV, dung luong bang
   PostgreSQL (neu --source db), cau hinh may.

Ket qua ghi vao results/ (eval_full_summary.txt, eval_effectiveness.csv, eval_pr_curve.csv/png, ...).
"""
import argparse
import math
import os
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audio_features import FEATURE_NAMES
from config import AUDIO_DIR, DB_DSN, FEATURES_CSV, METADATA_CSV, RESULTS_DIR
from scaling import transform
from search import SearchEngine, combine, group_distances

BLUE, ORANGE, GRAY, GRID = "#2a78d6", "#eb6834", "#52514e", "#dddddd"
UNKNOWN = ("unknown", "", "nan", "none")

# ten phuong an -> (trong so A,B,C hoac None = Euclid 71 chieu, co dung log2(tempo) khong)
VARIANTS = {
    "bang nhau (1/3, 1/3, 1/3)": ((1, 1, 1), False),
    "uu tien A (0.5, 0.25, 0.25)": ((2, 1, 1), False),
    "uu tien B (0.25, 0.5, 0.25)": ((1, 2, 1), False),
    "uu tien C (0.25, 0.25, 0.5)": ((1, 1, 2), False),
    "chi bo A": ((1, 0, 0), False),
    "chi bo B": ((0, 1, 0), False),
    "chi bo C": ((0, 0, 1), False),
    "Euclid 71 chieu": (None, False),
    "bang nhau + log2(tempo)": ((1, 1, 1), True),
}
DEFAULT = "bang nhau (1/3, 1/3, 1/3)"


# ------------------------------------------------------------------ nap du lieu
def load_queries(eng, source):
    """Tap 'test': (bang thong tin, vector da chuan hoa bang mean/std cua tap database)."""
    if source == "csv":
        meta = pd.read_csv(METADATA_CSV)
        feat = pd.read_csv(FEATURES_CSV)
        df = meta.merge(feat.drop(columns=["file_name"]), on="audio_id")
        df = df[df["split"] == "test"].reset_index(drop=True)
    else:
        from db import connect
        cols = ",".join(f"f.{n}" for n in FEATURE_NAMES)
        with connect(DB_DSN) as conn, conn.cursor() as cur:
            cur.execute("SELECT a.audio_id, a.file_name, a.file_path, a.genre, " + cols +
                        " FROM audio a JOIN audio_feature f USING (audio_id) "
                        "WHERE a.split = 'test' ORDER BY a.audio_id")
            df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64) if len(df) else np.zeros((0, len(FEATURE_NAMES)))
    return df[["audio_id", "file_name", "file_path", "genre"]], transform(X, eng.mean, eng.std)


def log_tempo(Zdb, Zq, eng):
    """Thay chieu tempo bang log2(BPM) roi chuan hoa lai tren tap database (khong can trich xuat lai)."""
    j = FEATURE_NAMES.index("tempo")
    ldb = np.log2(np.clip(Zdb[:, j] * eng.std[j] + eng.mean[j], 1.0, None))
    lq = np.log2(np.clip(Zq[:, j] * eng.std[j] + eng.mean[j], 1.0, None))
    m, s = ldb.mean(), (ldb.std() or 1.0)
    Zdb2, Zq2 = Zdb.copy(), Zq.copy()
    Zdb2[:, j], Zq2[:, j] = (ldb - m) / s, (lq - m) / s
    return Zdb2, Zq2


def clean_genre(s):
    return np.array([("unknown" if str(x).strip().lower() in UNKNOWN else str(x)) for x in s])


# ------------------------------------------------------------------ do chat luong
def dist(zq, Z, w):
    if w is None:
        return np.sqrt(((Z - zq) ** 2).sum(axis=1))
    return combine(group_distances(zq, Z), w)


def ranking_metrics(rel, kmax):
    """rel: mang bool theo thu tu xep hang. -> (dict chi so, P@K, R@K voi K = 1..kmax)."""
    n_rel = int(rel.sum())
    cum = np.cumsum(rel)
    ks = np.arange(1, len(rel) + 1)
    prec, rec = cum / ks, cum / n_rel
    k5 = min(5, len(rel))
    m = {"P@5": prec[k5 - 1], "R@5": rec[k5 - 1], "AP": float((prec * rel).sum() / n_rel),
         "R-precision": float(rel[:n_rel].mean()), "MRR": 1.0 / (int(np.argmax(rel)) + 1),
         "hit@1": float(rel[0]), "hit@5": float(rel[:k5].any())}
    p, r = np.full(kmax, np.nan), np.full(kmax, np.nan)
    kk = min(kmax, len(rel))
    p[:kk], r[:kk] = prec[:kk], rec[:kk]
    return m, p, r


def evaluate(Zq, gq, Zdb, gdb, w, kmax, loo=False):
    """Chay moi truy van. loo=True: truy van la chinh cac file cua database (loai chinh no khoi xep hang).
    Tra ve: bang tung truy van, P@K trung binh, R@K trung binh, muc ngau nhien, so cap trung vector (khoang cach 0)."""
    rows, P, R, rand_p, dup = [], [], [], [], 0
    for i in range(len(Zq)):
        if gq[i] == "unknown":
            continue
        D = dist(Zq[i], Zdb, w)
        rel = gdb == gq[i]
        top1 = gdb
        if loo:
            D, rel, top1 = np.delete(D, i), np.delete(rel, i), np.delete(gdb, i)
        if rel.sum() == 0:
            continue
        order = np.argsort(D, kind="stable")
        m, p, r = ranking_metrics(rel[order], kmax)
        m.update(genre=gq[i], top1_genre=top1[order[0]], query_index=i)
        rows.append(m)
        P.append(p)
        R.append(r)
        rand_p.append(rel.mean())
        dup += int(D.min() < 1e-9)
    if not rows:
        return None
    return {"per_query": pd.DataFrame(rows), "P": np.nanmean(P, axis=0), "R": np.nanmean(R, axis=0),
            "random_p": float(np.mean(rand_p)), "n_dup": dup}


def summarize(res, n_db):
    df = res["per_query"]
    num = ["P@5", "R@5", "AP", "R-precision", "MRR", "hit@1", "hit@5"]
    out = {k: float(df[k].mean()) for k in num}
    out["n_query"] = len(df)
    out["P@5 +-95%"] = 1.96 * float(df["P@5"].std(ddof=1) / math.sqrt(len(df))) if len(df) > 1 else float("nan")
    out["P@5 ngau nhien"] = res["random_p"]
    out["R@5 ngau nhien"] = min(5, n_db) / n_db
    out["so cap trung vector"] = res["n_dup"]
    return out


def run_effectiveness(eng, Zq, gq, gdb, kmax, loo):
    table, curves, details = [], {}, {}
    for name, (w, lt) in VARIANTS.items():
        Zd, Zqq = log_tempo(eng.Z, Zq, eng) if lt else (eng.Z, Zq)
        res = evaluate(Zqq, gq, Zd, gdb, w, kmax, loo=loo)
        if res is None:
            return None, None, None
        row = {"phuong_an": name}
        row.update(summarize(res, len(Zd)))
        table.append(row)
        curves[name] = (res["P"], res["R"])
        details[name] = res
    return pd.DataFrame(table), curves, details


# ------------------------------------------------------------------ do hieu qua
def search_time_ms(eng, Zq, reps=3):
    """Thoi gian search_vector (tinh khoang cach + sap xep + dung bang ket qua) tren moi truy van, ms."""
    ts = []
    for z in Zq[:50]:
        for _ in range(reps):
            t0 = time.perf_counter()
            eng.search_vector(z)
            ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000) if ts else float("nan")


def scaling_curve(Zdb, max_n, reps=5):
    """Nhan ban tap vector len N de do thoi gian tinh khoang cach + sap xep cho 1 truy van (brute force)."""
    sizes = sorted({n for n in (len(Zdb), 5_000, 20_000, 50_000, 100_000, 200_000, 500_000) if n <= max_n})
    rows, zq = [], Zdb[0]
    for n in sizes:
        Z = np.tile(Zdb, (math.ceil(n / len(Zdb)), 1))[:n]
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            D = combine(group_distances(zq, Z), (1, 1, 1))
            np.argsort(D, kind="stable")[:5]
            ts.append(time.perf_counter() - t0)
        rows.append({"so_vector": n, "ms_moi_truy_van": float(np.median(ts) * 1000),
                     "bo_nho_vector_MB": Z.nbytes / 1e6})
        del Z
    return pd.DataFrame(rows)


def extraction_time(meta, audio_dir, n):
    """Thoi gian doc + tien xu ly + trich xuat tren n file dau co that tren dia. None neu khong do duoc."""
    try:
        from audio_features import extract_features, load_audio, preprocess_array
    except Exception:  # noqa: BLE001
        return None
    t_load, t_ext = [], []
    for p in meta["file_path"]:
        path = os.path.join(audio_dir, p)
        if not os.path.exists(path):
            continue
        try:
            t0 = time.perf_counter()
            y = preprocess_array(load_audio(path))
            t1 = time.perf_counter()
            extract_features(y)
            t2 = time.perf_counter()
        except Exception:  # noqa: BLE001
            continue
        t_load.append(t1 - t0)
        t_ext.append(t2 - t1)
        if len(t_ext) >= n:
            break
    if not t_ext:
        return None
    return {"n": len(t_ext), "doc_tien_xu_ly_s": float(np.median(t_load)), "trich_xuat_s": float(np.median(t_ext))}


def storage_report(meta, audio_dir, source):
    """Dung luong: file WAV (thuc te neu co, neu khong uoc tinh 16-bit mono) so voi vector 71 chieu."""
    sizes, real = [], 0
    for _, r in meta.iterrows():
        p = os.path.join(audio_dir, r["file_path"])
        if os.path.exists(p):
            sizes.append(os.path.getsize(p))
            real += 1
        else:
            sizes.append(float(r["duration"]) * float(r["sample_rate"]) * 2 * max(int(r.get("channels", 1) or 1), 1) + 44)
    d = len(FEATURE_NAMES)
    rep = {"so_file": len(meta), "wav_trung_binh_KB": float(np.mean(sizes)) / 1024,
           "wav_tong_MB": float(np.sum(sizes)) / 1e6, "file_that_tren_dia": real,
           "vector_byte_moi_file": d * 8, "vector_tong_KB": len(meta) * d * 8 / 1024,
           "ti_le_nen": float(np.mean(sizes)) / (d * 8)}
    if source == "db":
        try:
            from db import connect
            with connect(DB_DSN) as conn, conn.cursor() as cur:
                pg = {}
                for t in ("audio", "audio_feature", "audio_feature_scaled", "scaler_param"):
                    cur.execute("SELECT pg_total_relation_size(%s)", (t,))
                    pg[t] = cur.fetchone()[0]
                cur.execute("SELECT pg_database_size(current_database())")
                pg["_ca_csdl"] = cur.fetchone()[0]
            rep["postgres_bytes"] = pg
        except Exception as e:  # noqa: BLE001
            rep["postgres_loi"] = f"{type(e).__name__}: {e}"
    return rep


def hardware():
    info = {"he_dieu_hanh": platform.platform(), "cpu": platform.processor() or platform.machine(),
            "so_luong_cpu": os.cpu_count(), "python": platform.python_version(), "numpy": np.__version__}
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo", encoding="utf-8") as f:
            info["ram_GB"] = round(int(f.readline().split()[1]) / 1e6, 1)
    except OSError:
        try:
            import psutil
            info["ram_GB"] = round(psutil.virtual_memory().total / 1e9, 1)
        except Exception:  # noqa: BLE001
            pass
    try:
        import librosa
        info["librosa"] = librosa.__version__
    except Exception:  # noqa: BLE001
        pass
    return info


# ------------------------------------------------------------------ cham tay
def make_manual_template(eng, Zq, qinfo, n, seed, path):
    """Chon n truy van tu tap test, xuat top 5 de nguoi nghe va dien cot 'giong' (1 = giong, 0 = khong)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Zq), size=min(n, len(Zq)), replace=False)
    rows = []
    for i in sorted(idx):
        res = eng.search_vector(Zq[i])
        for _, r in res.iterrows():
            rows.append({"query_file": qinfo.iloc[i]["file_name"], "query_path": qinfo.iloc[i]["file_path"],
                         "rank": int(r["rank"]), "result_file": r["file_name"], "result_path": r["file_path"],
                         "distance": round(float(r["distance"]), 4), "giong": "", "ghi_chu": ""})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Da ghi {path}: {len(idx)} truy van x 5 ket qua. Nghe, dien cot 'giong' (1 hoac 0), luu thanh "
          f"results/eval_manual.csv, roi chay --manual_score.")


def score_manual(path):
    df = pd.read_csv(path)
    df["giong"] = pd.to_numeric(df["giong"], errors="coerce")
    miss = int(df["giong"].isna().sum())
    if miss:
        print(f"CANH BAO: {miss} dong chua cham, bo qua.")
    df = df.dropna(subset=["giong"])
    per = df.groupby("query_file")["giong"].mean().rename("P@5")
    top1 = df[df["rank"] == 1].set_index("query_file")["giong"].rename("top1_giong")
    out = pd.concat([per, top1], axis=1)
    lines = [f"Cham tay: {len(per)} truy van, {len(df)} cap (truy van, ket qua)",
             f"Precision@5 trung binh : {per.mean():.3f}  (+-{1.96 * per.std(ddof=1) / math.sqrt(len(per)):.3f} 95%)"
             if len(per) > 1 else f"Precision@5 trung binh : {per.mean():.3f}",
             f"Ty le ket qua hang 1 giong: {out['top1_giong'].mean():.3f}", "", out.round(2).to_string()]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    txt = "\n".join(lines)
    (RESULTS_DIR / "eval_manual_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


# ------------------------------------------------------------------ ve hinh
def style(ax):
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_pr(curves, table, random_p, out, title):
    best = table.sort_values("AP", ascending=False).iloc[0]["phuong_an"]
    show = [DEFAULT] + ([best] if best != DEFAULT else []) + ["Euclid 71 chieu"]
    colors = [BLUE, ORANGE, GRAY]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for name, c in zip(show, colors):
        p, r = curves[name]
        ax.plot(r, p, color=c, lw=1.8, label=name)
        for k in (5, 10, 20):
            if k <= len(p) and not np.isnan(p[k - 1]):
                ax.plot(r[k - 1], p[k - 1], "o", color=c, ms=4)
                if name != DEFAULT:
                    continue
                ax.annotate(f"K={k}", (r[k - 1], p[k - 1]), textcoords="offset points", xytext=(4, 4), fontsize=6.5,
                            color=c)
    ax.axhline(random_p, color=GRAY, lw=1, ls="--")
    ax.text(ax.get_xlim()[1], random_p, " ngau nhien", va="center", fontsize=7, color=GRAY)
    ax.set_xlabel("Recall@K")
    ax.set_ylabel("Precision@K")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    style(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_variants(table, out):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["P@5"], 0.4, color=BLUE, label="Precision@5")
    ax.bar(x + 0.2, table["AP"], 0.4, color=ORANGE, label="mAP")
    ax.axhline(float(table["P@5 ngau nhien"].iloc[0]), color=GRAY, ls="--", lw=1)
    ax.set_xticks(x, table["phuong_an"], rotation=30, ha="right", fontsize=7)
    ax.set_title("Hieu qua tim kiem theo phuong an (tap test; gach = muc ngau nhien)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_scaling(sc, out):
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=150)
    ax.loglog(sc["so_vector"], sc["ms_moi_truy_van"], "o-", color=BLUE, lw=1.8)
    ax.set_xlabel("So vector trong CSDL")
    ax.set_ylabel("ms moi truy van (brute force)")
    ax.set_title("Thoi gian tim kiem tang theo kich thuoc CSDL", loc="left", fontsize=10)
    ax.grid(True, color=GRID, which="both")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["db", "csv"], default="csv")
    ap.add_argument("--audio_dir", default=str(AUDIO_DIR))
    ap.add_argument("--kmax", type=int, default=50, help="K lon nhat cho duong Precision-Recall")
    ap.add_argument("--extract_n", type=int, default=10, help="so file do thoi gian trich xuat (0 = bo qua)")
    ap.add_argument("--max_n", type=int, default=200000, help="kich thuoc lon nhat khi thu kha nang mo rong")
    ap.add_argument("--no_efficiency", action="store_true")
    ap.add_argument("--manual_template", type=int, default=0, metavar="N")
    ap.add_argument("--manual_score", default="", metavar="CSV")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.manual_score:
        score_manual(a.manual_score)
        return

    eng = SearchEngine.load(a.source)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    qinfo, Zq = load_queries(eng, a.source)
    gdb = clean_genre(eng.meta["genre"])
    gq = clean_genre(qinfo["genre"])
    n_db, n_q = len(eng.Z), len(Zq)

    if a.manual_template:
        if n_q == 0:
            raise SystemExit("Khong co file nao thuoc tap 'test'; chay build_metadata.py voi --test_ratio 0.1")
        make_manual_template(eng, Zq, qinfo, a.manual_template, a.seed, RESULTS_DIR / "eval_manual_template.csv")
        return

    L = [f"CSDL tim kiem: {n_db} file | tap test: {n_q} file | K lon nhat: {a.kmax}", ""]
    known = sorted(set(gdb) - {"unknown"})
    if len(known) < 2:
        L += ["Khong co nhan the loai (>= 2 the loai) trong metadata.csv: bo qua cac chi so Precision/Recall.",
              "Gan nhan bang build_metadata.py --labels labels.csv roi chay lai."]
        table = None
    else:
        sizes = pd.Series(gdb[gdb != "unknown"]).value_counts()
        L += [f"So the loai: {len(known)} | file moi the loai trong database: {sizes.min()} den {sizes.max()}",
              "'Dung' = cung the loai voi truy van. The loai chi la tham chieu: cung the loai khong dong nghia "
              "giong nhau, va hai file khac the loai van co the giong nhau.", ""]
        table, curves, det = (run_effectiveness(eng, Zq, gq, gdb, a.kmax, loo=False) if n_q else (None, None, None))
        if table is not None:
            table.to_csv(RESULTS_DIR / "eval_effectiveness.csv", index=False, float_format="%.4f")
            show = table.drop(columns=["n_query"]).round(3)
            L += [f"A1. TAP TEST ({int(table['n_query'].iloc[0])} truy van, tim trong {n_db} file)",
                  show.to_string(index=False), ""]
            pr = pd.DataFrame({"K": np.arange(1, a.kmax + 1)})
            for name, (p, r) in curves.items():
                pr[f"P@K | {name}"], pr[f"R@K | {name}"] = p, r
            pr.to_csv(RESULTS_DIR / "eval_pr_curve.csv", index=False, float_format="%.4f")
            plot_pr(curves, table, float(table["P@5 ngau nhien"].iloc[0]), RESULTS_DIR / "eval_pr_curve.png",
                    "Duong Precision-Recall tren tap test (K tang tu 1)")
            plot_variants(table, RESULTS_DIR / "eval_variants.png")
            d = det[DEFAULT]["per_query"]
            g = d.groupby("genre").agg(so_truy_van=("P@5", "size"), P5=("P@5", "mean"), AP=("AP", "mean"))
            g.round(3).to_csv(RESULTS_DIR / "eval_per_genre.csv")
            pd.crosstab(d["genre"], d["top1_genre"]).to_csv(RESULTS_DIR / "eval_confusion_top1.csv")
            L += ["P@5 va mAP theo the loai (phuong an mac dinh):", g.round(3).to_string(), ""]
            if det[DEFAULT]["n_dup"]:
                L += [f"CANH BAO: {det[DEFAULT]['n_dup']} truy van co file trung vector trong CSDL (khoang cach 0): "
                      "co the la file trung lap, hay loai bo truoc khi bao cao.", ""]
        else:
            L += ["Tap test rong hoac khong co truy van hop le (the loai khong co trong database): bo qua A1.", ""]
        t2, _, det2 = run_effectiveness(eng, eng.Z, gdb, gdb, a.kmax, loo=True)
        if t2 is not None:
            t2.to_csv(RESULTS_DIR / "eval_effectiveness_loo.csv", index=False, float_format="%.4f")
            L += [f"A2. LEAVE-ONE-OUT TREN TAP DATABASE ({int(t2['n_query'].iloc[0])} truy van, moi file lam truy van, "
                  "loai chinh no)", t2.drop(columns=["n_query"]).round(3).to_string(index=False), ""]

    if not a.no_efficiency:
        L += ["B. DO HIEU QUA", ""]
        L += ["Cau hinh may: " + ", ".join(f"{k}={v}" for k, v in hardware().items())]
        meta_all = pd.read_csv(METADATA_CSV)
        if n_q:
            L += [f"Thoi gian tim kiem moi truy van (tim trong {n_db} file): {search_time_ms(eng, Zq):.2f} ms"]
        if a.extract_n:
            et = extraction_time(eng.meta.head(max(a.extract_n * 3, 10)), a.audio_dir, a.extract_n)
            L += [(f"Thoi gian xu ly 1 file (trung vi tren {et['n']} file): doc va tien xu ly {et['doc_tien_xu_ly_s']:.2f}s"
                   f", trich xuat 71 chieu {et['trich_xuat_s']:.2f}s") if et else
                  "Thoi gian trich xuat: bo qua (khong thay file am thanh o --audio_dir hoac thieu librosa)."]
        sr = storage_report(meta_all, a.audio_dir, a.source)
        L += [f"Dung luong: WAV trung binh {sr['wav_trung_binh_KB']:.0f} KB/file ({sr['wav_tong_MB']:.0f} MB cho "
              f"{sr['so_file']} file; {sr['file_that_tren_dia']} file do tu dia, con lai uoc tinh), "
              f"vector {sr['vector_byte_moi_file']} byte/file ({sr['vector_tong_KB']:.0f} KB tong), "
              f"nho hon khoang {sr['ti_le_nen']:.0f} lan"]
        if "postgres_bytes" in sr:
            L += ["Bang PostgreSQL (pg_total_relation_size): " +
                  ", ".join(f"{k}={v / 1024:.0f} KB" for k, v in sr["postgres_bytes"].items())]
        elif "postgres_loi" in sr:
            L += [f"Khong doc duoc dung luong PostgreSQL: {sr['postgres_loi']}"]
        sc = scaling_curve(eng.Z, a.max_n)
        sc.to_csv(RESULTS_DIR / "eval_scaling.csv", index=False, float_format="%.3f")
        plot_scaling(sc, RESULTS_DIR / "eval_scaling.png")
        L += ["", "Kha nang mo rong (nhan ban vector, brute force, 1 truy van):", sc.round(2).to_string(index=False)]

    txt = "\n".join(L)
    (RESULTS_DIR / "eval_full_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\nDa ghi cac file trong {RESULTS_DIR}")


if __name__ == "__main__":
    main()
