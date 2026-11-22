"""
search.py - Tim 5 doan nhac giong nhat voi mot file am thanh moi (yeu cau 4).

    python search.py query.wav                      # doc CSDL PostgreSQL
    python search.py query.wav --source csv         # khong can PostgreSQL (doc metadata + features CSV)
    python search.py query.wav --weights 2,1,1      # doi trong so bo A, B, C

Khoang cach (muc 2.9.1):  D_g = sqrt( (1/d_g) * sum_j (z_xj - z_yj)^2 ),   D = sqrt( sum_g w_g * D_g^2 ).
Do tuong dong hien thi: 1 / (1 + D)  (don dieu giam theo D).
"""
import argparse
import time

import numpy as np
import pandas as pd

from audio_features import (FEATURE_NAMES, GROUP_INDEX, GROUPS, extract_features, preprocess)
from config import (DB_DSN, FEATURES_CSV, GROUP_WEIGHTS, METADATA_CSV, TOP_K)
from scaling import fit_scaler, transform


def normalize_weights(w):
    """dict hoac (A, B, C) -> dict tong bang 1."""
    if not isinstance(w, dict):
        w = dict(zip(GROUPS, w))
    tot = float(sum(w.values()))
    if tot <= 0:
        raise ValueError("Tong trong so phai duong")
    return {g: float(w[g]) / tot for g in GROUPS}


def group_distances(zq, Z):
    """zq: (71,), Z: (n, 71) -> (n, 3) gom D_A, D_B, D_C."""
    diff2 = (Z - zq) ** 2
    return np.column_stack([np.sqrt(diff2[:, GROUP_INDEX[g]].mean(axis=1)) for g in GROUPS])


def combine(Dg, weights):
    w = normalize_weights(weights)
    return np.sqrt((Dg ** 2) @ np.array([w[g] for g in GROUPS]))


class SearchEngine:
    def __init__(self, meta, Z, mean, std):
        self.meta = meta.reset_index(drop=True)      # audio_id, file_name, file_path, genre
        self.Z = np.asarray(Z, dtype=np.float64)     # vector da chuan hoa cua tap database
        self.mean, self.std = mean, std

    # ------------------------------------------------------------ nguon du lieu
    @classmethod
    def from_db(cls, dsn=DB_DSN):
        from db import connect
        with connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT feature_name, mean, std FROM scaler_param ORDER BY ordinal")
            sp = pd.DataFrame(cur.fetchall(), columns=["feature_name", "mean", "std"])
            if list(sp["feature_name"]) != FEATURE_NAMES:
                raise RuntimeError("scaler_param khong khop thu tu FEATURE_NAMES; nap lai CSDL")
            cols = ",".join(f"s.{n}" for n in FEATURE_NAMES)
            cur.execute("SELECT a.audio_id, a.file_name, a.file_path, a.genre, " + cols +
                        " FROM audio a JOIN audio_feature_scaled s USING (audio_id) "
                        "WHERE a.split = 'database' ORDER BY a.audio_id")
            df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
        return cls(df[["audio_id", "file_name", "file_path", "genre"]],
                   df[FEATURE_NAMES].to_numpy(), sp["mean"].to_numpy(), sp["std"].to_numpy())

    @classmethod
    def from_csv(cls, metadata_csv=METADATA_CSV, features_csv=FEATURES_CSV):
        meta = pd.read_csv(metadata_csv)
        feat = pd.read_csv(features_csv)
        df = meta.merge(feat.drop(columns=["file_name"]), on="audio_id")
        X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
        is_db = (df["split"] == "database").to_numpy()
        mean, std, _ = fit_scaler(X[is_db])
        sel = df[is_db]
        return cls(sel[["audio_id", "file_name", "file_path", "genre"]],
                   transform(X[is_db], mean, std), mean, std)

    @classmethod
    def load(cls, source="db"):
        return cls.from_db() if source == "db" else cls.from_csv()

    # ------------------------------------------------------------ truy van
    def embed(self, y):
        """y da tien xu ly -> (vector goc, vector chuan hoa bang tham so cua CSDL)."""
        raw = extract_features(y)
        return raw, transform(raw, self.mean, self.std)

    def search_vector(self, zq, k=TOP_K, weights=GROUP_WEIGHTS, exclude_ids=()):
        Dg = group_distances(zq, self.Z)
        D = combine(Dg, weights)
        order = np.argsort(D, kind="stable")
        if len(exclude_ids):
            order = order[~self.meta["audio_id"].iloc[order].isin(exclude_ids).to_numpy()]
        order = order[:k]
        out = self.meta.iloc[order].copy()
        out.insert(0, "rank", np.arange(1, len(order) + 1))
        out["distance"] = D[order]
        out["similarity"] = 1.0 / (1.0 + D[order])
        for j, g in enumerate(GROUPS):
            out[f"D_{g}"] = Dg[order, j]
        return out.reset_index(drop=True)

    def search_file(self, path, k=TOP_K, weights=GROUP_WEIGHTS, exclude_ids=()):
        y = preprocess(path)
        raw, zq = self.embed(y)
        return self.search_vector(zq, k, weights, exclude_ids), raw, zq


def parse_weights(s):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--weights can 3 so, vi du 1,1,1")
    return tuple(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--source", choices=["db", "csv"], default="db")
    ap.add_argument("--weights", type=parse_weights, default=(1, 1, 1))
    a = ap.parse_args()

    t0 = time.time()
    eng = SearchEngine.load(a.source)
    t1 = time.time()
    res, _, _ = eng.search_file(a.query, a.k, a.weights)
    t2 = time.time()
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    print(f"\nNap CSDL {t1 - t0:.2f}s | trich xuat + tim kiem {t2 - t1:.2f}s")
