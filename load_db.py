"""
load_db.py - Nap CSDL PostgreSQL (yeu cau 3).

    python load_db.py --metadata metadata.csv --features features/features.csv --reset

Cac buoc: tao bang -> nap audio (ca 500 file) -> nap vector goc -> tinh mean/std tren tap
'database' -> luu scaler_param -> nap vector da chuan hoa -> kiem tra.
"""
import argparse

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from audio_features import FEATURE_NAMES, GROUP_OF
from config import DB_DSN, FEATURES_CSV, METADATA_CSV
from db import connect, create_schema
from scaling import fit_scaler, transform

AUDIO_COLS = ["audio_id", "file_name", "file_path", "format", "genre", "instrument",
              "duration", "sample_rate", "channels", "split"]


def _none(v):
    return None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)


def load(metadata_csv, features_csv, dsn=DB_DSN, reset=False, replace=False):
    meta = pd.read_csv(metadata_csv)
    feat = pd.read_csv(features_csv)
    for frame in (meta, feat):
        if frame["audio_id"].isna().any() or not frame["audio_id"].is_unique:
            raise ValueError("audio_id bi thieu hoac trung")
    missing = meta.loc[~meta["audio_id"].isin(feat["audio_id"]), "file_name"]
    if not missing.empty or len(meta) != len(feat):
        raise ValueError(f"Metadata va features khong khop. File thieu vector: {missing.tolist()}")
    feat = feat.set_index("audio_id").loc[meta["audio_id"]]
    if not np.array_equal(meta["file_name"].to_numpy(), feat["file_name"].to_numpy()):
        raise ValueError("Ten file trong metadata va features khong khop theo audio_id")
    X = feat[FEATURE_NAMES].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise SystemExit("features.csv co NaN hoac vo cuc, kiem tra lai buoc trich xuat")

    is_db = (meta["split"] == "database").to_numpy()
    if not is_db.any():
        raise SystemExit("Khong co file nao thuoc tap 'database'")
    mean, std, const = fit_scaler(X[is_db])
    Z = transform(X, mean, std)
    if not all(np.isfinite(values).all() for values in (mean, std, Z)):
        raise ValueError("Tham so hoac vector Z-score co NaN hoac vo cuc")

    with connect(dsn) as conn:
        if not replace:
            create_schema(conn, reset)
        with conn.cursor() as cur:
            if replace:
                cur.execute("LOCK TABLE audio, audio_feature, audio_feature_scaled, scaler_param "
                            "IN SHARE ROW EXCLUSIVE MODE")
                cur.execute("DELETE FROM audio")
                cur.execute("DELETE FROM scaler_param")
            execute_values(cur, f"INSERT INTO audio ({','.join(AUDIO_COLS)}) VALUES %s",
                           [tuple(_none(v) for v in r) for r in meta[AUDIO_COLS].itertuples(index=False)])
            cols = ",".join(["audio_id"] + FEATURE_NAMES)
            ids = meta["audio_id"].tolist()
            for table, M in (("audio_feature", X), ("audio_feature_scaled", Z)):
                execute_values(cur, f"INSERT INTO {table} ({cols}) VALUES %s",
                               [(int(i), *map(float, row)) for i, row in zip(ids, M)])
            execute_values(cur, "INSERT INTO scaler_param (ordinal, feature_name, grp, mean, std, "
                                "was_constant) VALUES %s",
                           [(k + 1, n, GROUP_OF[k], float(mean[k]), float(std[k]), bool(const[k]))
                            for k, n in enumerate(FEATURE_NAMES)])
        with conn.cursor() as cur:                       # kiem tra
            cur.execute("SELECT (SELECT COUNT(*) FROM audio), (SELECT COUNT(*) FROM audio_feature), "
                        "(SELECT COUNT(*) FROM audio_feature_scaled), (SELECT COUNT(*) FROM scaler_param)")
            n_a, n_f, n_s, n_p = cur.fetchone()
            if (n_a, n_f, n_s, n_p) != (len(meta), len(meta), len(meta), len(FEATURE_NAMES)):
                raise RuntimeError("So ban ghi trong PostgreSQL khong khop voi du lieu dau vao")
            cur.execute("SELECT split, COUNT(*) FROM audio GROUP BY split ORDER BY split")
            splits = cur.fetchall()
    zd = Z[is_db]
    print(f"audio={n_a} | audio_feature={n_f} | audio_feature_scaled={n_s} | scaler_param={n_p}")
    print("Tap chia:", ", ".join(f"{s}={c}" for s, c in splits))
    print(f"Chuan hoa tren tap database: max|mean|={np.abs(zd.mean(0)).max():.2e}, "
          f"std trong [{zd.std(0).min():.3f}, {zd.std(0).max():.3f}]")
    if const.any():
        print("Chieu hang so (std dat = 1):", [n for n, c in zip(FEATURE_NAMES, const) if c])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=str(METADATA_CSV))
    ap.add_argument("--features", default=str(FEATURES_CSV))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--reset", action="store_true", help="xoa bang cu truoc khi nap")
    mode.add_argument("--replace", action="store_true", help="thay du lieu trong cac bang hien co")
    a = ap.parse_args()
    load(a.metadata, a.features, reset=a.reset, replace=a.replace)
