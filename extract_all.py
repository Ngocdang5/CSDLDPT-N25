"""
extract_all.py - Trich xuat vector 71 chieu cho TOAN BO file trong metadata.csv.

    python extract_all.py --audio_dir data/audio --metadata metadata.csv --jobs 4

Chay lai duoc: file da co trong features.csv se duoc bo qua. File loi ghi vao features/errors.csv.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from audio_features import FEATURE_NAMES, extract_from_file
from config import AUDIO_DIR, FEATURES_CSV, METADATA_CSV

EXPECTED_COLS = ["audio_id", "file_name"] + FEATURE_NAMES


def _load_existing(out_path, meta):
    done = pd.read_csv(out_path)
    if list(done.columns) != EXPECTED_COLS:
        raise SystemExit(f"{out_path} khong khop cau truc {len(EXPECTED_COLS)} cot; chay lai voi --force")
    if done["audio_id"].isna().any() or not done["audio_id"].is_unique:
        raise SystemExit(f"{out_path} co audio_id thieu hoac trung; chay lai voi --force")

    meta_names = dict(zip(meta["audio_id"], meta["file_name"]))
    bad_names = [
        int(aid) for aid, fn in zip(done["audio_id"], done["file_name"])
        if aid in meta_names and meta_names[aid] != fn
    ]
    if bad_names:
        raise SystemExit(
            f"{out_path} lech file_name voi metadata.csv (vi du audio_id={bad_names[0]}); chay lai voi --force"
        )

    vals = done[FEATURE_NAMES].to_numpy(dtype=float)
    valid = np.isfinite(vals).all(axis=1)
    if not valid.all():
        print(f"CANH BAO: bo qua {int((~valid).sum())} dong co NaN/vo cuc trong {out_path} de trich xuat lai.")
        done = done.loc[valid].copy()
    return done


def _work(args):
    audio_id, path = args
    try:
        return audio_id, extract_from_file(path), None
    except Exception as e:                       # noqa: BLE001 - ghi loi, chay tiep
        return audio_id, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default=str(AUDIO_DIR))
    ap.add_argument("--metadata", default=str(METADATA_CSV))
    ap.add_argument("--out", default=str(FEATURES_CSV))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="chi chay N file dau (de thu)")
    ap.add_argument("--force", action="store_true", help="trich xuat lai toan bo, bo qua features.csv cu")
    a = ap.parse_args()

    meta = pd.read_csv(a.metadata)
    if a.limit:
        meta = meta.head(a.limit)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    done = pd.DataFrame(columns=EXPECTED_COLS)
    if not a.force and os.path.exists(a.out):
        done = _load_existing(a.out, meta)
    todo = meta[~meta["audio_id"].isin(done["audio_id"])]
    print(f"Tong {len(meta)} file | da co {len(done)} | can chay {len(todo)} | jobs={a.jobs}")

    jobs = [(int(r.audio_id), os.path.join(a.audio_dir, r.file_path)) for r in todo.itertuples()]
    names = dict(zip(meta["audio_id"], meta["file_name"]))
    rows, errors, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, (aid, vec, err) in enumerate(ex.map(_work, jobs, chunksize=4), 1):
            if err:
                errors.append({"audio_id": aid, "file_name": names[aid], "loi": err})
            else:
                rows.append([aid, names[aid]] + vec.tolist())
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)

    new = pd.DataFrame(rows, columns=done.columns)
    out = (new if done.empty else pd.concat([done, new], ignore_index=True)).sort_values("audio_id")
    out[FEATURE_NAMES] = out[FEATURE_NAMES].astype(float)
    out.to_csv(a.out, index=False, float_format="%.6f")
    err_path = os.path.join(os.path.dirname(a.out) or ".", "errors.csv")
    pd.DataFrame(errors, columns=["audio_id", "file_name", "loi"]).to_csv(err_path, index=False)

    X = out[FEATURE_NAMES].to_numpy(dtype=float)
    print(f"Da luu {a.out}: {len(out)} file x {len(FEATURE_NAMES)} dac trung")
    print(f"NaN/vo cuc: {int((~np.isfinite(X)).sum())} | loi: {len(errors)} (xem {err_path})")
    if len(todo):
        print(f"Thoi gian trung binh: {(time.time() - t0) / len(todo):.2f}s/file")


if __name__ == "__main__":
    main()
