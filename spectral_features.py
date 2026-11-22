#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tính spectral contrast, spectral centroid và ước lượng giọng (chroma)
cho toàn bộ file âm thanh, rồi in thống kê để điền vào bảng báo cáo.

Chạy:
  py -3.12 spectral_features.py --input ".\\dataset_wav" --output ".\\ket_qua"
  (thêm --limit 5 để thử)
"""
import argparse
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

SR = 22050
MAX_SECONDS = 30
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif"}

# Profile Krumhansl-Schmuckler để ước lượng giọng
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key(chroma_mean):
    best, best_score = None, -2.0
    for i in range(12):
        for profile, mode in ((MAJOR, "major"), (MINOR, "minor")):
            score = np.corrcoef(chroma_mean, np.roll(profile, i))[0, 1]
            if score > best_score:
                best, best_score = f"{NOTES[i]} {mode}", score
    return best, float(best_score)


def describe(series, name, unit, fmt, high_text, low_text):
    """In trung bình, trung vị, min-max và số file nhóm cao/thấp (chia theo phân vị 1/3)."""
    q1, q2 = series.quantile(1 / 3), series.quantile(2 / 3)
    print(f"{name} trung bình: {series.mean():{fmt}}{unit}")
    print(f"Trung vị: {series.median():{fmt}}{unit} | Min-Max: {series.min():{fmt}} - {series.max():{fmt}}{unit}")
    print(f"{name} cao (>= {q2:{fmt}}{unit}): {int((series >= q2).sum())} file -> {high_text}")
    print(f"{name} thấp (<= {q1:{fmt}}{unit}): {int((series <= q1).sum())} file -> {low_text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="ket_qua")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"Không tìm thấy file âm thanh trong {input_dir}")

    rows, errors, shown = [], [], False
    for idx, path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {path.name}", flush=True)
        try:
            y, _ = librosa.load(str(path), sr=SR, mono=True)
            if y.size == 0:
                raise ValueError("File không có dữ liệu")
            n = MAX_SECONDS * SR
            if len(y) > n:
                s = (len(y) - n) // 2
                y = y[s:s + n]
            peak = float(np.max(np.abs(y)))
            if peak > 0:
                y = y / peak

            centroid = librosa.feature.spectral_centroid(y=y, sr=SR)[0]
            contrast = librosa.feature.spectral_contrast(y=y, sr=SR)
            chroma = librosa.feature.chroma_stft(y=y, sr=SR)
            key, key_score = estimate_key(chroma.mean(axis=1))

            rows.append({
                "ten_file": str(path.relative_to(input_dir)),
                "centroid_mean_hz": float(np.mean(centroid)),
                "centroid_std_hz": float(np.std(centroid)),
                "contrast_mean": float(np.mean(contrast)),
                "contrast_std": float(np.std(contrast)),
                "giong_uoc_tinh": key,
                "giong_do_tin_cay": round(key_score, 3),
            })
        except Exception as e:
            print(f"   LỖI -> {type(e).__name__}: {e}")
            if not shown:
                traceback.print_exc()
                shown = True
            errors.append({"ten_file": str(path), "loi": str(e)})

    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "loi_pho.csv", index=False, encoding="utf-8-sig")
    if not rows:
        raise SystemExit("Không xử lý thành công file nào.")

    df = pd.DataFrame(rows)
    out = output_dir / "dac_trung_pho.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("\n===== SỐ LIỆU ĐỂ ĐIỀN BẢNG =====")
    print(f"Số file: {len(df)}")

    print("\n[Cấu trúc phổ - Spectral contrast]")
    describe(df["contrast_mean"], "Spectral contrast", "", ".3f",
             "nốt rõ, âm hài hòa", "nhiều nhiễu/tiếng gõ")

    print("\n[Cấu trúc phổ - Spectral centroid]")
    describe(df["centroid_mean_hz"], "Spectral centroid", " Hz", ".1f",
             "âm sáng", "âm trầm, ấm")

    print("\n[Hòa âm, giọng] (ước lượng, chỉ mang tính tham khảo)")
    vc = df["giong_uoc_tinh"].value_counts()
    for k, v in vc.head(5).items():
        print(f"  {k}: {v} file ({v / len(df) * 100:.1f}%)")
    major = int(df["giong_uoc_tinh"].str.endswith("major").sum())
    print(f"  Trưởng (major): {major} file | Thứ (minor): {len(df) - major} file")
    print(f"\nĐã lưu: {out}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()