#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phân tích bộ dữ liệu âm thanh:
- Tiền xử lý: mono, 22050 Hz, lấy đoạn giữa tối đa 30 giây, chuẩn hóa đỉnh
- Trích xuất: thời lượng, tempo ước lượng, MFCC, RMS
- Tùy chọn nhận dạng nhãn âm thanh bằng mô hình AudioSet của Hugging Face
- Xuất CSV chi tiết, thống kê tổng hợp và báo cáo TXT

Cách chạy:
  py -3.12 audio_dataset_analysis.py --input ".\\dataset_wav" --output ".\\ket_qua"
Thêm --classify để chạy mô hình AI nhận dạng nhãn âm thanh.
Thêm --limit 5 để thử với 5 file đầu.
"""
import argparse
import json
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

SR = 22050
SR_CLASSIFIER = 16000  # mô hình AST được huấn luyện ở 16 kHz
MAX_SECONDS = 30

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif"}

INSTRUMENT_KEYWORDS = {
    "guitar": ["guitar", "electric guitar", "acoustic guitar", "bass guitar"],
    "piano/keyboard": ["piano", "keyboard", "organ", "synthesizer"],
    "trống/bộ gõ": ["drum", "snare", "bass drum", "cymbal", "percussion", "tambourine"],
    "violin/dây kéo": ["violin", "cello", "string section", "bowed string"],
    "saxophone": ["saxophone"],
    "trumpet/kèn": ["trumpet", "trombone", "brass instrument", "french horn"],
    "flute": ["flute", "recorder"],
    "voice/vocal": ["singing", "vocal", "choir", "speech"],
    "bass": ["bass"],
}
GENRE_KEYWORDS = {
    "rock": ["rock music", "heavy metal", "punk rock", "grunge"],
    "pop": ["pop music"],
    "jazz": ["jazz"],
    "classical": ["classical music", "opera"],
    "hip-hop/rap": ["hip hop", "rap"],
    "electronic/dance": ["electronic music", "house music", "techno", "dubstep", "disco"],
    "country/folk": ["country", "folk music", "bluegrass"],
    "blues": ["blues"],
    "reggae": ["reggae"],
    "ambient": ["ambient music"],
}


def peak_normalize(y):
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32), peak


def get_audio_files(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def classify_label(label, mapping):
    low = label.lower()
    return [cat for cat, words in mapping.items() if any(w in low for w in words)]


def load_audio(path):
    """Đọc mono, resample về SR. Thử librosa trước, nếu lỗi thì dùng soundfile + scipy."""
    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        return y.astype(np.float32)
    except Exception as first_error:
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            y = data.mean(axis=1)
            if sr != SR:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(int(sr), SR)
                y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
            return y
        except Exception as second_error:
            raise RuntimeError(
                f"librosa.load: {first_error!r} | soundfile: {second_error!r}"
            ) from second_error


def estimate_tempo(y):
    """Ước lượng tempo; nếu lỗi thì trả về NaN thay vì làm hỏng cả file."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=SR)
        return float(np.asarray(tempo).reshape(-1)[0])
    except Exception:
        pass
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=SR)
        tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=SR)
        return float(np.asarray(tempo).reshape(-1)[0])
    except Exception:
        return float("nan")


def save_errors(errors, output_dir):
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "loi_xu_ly.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="Thống kê và trích xuất đặc trưng file âm thanh.")
    parser.add_argument("--input", required=True, help="Thư mục chứa file âm thanh (quét cả thư mục con).")
    parser.add_argument("--output", default="ket_qua_am_thanh", help="Thư mục lưu kết quả.")
    parser.add_argument("--classify", action="store_true",
                        help="Bật phân loại AI (cần internet lần đầu để tải model và cài transformers/torch).")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ xử lý N file để thử; 0 = tất cả.")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Không tìm thấy thư mục đầu vào: {input_dir}")

    files = get_audio_files(input_dir)
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"Không tìm thấy file âm thanh trong {input_dir}. Định dạng hỗ trợ: {sorted(AUDIO_EXTS)}")

    classifier = None
    if args.classify:
        try:
            from transformers import pipeline
            classifier = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-audioset-10-10-0.4593",
                top_k=10,
            )
        except Exception as e:
            raise SystemExit(
                "Không khởi tạo được mô hình AI. Hãy cài: pip install transformers torch. "
                f"Chi tiết: {e}"
            )

    rows = []
    errors = []
    first_traceback_shown = False

    for idx, path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {path.name}", flush=True)
        try:
            y = load_audio(path)
            if y.size == 0:
                raise ValueError("File không có dữ liệu âm thanh")
            duration_original = len(y) / SR

            # Lấy đoạn giữa tối đa 30 giây
            target_samples = MAX_SECONDS * SR
            if len(y) > target_samples:
                start = (len(y) - target_samples) // 2
                y = y[start:start + target_samples]
            segment_duration = len(y) / SR

            # Chuẩn hóa đỉnh
            y_norm, peak_before = peak_normalize(y)

            # MFCC: 13 hệ số; mean/std từng hệ số
            mfcc = librosa.feature.mfcc(y=y_norm, sr=SR, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            # RMS và tempo
            rms = librosa.feature.rms(y=y_norm)[0]
            tempo_value = estimate_tempo(y_norm)

            row = {
                "ten_file": str(path.relative_to(input_dir)),
                "dinh_dang": path.suffix.lower(),
                "thoi_luong_goc_giay": round(duration_original, 3),
                "thoi_luong_doan_giay": round(segment_duration, 3),
                "tan_so_lay_mau_hz": SR,
                "kenh": "Mono",
                "chuan_hoa_dinh": "Có",
                "peak_truoc_chuan_hoa": round(peak_before, 6),
                "tempo_bpm_uoc_tinh": round(tempo_value, 2) if np.isfinite(tempo_value) else np.nan,
                "rms_mean": round(float(np.mean(rms)), 6),
                "rms_std": round(float(np.std(rms)), 6),
                "mfcc_mean": json.dumps([round(float(v), 6) for v in mfcc_mean], ensure_ascii=False),
                "mfcc_std": json.dumps([round(float(v), 6) for v in mfcc_std], ensure_ascii=False),
                "nhan_ai_top1": "",
                "nhan_ai_top1_score": "",
                "nhan_ai_top10": "",
                "nhac_cu_du_doan": "",
                "the_loai_du_doan": "",
                "so_nhan_nhac_cu_du_doan": 0,
            }

            if classifier is not None:
                # Mô hình AST dùng 16 kHz -> tự resample để chắc chắn đúng
                y_cls = librosa.resample(y_norm, orig_sr=SR, target_sr=SR_CLASSIFIER)
                preds = classifier({"raw": y_cls, "sampling_rate": SR_CLASSIFIER})
                row["nhan_ai_top1"] = preds[0]["label"] if preds else ""
                row["nhan_ai_top1_score"] = round(float(preds[0]["score"]), 5) if preds else ""
                row["nhan_ai_top10"] = json.dumps(
                    [{"label": p["label"], "score": round(float(p["score"]), 5)} for p in preds],
                    ensure_ascii=False,
                )
                instruments, genres = [], []
                for pred in preds:
                    instruments.extend(classify_label(pred["label"], INSTRUMENT_KEYWORDS))
                    genres.extend(classify_label(pred["label"], GENRE_KEYWORDS))
                instruments = sorted(set(instruments))
                genres = sorted(set(genres))
                row["nhac_cu_du_doan"] = "; ".join(instruments) if instruments else "Chưa xác định"
                row["the_loai_du_doan"] = "; ".join(genres) if genres else "Chưa xác định"
                row["so_nhan_nhac_cu_du_doan"] = len(instruments)

            rows.append(row)

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"   LỖI -> {msg}", flush=True)
            if not first_traceback_shown:
                traceback.print_exc()  # chỉ in traceback đầy đủ cho lỗi đầu tiên
                first_traceback_shown = True
            errors.append({"ten_file": str(path), "loi": msg})

    # Luôn ghi file lỗi TRƯỚC khi có thể thoát
    save_errors(errors, output_dir)

    if not rows:
        raise SystemExit(
            "Không xử lý thành công file nào. Xem thông báo LỖI/traceback ở trên "
            f"hoặc file {output_dir / 'loi_xu_ly.csv'}"
        )

    df = pd.DataFrame(rows)
    detail_path = output_dir / "ket_qua_chi_tiet.csv"
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    # Thống kê nhạc cụ/thể loại (chỉ có ý nghĩa khi bật --classify)
    instrument_counter, genre_counter = {}, {}
    one_instrument = multi_instrument = None
    if args.classify:
        one_instrument = int((df["so_nhan_nhac_cu_du_doan"] == 1).sum())
        multi_instrument = int((df["so_nhan_nhac_cu_du_doan"] >= 2).sum())
        for value in df["nhac_cu_du_doan"].fillna(""):
            for item in value.split("; "):
                if item and item != "Chưa xác định":
                    instrument_counter[item] = instrument_counter.get(item, 0) + 1
        for value in df["the_loai_du_doan"].fillna(""):
            for item in value.split("; "):
                if item and item != "Chưa xác định":
                    genre_counter[item] = genre_counter.get(item, 0) + 1

    valid_tempo = df["tempo_bpm_uoc_tinh"].replace([np.inf, -np.inf], np.nan).dropna()
    valid_tempo = valid_tempo[valid_tempo > 0]
    tempo_range = f"{valid_tempo.min():.1f} - {valid_tempo.max():.1f}" if len(valid_tempo) else "Không đủ dữ liệu"

    summary_rows = [
        {"chi_tieu": "Tổng số file âm thanh tìm thấy", "gia_tri": len(files)},
        {"chi_tieu": "Số file xử lý thành công", "gia_tri": len(df)},
        {"chi_tieu": "Số file lỗi", "gia_tri": len(errors)},
        {"chi_tieu": "Tổng thời lượng gốc (giờ)", "gia_tri": round(df["thoi_luong_goc_giay"].sum() / 3600, 3)},
        {"chi_tieu": "Tổng thời lượng gốc (phút)", "gia_tri": round(df["thoi_luong_goc_giay"].sum() / 60, 2)},
        {"chi_tieu": "Số file một nhạc cụ (AI dự đoán, gần đúng)", "gia_tri": one_instrument if args.classify else "Chưa chạy AI"},
        {"chi_tieu": "Số file nhiều nhạc cụ (AI dự đoán, gần đúng)", "gia_tri": multi_instrument if args.classify else "Chưa chạy AI"},
        {"chi_tieu": "Nhạc cụ xuất hiện nhiều nhất (AI dự đoán)",
         "gia_tri": json.dumps(sorted(instrument_counter.items(), key=lambda x: x[1], reverse=True), ensure_ascii=False)},
        {"chi_tieu": "Phân bố thể loại (AI dự đoán)",
         "gia_tri": json.dumps(sorted(genre_counter.items(), key=lambda x: x[1], reverse=True), ensure_ascii=False)},
        {"chi_tieu": "Khoảng tempo BPM ước tính", "gia_tri": tempo_range},
        {"chi_tieu": "Tiền xử lý", "gia_tri": "Mono; 22050 Hz; lấy đoạn giữa tối đa 30 giây; peak normalize"},
        {"chi_tieu": "MFCC", "gia_tri": "13 hệ số; lưu mean/std trong CSV chi tiết"},
        {"chi_tieu": "RMS Energy", "gia_tri": "Lưu mean/std trong CSV chi tiết"},
        {"chi_tieu": "Nhận dạng AI", "gia_tri": "Có" if args.classify else "Không; chỉ trích xuất đặc trưng âm thanh"},
    ]
    summary_path = output_dir / "thong_ke_bo_du_lieu.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    if args.classify:
        pd.DataFrame(
            [{"nhac_cu_du_doan": k, "so_file": v}
             for k, v in sorted(instrument_counter.items(), key=lambda x: x[1], reverse=True)]
        ).to_csv(output_dir / "phan_bo_nhac_cu.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(
            [{"the_loai_du_doan": k, "so_file": v}
             for k, v in sorted(genre_counter.items(), key=lambda x: x[1], reverse=True)]
        ).to_csv(output_dir / "phan_bo_the_loai.csv", index=False, encoding="utf-8-sig")

    report = [
        "BÁO CÁO THỐNG KÊ BỘ DỮ LIỆU ÂM THANH",
        "=" * 42,
        f"Thư mục đầu vào: {input_dir}",
        f"Số file tìm thấy: {len(files)}",
        f"Số file thành công: {len(df)}",
        f"Số file lỗi: {len(errors)}",
        f"Tổng thời lượng gốc: {df['thoi_luong_goc_giay'].sum() / 3600:.2f} giờ",
        f"Tempo BPM ước tính: {tempo_range}",
        "Tiền xử lý: Mono, 22050 Hz, đoạn giữa tối đa 30 giây, peak normalize.",
        "MFCC: 13 hệ số; RMS: mean/std.",
        "Lưu ý: tempo là ước lượng; nhãn nhạc cụ/thể loại của AI (nếu bật) chỉ để tham khảo, cần kiểm tra thủ công.",
        "",
        "Các tệp đầu ra:",
        "- ket_qua_chi_tiet.csv",
        "- thong_ke_bo_du_lieu.csv",
        "- phan_bo_nhac_cu.csv (nếu bật --classify)",
        "- phan_bo_the_loai.csv (nếu bật --classify)",
        "- loi_xu_ly.csv (nếu có lỗi)",
    ]
    (output_dir / "bao_cao.txt").write_text("\n".join(report), encoding="utf-8")

    print("\nHOÀN TẤT")
    print(f"File xử lý thành công: {len(df)}/{len(files)}")
    print(f"Kết quả lưu tại: {output_dir}")
    print(f"- {detail_path}")
    print(f"- {summary_path}")
    if errors:
        print(f"- Có {len(errors)} file lỗi; xem {output_dir / 'loi_xu_ly.csv'}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()