"""
audio_inspect.py - Kiem tra nhanh 1 file nhac (hoac ca thu muc)

Chuc nang
  1. Lay sample_rate, so kenh, thoi luong, dinh dang.
  2. Kiem tra mono hay stereo. Neu stereo, kiem tra them 2 kenh co GIONG HET nhau
     khong ("mono gia"), vi nhieu file stereo thuc chat chi la mono nhan doi.
  3. Nhan dien cac NHAC CU xuat hien trong file bang mo hinh AudioSet da huan luyen
     san (Audio Spectrogram Transformer, Hugging Face). Day la mo hinh phan loai
     nhieu nhan: 1 file co the co nhieu nhac cu cung luc.

Cai dat
    pip install soundfile librosa numpy pandas
    pip install torch transformers          # chi can cho phan nhan dien nhac cu

Cach dung
    # 1 file: thong tin + nhac cu
    python audio_inspect.py song.wav

    # chi xem sample_rate / mono (khong tai mo hinh, chay rat nhanh)
    python audio_inspect.py song.wav --no-instruments

    # ca thu muc -> ghi them audio_inspect.csv
    python audio_inspect.py --dir data/raw --out audio_inspect.csv

    # chinh nguong / so nhac cu in ra / xem tat ca nhan cua mo hinh
    python audio_inspect.py song.wav --threshold 0.10 --top 8 --verbose

Luu y ve phan nhac cu
  - Day la ket qua du doan cua mo hinh, KHONG phai su that tuyet doi. Nen nghe lai
    mot so file de kiem tra, nhat la khi diem so thap.
  - Lan chay dau mo hinh duoc tai ve (~350 MB) va can ket noi mang.
  - Cac nhan rat gan nhau (Guitar / Electric guitar / Acoustic guitar) co the cung
    xuat hien trong 1 file.
"""
import argparse
import os
import sys

import numpy as np

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODEL_SR = 16000          # mo hinh AST dung 16 kHz
CHUNK_SEC = 10            # AST xu ly cua so 10 giay

# Cac nhan nhac cu trong AudioSet (so khop chinh xac, khong phan biet hoa/thuong).
# Khong gom giong hat/nguoi noi, va cac nhan the loai (Rock music, Jazz...).
INSTRUMENT_LABELS = {
    # day / ban phim
    "guitar", "electric guitar", "bass guitar", "acoustic guitar",
    "steel guitar, slide guitar", "banjo", "sitar", "mandolin", "zither", "ukulele",
    "piano", "electric piano", "organ", "electronic organ", "hammond organ",
    "synthesizer", "sampler", "harpsichord", "keyboard (musical)",
    "violin, fiddle", "cello", "double bass", "bowed string instrument",
    "string section", "harp", "pizzicato", "plucked string instrument",
    # hoi / dong
    "flute", "saxophone", "clarinet", "trumpet", "trombone", "french horn",
    "brass instrument", "wind instrument, woodwind instrument",
    "harmonica", "accordion", "bagpipes", "didgeridoo", "theremin",
    # bo go / go
    "drum kit", "drum machine", "drum", "snare drum", "bass drum", "timpani",
    "tabla", "cymbal", "hi-hat", "tambourine", "maraca", "gong", "percussion",
    "marimba, xylophone", "glockenspiel", "vibraphone", "steelpan",
    "mallet percussion", "tubular bells", "wood block", "bell", "chime",
    "wind chime", "orchestra", "musical instrument",
}


# ------------------------------------------------------------------ thong tin file
def read_info(path, check_seconds=30):
    """-> dict: sample_rate, channels, duration, format, is_mono, fake_stereo, note"""
    import soundfile as sf

    info = {"file": os.path.basename(path),
            "format": os.path.splitext(path)[1].lstrip(".").lower()}
    data = None
    try:
        i = sf.info(path)
        info.update(sample_rate=int(i.samplerate), channels=int(i.channels),
                    duration=round(float(i.duration), 2))
        if i.channels > 1:
            frames = min(int(i.frames), int(i.samplerate * check_seconds))
            data, _ = sf.read(path, frames=frames, always_2d=True)
    except Exception:
        # du phong (vi du mp3/m4a ma soundfile khong doc duoc)
        import librosa
        y, sr = librosa.load(path, sr=None, mono=False)
        y = np.atleast_2d(y)
        info.update(sample_rate=int(sr), channels=int(y.shape[0]),
                    duration=round(y.shape[1] / sr, 2))
        data = y.T if y.shape[0] > 1 else None

    info["is_mono"] = info["channels"] == 1
    info["fake_stereo"] = False
    if info["channels"] == 1:
        info["note"] = "mono (1 kenh)"
    else:
        info["fake_stereo"] = channels_identical(data)
        if info["fake_stereo"]:
            info["note"] = (f"{info['channels']} kenh nhung cac kenh GIONG HET nhau "
                            "(thuc chat la mono)")
        else:
            info["note"] = f"stereo that ({info['channels']} kenh, noi dung khac nhau)"
    return info


def channels_identical(data, tol=1e-4):
    """data: (n_samples, n_channels). True neu moi kenh gan nhu giong kenh dau."""
    if data is None or data.shape[1] < 2:
        return False
    ref = data[:, 0]
    return all(np.max(np.abs(data[:, c] - ref)) <= tol for c in range(1, data.shape[1]))


# --------------------------------------------------------------- nhan dien nhac cu
def load_model():
    try:
        import torch
        from transformers import ASTForAudioClassification, AutoFeatureExtractor
    except ImportError:
        sys.exit("Thieu thu vien. Chay:  pip install torch transformers")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ASTForAudioClassification.from_pretrained(MODEL_NAME).eval()
    return torch, extractor, model


def predict_chunks(path, max_seconds=60, model_bundle=None):
    """Chia file thanh cac cua so 10 giay -> ma tran diem (so_cua_so x 527 nhan)."""
    import librosa
    torch, extractor, model = model_bundle or load_model()

    y, _ = librosa.load(path, sr=MODEL_SR, mono=True)
    n = MODEL_SR * max_seconds
    if len(y) > n:                                   # lay doan giua cho nhanh
        s = (len(y) - n) // 2
        y = y[s:s + n]
    step = MODEL_SR * CHUNK_SEC
    chunks = [y[i:i + step] for i in range(0, len(y), step)]
    if len(chunks) > 1 and len(chunks[-1]) < step // 2:   # bo doan cuoi qua ngan
        chunks = chunks[:-1]

    scores = []
    with torch.no_grad():
        for c in chunks:
            feats = extractor(c, sampling_rate=MODEL_SR, return_tensors="pt")
            logits = model(**feats).logits
            scores.append(torch.sigmoid(logits)[0].numpy())   # da nhan: sigmoid
    return np.array(scores), model.config.id2label


def summarize_instruments(scores, id2label, threshold=0.10, top=10, only_instruments=True):
    """scores: (n_chunks, n_labels). Tra ve list [(nhan, diem_tb, ti_le_cua_so_co_mat)]."""
    mean = scores.mean(axis=0)
    present = (scores >= threshold).mean(axis=0)
    out = []
    for idx in np.argsort(-mean):
        label = id2label[int(idx)]
        if only_instruments and label.lower() not in INSTRUMENT_LABELS:
            continue
        if mean[idx] < threshold:
            break
        out.append((label, float(mean[idx]), float(present[idx])))
        if len(out) >= top:
            break
    return out


# ------------------------------------------------------------------------- in an
def print_report(info, instruments=None):
    print(f"File         : {info['file']}")
    print(f"Dinh dang    : {info['format']}")
    print(f"Sample rate  : {info['sample_rate']} Hz")
    print(f"So kenh      : {info['channels']}")
    print(f"Mono?        : {'CO' if info['is_mono'] else 'KHONG'}  -> {info['note']}")
    print(f"Thoi luong   : {info['duration']} giay")
    if instruments is not None:
        print("Nhac cu      :", end=" ")
        if not instruments:
            print("khong co nhan nhac cu nao vuot nguong")
        else:
            print()
            for label, score, frac in instruments:
                print(f"   - {label:<34} diem {score:.2f}   co mat o {frac:.0%} thoi luong")


def find_audio(root):
    out = []
    for dp, _, fns in os.walk(root):
        out += [os.path.join(dp, f) for f in fns
                if os.path.splitext(f)[1].lower() in AUDIO_EXT]
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description="Kiem tra sample rate, mono va nhac cu")
    ap.add_argument("path", nargs="?", help="duong dan 1 file audio")
    ap.add_argument("--dir", help="quet ca thu muc thay vi 1 file")
    ap.add_argument("--out", default="audio_inspect.csv", help="CSV khi dung --dir")
    ap.add_argument("--no-instruments", action="store_true",
                    help="bo qua nhan dien nhac cu (nhanh, khong can torch)")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="diem toi thieu de coi nhac cu la co mat (mac dinh 0.10)")
    ap.add_argument("--top", type=int, default=10, help="so nhac cu toi da in ra")
    ap.add_argument("--max_seconds", type=int, default=60,
                    help="chi phan tich toi da so giay (lay doan giua)")
    ap.add_argument("--verbose", action="store_true",
                    help="in them 10 nhan co diem cao nhat cua mo hinh (ke ca khong phai nhac cu)")
    a = ap.parse_args()

    if not a.path and not a.dir:
        ap.error("can truyen 1 file hoac --dir thu_muc")

    files = find_audio(a.dir) if a.dir else [a.path]
    if not files:
        sys.exit("Khong tim thay file audio nao.")

    bundle = None if a.no_instruments else load_model()
    rows = []
    for p in files:
        try:
            info = read_info(p)
        except Exception as e:
            print(f"[LOI] {p}: {type(e).__name__}: {e}")
            continue
        inst = None
        if bundle is not None:
            try:
                scores, id2label = predict_chunks(p, a.max_seconds, bundle)
                inst = summarize_instruments(scores, id2label, a.threshold, a.top)
                if a.verbose:
                    mean = scores.mean(axis=0)
                    info["top_labels"] = [(id2label[int(i)], float(mean[i]))
                                          for i in np.argsort(-mean)[:10]]
            except Exception as e:
                print(f"[LOI nhac cu] {p}: {type(e).__name__}: {e}")
        if not a.dir:
            print_report(info, inst)
            if a.verbose and "top_labels" in info:
                print("Top nhan mo hinh:")
                for lab, s in info["top_labels"]:
                    print(f"   {lab:<34} {s:.2f}")
        row = {k: info[k] for k in ("file", "format", "sample_rate", "channels",
                                    "is_mono", "fake_stereo", "duration")}
        row["instruments"] = ("; ".join(f"{l} ({s:.2f})" for l, s, _ in inst)
                              if inst is not None else "")
        row["path"] = p
        rows.append(row)

    if a.dir:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(a.out, index=False, encoding="utf-8-sig")
        print(f"Da quet {len(df)} file -> {a.out}")
        print("Sample rate:", ", ".join(f"{k} Hz x{v}"
                                        for k, v in df['sample_rate'].value_counts().items()))
        print("Mono:", int(df['is_mono'].sum()), "| stereo:", int((~df['is_mono']).sum()),
              "| stereo nhung 2 kenh giong het:", int(df['fake_stereo'].sum()))


if __name__ == "__main__":
    main()
