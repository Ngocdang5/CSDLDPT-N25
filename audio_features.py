"""
audio_features.py - Tien xu ly + trich xuat vector 71 chieu (muc 2 bao cao).

Bo A (42): MFCC 26, Spectral Contrast 14, Spectral Centroid 2   -> pho va am sac
Bo B (24): Chroma 24                                            -> hoa am
Bo C (5) : ZCR 2, RMS 2, Tempo 1                                -> nhip, nang luong, dang song

Database va file truy van PHAI di qua dung cac ham nay.
"""
import librosa
import numpy as np

from config import DURATION, MIN_SECONDS, SR


# ---------------------------------------------------------------- ten cot
def feature_names():
    """Thu tu CO DINH cua 71 chieu (trung voi cot trong schema.sql)."""
    names = []
    for i in range(1, 14):
        names += [f"mfcc_{i}_mean", f"mfcc_{i}_std"]            # 26  (mfcc_1 = MFCC 0)
    for i in range(1, 8):
        names += [f"contrast_{i}_mean", f"contrast_{i}_std"]    # 14
    names += ["centroid_mean", "centroid_std"]                  # 2
    for i in range(1, 13):
        names += [f"chroma_{i}_mean", f"chroma_{i}_std"]        # 24
    names += ["zcr_mean", "zcr_std", "rms_mean", "rms_std"]     # 4
    names += ["tempo"]                                          # 1
    return names


FEATURE_NAMES = feature_names()
N_FEATURES = len(FEATURE_NAMES)   # 71


def group_of(name):
    if name.startswith(("mfcc_", "contrast_", "centroid_")):
        return "A"
    if name.startswith("chroma_"):
        return "B"
    return "C"


GROUPS = ("A", "B", "C")
GROUP_OF = [group_of(n) for n in FEATURE_NAMES]
GROUP_INDEX = {g: np.array([i for i, x in enumerate(GROUP_OF) if x == g]) for g in GROUPS}
GROUP_DIMS = {g: len(ix) for g, ix in GROUP_INDEX.items()}      # {'A': 42, 'B': 24, 'C': 5}
GROUP_TITLES = {"A": "Pho va am sac", "B": "Hoa am", "C": "Nhip, nang luong, dang song"}


# ---------------------------------------------------------------- tien xu ly
def load_audio(path):
    """Doc file -> mono, SR Hz (chua cat, chua chuan hoa)."""
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def preprocess_array(y, duration=DURATION):
    """Cat doan giua toi da `duration` giay, chuan hoa dinh ve [-1, 1]."""
    if y.size < int(MIN_SECONDS * SR):
        raise ValueError(f"File qua ngan ({y.size / SR:.1f}s < {MIN_SECONDS}s)")
    n = int(duration * SR)
    if y.size > n:
        s = (y.size - n) // 2
        y = y[s:s + n]
    peak = float(np.max(np.abs(y)))
    if peak <= 0:
        raise ValueError("File im lang hoan toan")
    return (y / peak).astype(np.float32)


def preprocess(path, duration=DURATION):
    return preprocess_array(load_audio(path), duration)


# ---------------------------------------------------------------- trich xuat
def _ms(matrix):
    """mean, std theo truc thoi gian, xen ke [mean1, std1, mean2, std2, ...]."""
    m = matrix.mean(axis=1)
    s = matrix.std(axis=1)
    return np.column_stack([m, s]).ravel().tolist()


def extract_features(y, sr=SR):
    """y da tien xu ly -> vector float64 do dai 71 theo FEATURE_NAMES."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)           # 7 dai
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    vec = (_ms(mfcc) + _ms(contrast) + _ms(centroid) + _ms(chroma)
           + _ms(zcr) + _ms(rms) + [tempo])
    vec = np.asarray(vec, dtype=np.float64)
    if vec.size != N_FEATURES:
        raise AssertionError(f"Sai so chieu vector: {vec.size} != {N_FEATURES}")
    if not np.all(np.isfinite(vec)):
        raise ValueError("Vector co NaN hoac vo cuc")
    return vec


def extract_from_file(path, *, include_waveform=False):
    """Run the file pipeline, optionally returning the samples used for extraction."""
    y = preprocess(path)
    raw = extract_features(y)
    return (raw, y) if include_waveform else raw
