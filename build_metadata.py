"""
build_metadata.py - Nguoi 3 (Giai doan 1): tao metadata.csv + thong ke dataset

Chuc nang
  1. Quet thu muc audio, doc duration / sample_rate / so kenh cua tung file.
  2. Xac dinh genre: (a) tu file --labels, (b) tu ten thu muc con (kieu GTZAN
     genres/blues/*.wav), (c) tu tien to ten file (blues.00001.wav), neu khong
     duoc thi "unknown".
  3. Loai file loi / qua ngan (ghi ra rejected.csv, KHONG dua vao metadata.csv).
  4. Chia tap: "database" (nap CSDL) va "test" (chi dung lam query),
     phan tang theo genre, co seed de lap lai duoc.
  5. Xuat metadata.csv + thu muc stats/ gom bang tom tat va 3 bieu do.

Cach dung
    python build_metadata.py --audio_dir data/raw
    python build_metadata.py --audio_dir data/raw --labels labels.csv --test_ratio 0.2

labels.csv (tuy chon) co cot:  file_name,genre[,instrument]

Dau ra
    metadata.csv      audio_id,file_name,file_path,format,genre,instrument,
                      duration,sample_rate,channels,split
    rejected.csv      file bi loai + ly do
    stats/summary.txt, stats/genre_table.csv,
    stats/genre_distribution.png, stats/duration_hist.png, stats/split_by_genre.png
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}

# Mau theo bang mau tham chieu (slot 1 xanh duong, slot 2 cam: da kiem dinh CVD)
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------- quet file
def find_audio(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in AUDIO_EXT:
                out.append(os.path.join(dp, fn))
    return sorted(out, key=lambda p: os.path.relpath(p, root).lower())


def infer_genre(path, root):
    parts = os.path.relpath(path, root).split(os.sep)
    if len(parts) > 1:                       # genres/blues/xxx.wav -> "blues"
        return parts[0]
    stem = os.path.splitext(parts[0])[0]
    if "." in stem:                          # blues.00001 -> "blues"
        return stem.split(".")[0]
    return "unknown"


def audio_info(path):
    """-> (duration_s, sample_rate, channels). Nem loi neu file hong."""
    try:
        i = sf.info(path)
        return float(i.duration), int(i.samplerate), int(i.channels)
    except Exception:
        import librosa                       # du phong cho mp3 neu soundfile khong doc duoc
        sr = int(librosa.get_samplerate(path))
        dur = float(librosa.get_duration(path=path))
        return dur, sr, np.nan


# ------------------------------------------------------------ tao metadata
def build_table(audio_dir, labels_csv, min_duration):
    labels = {}
    if labels_csv:
        lab = pd.read_csv(labels_csv)
        for c in ("file_name", "genre"):
            if c not in lab.columns:
                sys.exit(f"labels.csv thieu cot '{c}'")
        labels = lab.set_index("file_name").to_dict("index")

    rows, rejected = [], []
    for p in find_audio(audio_dir):
        name = os.path.basename(p)
        rel = os.path.relpath(p, audio_dir).replace(os.sep, "/")
        try:
            dur, sr, ch = audio_info(p)
        except Exception as e:
            rejected.append((name, rel, f"khong doc duoc: {type(e).__name__}"))
            continue
        if dur < min_duration:
            rejected.append((name, rel, f"qua ngan ({dur:.1f}s < {min_duration}s)"))
            continue
        lab = labels.get(name, {})
        rows.append({
            "file_name": name,
            "file_path": os.path.relpath(p, audio_dir).replace(os.sep, "/"),
            "format": os.path.splitext(name)[1].lstrip(".").lower(),
            "genre": lab.get("genre") or infer_genre(p, audio_dir),
            "instrument": lab.get("instrument", ""),
            "duration": round(dur, 2),
            "sample_rate": sr,
            "channels": ch,
        })

    df = pd.DataFrame(rows)
    rej = pd.DataFrame(rejected, columns=["file_name", "file_path", "reason"])
    if df.empty:
        sys.exit("Khong tim thay file audio hop le nao.")

    dup = df[df.duplicated("file_name", keep=False)]
    if len(dup):
        print(f"CANH BAO: {dup.file_name.nunique()} ten file bi trung "
              f"(vi du: {dup.file_name.iloc[0]}). Doi ten de tranh nham lan.")
    df.insert(0, "audio_id", np.arange(1, len(df) + 1))
    return df, rej


def assign_split(df, test_ratio, seed):
    """Phan tang theo genre: moi genre lay ~test_ratio file lam test."""
    if test_ratio <= 0:                      # --test_ratio 0: tat ca file vao tap database
        df["split"] = "database"
        return df
    rng = np.random.default_rng(seed)
    split = pd.Series("database", index=df.index)
    for _, idx in df.groupby("genre").groups.items():
        idx = np.array(list(idx))
        if len(idx) < 2:
            continue
        n_test = max(1, int(round(len(idx) * test_ratio)))
        split.loc[rng.choice(idx, size=n_test, replace=False)] = "test"
    df["split"] = split
    return df


# ------------------------------------------------------------------ thong ke
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(w=7.5, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def chart_genre(df, out):
    counts = df["genre"].value_counts().sort_values(ascending=False)
    fig, ax = new_fig(max(6.5, 0.8 * len(counts) + 2))
    bars = ax.bar(counts.index, counts.values, color=BLUE, width=0.6)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom",
                color=INK, fontsize=9)
    ax.set_title(f"So file theo the loai (tong {len(df)} file)", loc="left",
                 color=INK, fontsize=12, pad=12)
    ax.set_ylabel("So file", color=INK2, fontsize=9)
    ax.set_ylim(0, counts.max() * 1.15)
    plt.setp(ax.get_xticklabels(), rotation=30 if len(counts) > 6 else 0, ha="right"
             if len(counts) > 6 else "center")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def chart_duration(df, out):
    fig, ax = new_fig()
    d = df["duration"]
    bins = min(30, max(5, int(np.sqrt(len(d)) * 1.5)))
    ax.hist(d, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    med = d.median()
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)          # chua cho de nhan phia tren cac cot
    ax.axvline(med, color=INK2, linewidth=1.2, linestyle="--")
    ax.text(med, ax.get_ylim()[1] * 0.98, f" trung vi {med:.1f}s", color=INK,
            fontsize=9, va="top")
    ax.set_title("Phan bo do dai file (giay)", loc="left", color=INK, fontsize=12, pad=12)
    ax.set_xlabel("Do dai (giay)", color=INK2, fontsize=9)
    ax.set_ylabel("So file", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def chart_split(df, out):
    t = (df.groupby(["genre", "split"]).size().unstack(fill_value=0)
         .reindex(columns=["database", "test"], fill_value=0))
    t = t.loc[t.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = new_fig(max(6.5, 0.8 * len(t) + 2))
    ax.bar(t.index, t["database"], color=BLUE, width=0.6, edgecolor=SURFACE,
           linewidth=2, label="database (nap CSDL)")
    ax.bar(t.index, t["test"], bottom=t["database"], color=ORANGE, width=0.6,
           edgecolor=SURFACE, linewidth=2, label="test (chi dung lam query)")
    ax.set_title("Chia tap database / test theo the loai", loc="left", color=INK,
                 fontsize=12, pad=12)
    ax.set_ylabel("So file", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    ax.set_ylim(0, t.sum(axis=1).max() * 1.2)
    plt.setp(ax.get_xticklabels(), rotation=30 if len(t) > 6 else 0, ha="right"
             if len(t) > 6 else "center")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def write_stats(df, rej, stats_dir):
    os.makedirs(stats_dir, exist_ok=True)
    g = df.groupby("genre").agg(
        tong=("audio_id", "count"),
        database=("split", lambda s: int((s == "database").sum())),
        test=("split", lambda s: int((s == "test").sum())),
        duration_tb=("duration", "mean"),
    ).sort_values("tong", ascending=False).round(2)
    g.to_csv(os.path.join(stats_dir, "genre_table.csv"))

    d = df["duration"]
    lines = [
        f"Tong so file hop le : {len(df)}",
        f"File bi loai        : {len(rej)}",
        f"So the loai         : {df['genre'].nunique()}",
        f"Do dai (giay)       : min {d.min():.1f} | tb {d.mean():.1f} | trung vi "
        f"{d.median():.1f} | max {d.max():.1f} | tong {d.sum() / 60:.1f} phut",
        "Sample rate         : " + ", ".join(f"{k} Hz x{v}" for k, v in
                                             df['sample_rate'].value_counts().items()),
        "Dinh dang           : " + ", ".join(f"{k} x{v}" for k, v in
                                             df['format'].value_counts().items()),
        "Tap chia            : " + ", ".join(f"{k} x{v}" for k, v in
                                             df['split'].value_counts().items()),
        "", "Bang theo the loai:", g.to_string(),
    ]
    if (df["genre"] == "unknown").any():
        lines.insert(0, f"CANH BAO: {(df['genre'] == 'unknown').sum()} file khong xac "
                        f"dinh duoc genre -> bo sung --labels labels.csv")
    txt = "\n".join(lines)
    with open(os.path.join(stats_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    chart_genre(df, os.path.join(stats_dir, "genre_distribution.png"))
    chart_duration(df, os.path.join(stats_dir, "duration_hist.png"))
    chart_split(df, os.path.join(stats_dir, "split_by_genre.png"))
    return txt


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--labels", default=None, help="CSV: file_name,genre[,instrument]")
    ap.add_argument("--out", default="metadata.csv")
    ap.add_argument("--stats_dir", default="stats")
    ap.add_argument("--min_duration", type=float, default=5.0)
    ap.add_argument("--test_ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df, rej = build_table(a.audio_dir, a.labels, a.min_duration)
    df = assign_split(df, a.test_ratio, a.seed)
    df.to_csv(a.out, index=False)
    rej.to_csv("rejected.csv", index=False)
    print(write_stats(df, rej, a.stats_dir))
    print(f"\nDa luu {a.out} ({len(df)} dong), rejected.csv ({len(rej)} dong), {a.stats_dir}/")
