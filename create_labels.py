
"""
create_labels.py

Gan nhan tu dong cho bo du lieu am thanh
dua tren metadata chinh thuc cua FMA.

Dau vao:
    data/audio/
    fma_metadata/tracks.csv
    fma_metadata/genres.csv

Dau ra:
    labels.csv
    unmatched_labels.csv
    label_statistics.cs
Chay:
    python create_labels.py
"""

import ast
from pathlib import Path

import pandas as pd

from config import AUDIO_DIR


ROOT = Path(__file__).resolve().parent

FMA_DIR = ROOT / "fma_metadata"

TRACKS_CSV = FMA_DIR / "tracks.csv"
GENRES_CSV = FMA_DIR / "genres.csv"

LABELS_CSV = ROOT / "labels.csv"
UNMATCHED_CSV = ROOT / "unmatched_labels.csv"
STATISTICS_CSV = ROOT / "label_statistics.csv"

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg"
}


def check_input():

    required_paths = [
        AUDIO_DIR,
        TRACKS_CSV,
        GENRES_CSV
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Khong tim thay: {path}"
            )

    print("Kiem tra du lieu dau vao: OK")

def load_fma_metadata():

    print("\nDang doc tracks.csv...")

    tracks = pd.read_csv(
        TRACKS_CSV,
        index_col=0,
        header=[0, 1],
        low_memory=False
    )

    print("Dang doc genres.csv...")

    genres = pd.read_csv(
        GENRES_CSV,
        index_col=0
    )

    # Chuan hoa ID thanh so nguyen
    tracks.index = tracks.index.astype(int)
    genres.index = genres.index.astype(int)

    # Tao bang tra cuu ma the loai -> ten the loai
    genre_map = genres["title"].to_dict()

    print(f"So track trong FMA: {len(tracks)}")
    print(f"So the loai trong FMA: {len(genres)}")

    return tracks, genre_map

def clean_value(value, default="Unknown"):

    if pd.isna(value):
        return default

    value = str(value).strip()

    return value if value else default


def get_track_id(path):

    # Vi du: 014569.wav -> 14569

    if path.stem.isdigit():
        return int(path.stem)

    return None


def decode_genres(value, genre_map):

    # Chuyen danh sach ID the loai
    # thanh danh sach ten the loai

    if pd.isna(value):
        return "Unknown"

    try:
        genre_ids = ast.literal_eval(str(value))

        if not isinstance(genre_ids, (list, tuple)):
            return "Unknown"

        genre_names = []

        for genre_id in genre_ids:

            name = genre_map.get(int(genre_id))

            if name and name not in genre_names:
                genre_names.append(name)

        if genre_names:
            return ";".join(genre_names)

    except (ValueError, SyntaxError, TypeError):
        pass

    return "Unknown"

def label_audio(path, tracks, genre_map):

    track_id = get_track_id(path)

    result = {
        "track_id": track_id,
        "file_name": path.name,
        "file_path": path.relative_to(
            AUDIO_DIR
        ).as_posix(),

        "title": "Unknown",
        "artist": "Unknown",
        "album": "Unknown",

        "genre": "Unknown",
        "genres_all": "Unknown",

        "instrument": "Unknown",

        "genre_source": "Unknown",
        "label_status": "not_found"
    }

    # Ten file khong phai ID cua FMA
    if track_id is None:

        result["label_status"] = "invalid_track_id"

        return result

    # Khong tim thay track trong FMA
    if track_id not in tracks.index:

        return result

    # Lay thong tin cua track
    row = tracks.loc[track_id]

    result["title"] = clean_value(
        row[("track", "title")]
    )

    result["artist"] = clean_value(
        row[("artist", "name")]
    )

    result["album"] = clean_value(
        row[("album", "title")]
    )

    result["genre"] = clean_value(
        row[("track", "genre_top")]
    )

    result["genres_all"] = decode_genres(
        row[("track", "genres_all")],
        genre_map
    )

    if result["genre"] != "Unknown":

        result["genre_source"] = "FMA"
        result["label_status"] = "matched"

    else:

        result["label_status"] = "missing_genre"

    return result

def create_labels(tracks, genre_map):

    audio_files = sorted(
        path
        for path in AUDIO_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
    )

    total = len(audio_files)

    print(f"\nTong so file am thanh: {total}")

    if total == 0:
        raise RuntimeError(
            "Khong tim thay file am thanh."
        )

    labels = []

    for index, path in enumerate(audio_files, start=1):

        result = label_audio(
            path,
            tracks,
            genre_map
        )

        labels.append(result)

        print(
            f"[{index}/{total}] "
            f"{path.name} | "
            f"{result['genre']} | "
            f"{result['label_status']}"
        )

    df = pd.DataFrame(labels)

    df["track_id"] = df["track_id"].astype("Int64")

    return df

def save_results(df):

    # Luu toan bo nhan
    df.to_csv(
        LABELS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # Luu cac file chua gan duoc nhan the loai
    unmatched = df[
        df["label_status"] != "matched"
    ]

    unmatched.to_csv(
        UNMATCHED_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # Thong ke so luong theo the loai
    statistics = (
        df["genre"]
        .value_counts()
        .rename_axis("genre")
        .reset_index(name="count")
    )

    statistics["percentage"] = (
        statistics["count"] / len(df) * 100
    ).round(2)

    statistics.to_csv(
        STATISTICS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    return statistics

def print_report(df, statistics):

    total = len(df)

    matched = (
        df["label_status"] == "matched"
    ).sum()

    unmatched = total - matched

    print("\n" + "=" * 60)
    print("BAO CAO KET QUA GAN NHAN")
    print("=" * 60)

    print(f"Tong so file: {total}")
    print(f"Gan nhan thanh cong: {matched}")
    print(f"Chua co nhan: {unmatched}")

    print(
        f"Ty le thanh cong: "
        f"{matched / total * 100:.2f}%"
    )

    print("\nThong ke trang thai:")
    print(df["label_status"].value_counts())

    print("\nThong ke the loai:")
    print(statistics.to_string(index=False))

    print("\nCac file dau ra:")

    print(f"1. {LABELS_CSV}")
    print(f"2. {UNMATCHED_CSV}")
    print(f"3. {STATISTICS_CSV}")

    print("\nHoan thanh gan nhan!")

def main():

    check_input()

    tracks, genre_map = load_fma_metadata()

    df = create_labels(
        tracks,
        genre_map
    )

    statistics = save_results(df)

    print_report(
        df,
        statistics
    )


if __name__ == "__main__":
    main()