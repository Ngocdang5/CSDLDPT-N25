"""config.py - Hang so va duong dan dung chung cho toan bo du an."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Tien xu ly (PHAI giong nhau cho database va file truy van; khop muc 1.3 bao cao)
SR = 22050            # Hz
DURATION = 30         # giay, cat doan giua neu file dai hon
MIN_SECONDS = 5.0     # file truy van ngan hon muc nay bi tu choi

# ---- Duong dan
AUDIO_DIR = Path(os.getenv("MUSIC_AUDIO_DIR", ROOT / "data" / "dataset_wav"))
METADATA_CSV = ROOT / "metadata.csv"
FEATURES_CSV = ROOT / "features" / "features.csv"
RESULTS_DIR = ROOT / "results"

# ---- PostgreSQL: dat bien moi truong DATABASE_URL, vi du
#      postgresql://user:matkhau@localhost:5432/musicdb
DB_DSN = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/musicdb")

# ---- Tim kiem
TOP_K = 5
GROUP_WEIGHTS = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}   # mac dinh bang nhau (muc 2.9.1)
