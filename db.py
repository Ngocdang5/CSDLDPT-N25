"""db.py - Ket noi PostgreSQL va tao luoc do (4 bang: audio, audio_feature, audio_feature_scaled, scaler_param).

    python db.py --dump-schema schema.sql     # ghi file schema.sql de nop kem bao cao
    python db.py --create --reset             # tao bang trong CSDL dang tro toi boi DATABASE_URL
"""
import argparse

import psycopg2

from audio_features import FEATURE_NAMES
from config import DB_DSN


def connect(dsn=DB_DSN):
    return psycopg2.connect(dsn)


def _feature_cols(indent="    "):
    return ",\n".join(f"{indent}{n} DOUBLE PRECISION NOT NULL" for n in FEATURE_NAMES)


def schema_sql():
    return f"""-- schema.sql : CSDL dac trung am thanh (PostgreSQL)
-- Sinh tu db.py; thu tu 71 cot theo audio_features.FEATURE_NAMES.

CREATE TABLE audio (
    audio_id    INTEGER PRIMARY KEY,
    file_name   VARCHAR(255) NOT NULL UNIQUE,
    file_path   VARCHAR(512) NOT NULL,
    format      VARCHAR(10),
    genre       VARCHAR(50),
    instrument  VARCHAR(100),
    duration    DOUBLE PRECISION,
    sample_rate INTEGER,
    channels    INTEGER,
    split       VARCHAR(10) NOT NULL CHECK (split IN ('database', 'test'))
);

-- Vector 71 chieu goc (chua chuan hoa), quan he 1-1 voi audio
CREATE TABLE audio_feature (
    audio_id INTEGER PRIMARY KEY REFERENCES audio(audio_id) ON DELETE CASCADE,
{_feature_cols()}
);

-- Vector da chuan hoa Z-score (dung cho tim kiem), cung cau truc
CREATE TABLE audio_feature_scaled (
    audio_id INTEGER PRIMARY KEY REFERENCES audio(audio_id) ON DELETE CASCADE,
{_feature_cols()}
);

-- Tham so chuan hoa tinh tren tap 'database', dung lai cho file truy van
CREATE TABLE scaler_param (
    ordinal      INTEGER PRIMARY KEY,
    feature_name VARCHAR(50) NOT NULL UNIQUE,
    grp          CHAR(1) NOT NULL CHECK (grp IN ('A', 'B', 'C')),
    mean         DOUBLE PRECISION NOT NULL,
    std          DOUBLE PRECISION NOT NULL,
    was_constant BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX idx_audio_genre ON audio(genre);
CREATE INDEX idx_audio_split ON audio(split);
"""


def create_schema(conn, reset=False):
    with conn.cursor() as cur:
        if reset:
            cur.execute("DROP TABLE IF EXISTS audio_feature_scaled, audio_feature, "
                        "scaler_param, audio CASCADE")
        cur.execute(schema_sql())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-schema", metavar="FILE")
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if a.dump_schema:
        with open(a.dump_schema, "w", encoding="utf-8") as f:
            f.write(schema_sql())
        print("Da ghi", a.dump_schema)
    if a.create:
        with connect() as conn:
            create_schema(conn, a.reset)
        print("Da tao bang.")
