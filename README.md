# He CSDL luu tru va tim kiem doan nhac khong loi (Nhom 25)

Moi doan nhac duoc bieu dien bang vector 71 chieu (7 dac trung, muc 2 bao cao). Vector luu trong PostgreSQL.
Tim kiem: vector cua file moi duoc chuan hoa bang mean/std cua CSDL, so voi cac vector da luu bang khoang cach
co trong so theo 3 bo (A 42 chieu, B 24 chieu, C 5 chieu), tra ve 5 file gan nhat.

## Cau truc

| File | Vai tro | Yeu cau |
|---|---|---|
| `config.py` | Hang so (SR 22 050 Hz, 30 giay), duong dan, chuoi ket noi PostgreSQL | |
| `audio_features.py` | Tien xu ly va trich xuat 71 chieu, ten cot, 3 bo A/B/C | 2, 3 |
| `build_metadata.py` | Tao `metadata.csv`, chia tap `database`/`test`, bieu do thong ke | 1 |
| `extract_all.py` | Trich xuat 71 chieu cho tat ca file (chay song song, chay tiep duoc) | 3 |
| `scaling.py` | Chuan hoa Z-score (chieu hang so dat std = 1) | 3, 4 |
| `db.py`, `schema.sql` | Luoc do PostgreSQL: `audio`, `audio_feature`, `audio_feature_scaled`, `scaler_param` | 3 |
| `load_db.py` | Nap CSDL, tinh va luu tham so chuan hoa, kiem tra | 3 |
| `queries.sql` | Truy van SQL mau de chup man hinh bao cao | 3 |
| `search.py` | Tim top 5 (dong lenh); lop `SearchEngine` | 4 |
| `explain_query.py` | Ket qua trung gian cua mot truy van (bang va 2 hinh) | 4b |
| `evaluate.py` | Danh gia: tu truy hoi, so sanh trong so, thoi gian | 4c |
| `app.py` | Giao dien demo Streamlit | 5 |
| `audio_inspect.py` | Kiem tra sample rate, mono/stereo gia (tuy chon) | 1 |
| `make_demo_data.py` | Sinh am thanh tong hop de thu pipeline (khong phai du lieu that) | |
| `tests/` | pytest kiem tra bo vector, chuan hoa, khoang cach, top 5 | |

## Cai dat

```
python -m venv .venv
.venv\Scripts\activate                 (Windows)   |   source .venv/bin/activate   (Linux, macOS)
pip install -r requirements.txt
```

Dung `librosa` ban 0.10 den 0.11 (da pin `<1.0`). Ban 1.0 di cung numba rat moi co the lam chet tien trinh
o buoc tempo.

PostgreSQL:
```
createdb musicdb                          (hoac tao bang pgAdmin)
set DATABASE_URL=postgresql://postgres:MATKHAU@localhost:5432/musicdb      (Windows)
export DATABASE_URL=postgresql://postgres:MATKHAU@localhost:5432/musicdb   (Linux, macOS)
```

## Chay tu dau den cuoi (596 file that)

Dat 596 file `.wav` vao `data/dataset_wav/` (co the nam trong thu muc con), hoac dat `MUSIC_AUDIO_DIR`.

```
# 1) Metadata + thong ke. Mac dinh --test_ratio 0: tat ca file vao tap tim kiem.
#    Neu muon tach mot phan lam truy van thu, them --test_ratio 0.1. Co nhan the loai thi them --labels labels.csv
python build_metadata.py --audio_dir data/dataset_wav

# 2) Trich xuat 71 chieu cho ca 596 file -> features/features.csv (loi ghi vao features/errors.csv)
python extract_all.py --jobs 4

# 3) Nap PostgreSQL (tat ca 596 file duoc luu; chuan hoa chi tinh tren tap database)
python load_db.py --reset
psql -d musicdb -f queries.sql           # truy van mau cho bao cao

# 4) Tim kiem, ket qua trung gian, danh gia
python search.py data/dataset_wav/<file>.wav
python explain_query.py data/dataset_wav/<file>.wav
python evaluate.py --n 100

# 5) Demo
streamlit run app.py
```

Thu nhanh khong can PostgreSQL: them `--source csv` vao `search.py`, `explain_query.py`, `evaluate.py`,
hoac `APP_SOURCE=csv streamlit run app.py`.

Thu pipeline bang du lieu tong hop: `python make_demo_data.py --n 60 --labels labels.csv`, roi chay lai cac buoc
tren. Kiem tra ma: `python -m pytest -q tests`. Test PostgreSQL chi chay khi dat `TEST_DATABASE_URL` va no XOA
cac bang cua CSDL do, nen chi tro vao mot CSDL thu rieng.

## Luu y

- Thu tu 71 cot khac file `feature_extraction.py` cu (bo A xep lien nhau); khong tron `features.csv` cu voi ban moi.
  `mfcc_1` la MFCC 0 (13 he so, gom he so 0).
- Tien xu ly (mono, 22 050 Hz, cat 30 giay o giua, chuan hoa dinh) dung cung mot ham cho CSDL va file truy van.
- Do tuong dong hien thi la `1 / (1 + D)`, chi dung de xep hang va hien thi.
- Tempo lay tu `librosa.beat.beat_track`; gia tri cuc doan (46 hoac 235 BPM) co the la loi gap doi hoac mot nua.
- Phep tu truy hoi trong `evaluate.py` kha de (file goc phai nam trong top). Chi so "cung the loai" chi la
  tham chieu, khong phai dap an dung cho "giong nhau".
- Ma nay moi duoc chay thu bang am thanh tong hop (60 file) tren PostgreSQL 16. Chua chay tren 500 file that
  cua nhom, nen chua co so lieu that nao trong thu muc nay.
