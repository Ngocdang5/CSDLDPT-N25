# Hệ CSDL lưu trữ và tìm kiếm đoạn nhạc không lời (Nhóm 25)

Mỗi đoạn nhạc được biểu diễn bằng vector 71 chiều (7 đặc trưng âm thanh, mục 2 báo cáo) lưu trong PostgreSQL.
Khi tìm kiếm, vector của file truy vấn được chuẩn hóa Z-score bằng `mean`/`std` của CSDL, so sánh với các vector đã lưu theo khoảng cách có trọng số trên 3 nhóm đặc trưng (**A**: 42 chiều phổ & âm sắc, **B**: 24 chiều hòa âm, **C**: 5 chiều nhịp & năng lượng), và trả về Top-K (mặc định 5) đoạn nhạc gần nhất.

## Cấu trúc dự án

| File | Vai trò | Yêu cầu |
|---|---|---|
| `config.py` | Hằng số tiền xử lý (`SR=22050`, `DURATION=30s`, `MIN_SECONDS=5s`), đường dẫn, `DATABASE_URL` | |
| `audio_features.py` | Tiền xử lý và trích xuất vector 71 chiều, định nghĩa 3 nhóm A/B/C | 2, 3 |
| `build_metadata.py` | Quét thư mục audio, tạo `metadata.csv` (mặc định `test_ratio=0`), xuất thống kê `stats/` | 1 |
| `extract_all.py` | Trích xuất 71 chiều song song cho toàn bộ file; hỗ trợ chạy tiếp, kiểm tra file cũ và `--force` | 3 |
| `scaling.py` | Chuẩn hóa Z-score (chiều hằng số đặt `std = 1`) | 3, 4 |
| `db.py`, `schema.sql` | Lược đồ PostgreSQL (`audio`, `audio_feature`, `audio_feature_scaled`, `scaler_param`), quản lý transaction theo caller | 3 |
| `load_db.py` | Nạp CSDL PostgreSQL, tính và lưu tham số chuẩn hóa trên tập `database`, đối chiếu số lượng | 3 |
| `queries.sql` | Các truy vấn SQL mẫu phục vụ báo cáo | 3 |
| `search.py` | Tìm Top-K qua CLI và lớp `SearchEngine` (kiểm tra trọng số không âm, vector 71 chiều, `k >= 1`) | 4 |
| `explain_query.py` | Xuất kết quả trung gian của một truy vấn (bảng và biểu đồ) | 4b |
| `evaluate.py` | Đánh giá 4 nhóm thực nghiệm: E1 (tự truy hồi nguyên bản & đoạn cắt), E2 (truy vấn mới ngoài CSDL), E3 (trọng số), E4 (hiệu năng) | 4c |
| `app.py` | Giao diện demo Streamlit (có nút tải lại CSDL, fallback sang CSV khi DB lỗi) | 5 |
| `audio_inspect.py` | Kiểm tra sample rate, mono/stereo giả (tùy chọn) | 1 |
| `make_demo_data.py` | Sinh âm thanh tổng hợp để kiểm thử pipeline | |
| `tests/test_pipeline.py` | Bộ kiểm thử `pytest` cho trích xuất, chuẩn hóa, tìm kiếm, đánh giá và nạp/truy vấn PostgreSQL | |

## Cài đặt môi trường

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # Linux, macOS
pip install -r requirements.txt
```

> Lưu ý: Dùng `librosa` bản `0.10` đến `<1.0` (đã giới hạn trong `requirements.txt`).

Thiết lập PostgreSQL và biến môi trường `DATABASE_URL` (không hardcode mật khẩu trong `config.py`):

```bash
createdb musicdb
set DATABASE_URL=postgresql://postgres:MATKHAU@localhost:5432/musicdb      # Windows (CMD)
# export DATABASE_URL=postgresql://postgres:MATKHAU@localhost:5432/musicdb # Linux, macOS
```

## Quy trình chạy đầy đủ (596 file)

Đặt 596 file `.wav` vào `data/dataset_wav/` (hoặc đặt biến môi trường `MUSIC_AUDIO_DIR`).

```bash
# 1) Tạo metadata.csv + thống kê (mặc định --test_ratio 0.0: toàn bộ 596 file vào tập database)
python build_metadata.py --audio_dir data/dataset_wav

# 2) Trích xuất 71 đặc trưng -> features/features.csv (lỗi ghi vào features/errors.csv)
#    Thêm --force nếu muốn trích xuất lại toàn bộ từ đầu
python extract_all.py --jobs 4

# 3) Nạp dữ liệu vào PostgreSQL
python load_db.py --reset
psql -d musicdb -f queries.sql

# 4) Tìm kiếm và giải thích truy vấn
python search.py data/dataset_wav/<file>.wav
python explain_query.py data/dataset_wav/<file>.wav

# 5) Chạy giao diện demo
streamlit run app.py
```

Chạy nhanh không cần PostgreSQL: thêm `--source csv` vào `search.py`, `explain_query.py`, `evaluate.py`, hoặc chạy `APP_SOURCE=csv streamlit run app.py`.

## Đánh giá hệ thống (`evaluate.py`)

`evaluate.py` hỗ trợ 3 chế độ chạy (không ghi đè file tóm tắt của nhau):

```bash
# E1 (tự truy hồi nguyên bản & đoạn cắt 20s) + E3 (8 phương án trọng số) + E4 (hiệu năng từng công đoạn)
# -> xuất results/eval_summary_self.txt, eval_exact_retrieval.csv, eval_segment_retrieval.csv,
#    eval_weights.csv, eval_weights.png, eval_performance.csv
python evaluate.py --source csv --n 50

# E2 (đánh giá truy vấn mới ngoài CSDL, đặt file trong data/queries/)
# -> xuất results/eval_summary_external.txt, eval_external_queries.csv,
#    eval_external_top5.csv, eval_external_dist.png
python evaluate.py --source csv --query_dir data/queries

# Chạy gộp cả E1 và E2 -> xuất thêm results/eval_summary.txt
python evaluate.py --source csv --self --query_dir data/queries
```

Tùy chọn thêm `--genre` nếu muốn chạy thêm chỉ số phụ `leave-one-out genre-precision@5` sau E1.

## Kiểm thử

```bash
python -m pytest -q tests
```

- Các test đơn vị và tích hợp dùng âm thanh tổng hợp chạy trực tiếp không cần PostgreSQL.
- Hai test PostgreSQL (`test_postgres_schema_roundtrip`, `test_postgres_load_and_search_roundtrip`) chỉ chạy khi đặt biến môi trường `TEST_DATABASE_URL` (lưu ý: test sẽ reset bảng trong CSDL thử nghiệm đó).

## Ghi chú kỹ thuật

- Thứ tự 71 cột cố định theo `audio_features.FEATURE_NAMES` (`mfcc_1` là MFCC 0, gồm 13 hệ số). Không trộn `features.csv` cũ có thứ tự cột khác.
- Tiền xử lý (mono, 22 050 Hz, cắt tối đa 30 giây ở giữa, chuẩn hóa đỉnh về `[-1, 1]`) dùng chung một hàm cho cả CSDL và file truy vấn.
- Độ tương đồng hiển thị là `1 / (1 + D)`, đơn điệu giảm theo khoảng cách `D` để xếp hạng và trực quan hóa.
- Phép tự truy hồi (E1) dùng để kiểm tra tính đúng đắn của pipeline và độ ổn định đặc trưng, không phải độ chính xác tổng quát trên nhạc mới.
