"""
app.py - Giao dien demo (yeu cau 5).

    streamlit run app.py
Doi nguon du lieu bang bien moi truong APP_SOURCE=csv (mac dinh db = PostgreSQL).
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from audio_features import FEATURE_NAMES, GROUP_DIMS, GROUP_OF, GROUP_TITLES, GROUPS
from config import AUDIO_DIR, SR, TOP_K
from search import SearchEngine

st.set_page_config(page_title="Tim kiem doan nhac tuong dong", layout="wide")


@st.cache_resource(show_spinner="Dang nap CSDL...")
def get_engine(source):
    return SearchEngine.load(source)


@st.fragment
def show_feature_chart(features):
    group_col, value_col = st.columns(2)
    group = group_col.selectbox("Nhóm đặc trưng", GROUPS)
    value = value_col.radio("Giá trị trên biểu đồ", ["Z-score", "Giá trị gốc"], horizontal=True)
    st.bar_chart(features.loc[features["Nhóm"] == group].set_index("Đặc trưng")[value],
                 x_label="Đặc trưng", y_label=value, horizontal=True, sort=False)


source = os.getenv("APP_SOURCE", "db")
try:
    eng = get_engine(source)
except Exception as e:                                   # noqa: BLE001
    if source != "csv":
        st.warning(f"Khong nap duoc nguon '{source}' ({type(e).__name__}: {e}); chuyen sang CSV.")
        source = "csv"
        try:
            eng = get_engine("csv")
        except Exception as e2:                          # noqa: BLE001
            st.error(f"Khong nap duoc ca PostgreSQL lan CSV: {e2}")
            st.stop()
    else:
        st.error(f"Khong nap duoc nguon CSV: {e}")
        st.stop()

st.title("Tim kiem doan nhac tuong dong theo noi dung")
st.caption(f"CSDL: {len(eng.meta)} doan nhac (nguon: {'PostgreSQL' if source == 'db' else 'CSV'}). "
           f"Vector 71 chieu: " + ", ".join(f"bo {g} {GROUP_DIMS[g]}" for g in GROUPS))

with st.sidebar:
    if st.button("Tai lai CSDL", use_container_width=True):
        get_engine.clear()
        st.rerun()
    st.header("Trong so cac bo")
    w = [st.slider(f"Bo {g}: {GROUP_TITLES[g]}", 0.0, 1.0, 1 / 3, 0.05) for g in GROUPS]
    if sum(w) == 0:
        st.error("Can it nhat mot trong so > 0")
        st.stop()
    k = st.number_input("So ket qua", 1, 20, TOP_K)

up = st.file_uploader("Tai len file am thanh truy van (wav, mp3, ogg, flac)",
                      type=["wav", "mp3", "ogg", "flac"])
if up is not None:
    st.audio(up)
    with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tmp:
        tmp.write(up.getbuffer())
        tmp_path = tmp.name
    try:
        result = eng.search_file_details(tmp_path, int(k), tuple(w))
    except Exception as e:                               # noqa: BLE001
        st.error(f"Khong xu ly duoc file: {e}")
        st.stop()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    extraction, search = st.columns(2)
    extraction.metric("Tiền xử lý và trích xuất", f"{result.extraction_seconds:.3f} s")
    search.metric("Chuẩn hóa và tìm kiếm", f"{result.search_seconds:.3f} s")

    res = result.matches
    st.subheader(f"{len(res)} đoạn nhạc giống nhất")
    for _, r in res.iterrows():
        c1, c2 = st.columns([2, 3])
        c1.markdown(f"**#{int(r['rank'])}  {r['file_name']}**  \n"
                    f"khoang cach {r['distance']:.3f} | tuong dong {r['similarity']:.3f}  \n"
                    f"D_A {r['D_A']:.2f} · D_B {r['D_B']:.2f} · D_C {r['D_C']:.2f}")
        p = AUDIO_DIR / r["file_path"]
        if p.exists():
            c2.audio(str(p))
        else:
            c2.caption(f"Khong tim thay {p}")
    st.subheader("Khoảng cách theo nhóm A, B, C")
    if not res.empty:
        st.bar_chart(res.set_index("rank")[[f"D_{g}" for g in GROUPS]],
                     x_label="Hạng kết quả", y_label="Khoảng cách nhóm", stack=False)

    st.subheader("Dạng sóng sau tiền xử lý")
    y = result.waveform
    # Preserve peaks while limiting the number of points sent to the browser.
    window = (y.size + 1999) // 2000
    starts = np.arange(0, y.size, window)
    waveform = pd.DataFrame({
        "Biên độ thấp nhất": np.minimum.reduceat(y, starts),
        "Biên độ cao nhất": np.maximum.reduceat(y, starts),
    }, index=starts / SR)
    st.line_chart(waveform, x_label="Thời gian trong đoạn phân tích (s)", y_label="Biên độ")

    st.subheader("71 đặc trưng của đoạn truy vấn")
    features = pd.DataFrame({
        "Đặc trưng": FEATURE_NAMES,
        "Nhóm": GROUP_OF,
        "Giá trị gốc": result.raw,
        "Z-score": result.standardized,
    })
    st.dataframe(features, hide_index=True)
    show_feature_chart(features)
else:
    st.info("Tai len mot file de bat dau.")
