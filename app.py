"""
app.py - Giao dien demo (yeu cau 5).

    streamlit run app.py
Doi nguon du lieu bang bien moi truong APP_SOURCE=csv (mac dinh db = PostgreSQL).
"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from audio_features import GROUP_DIMS, GROUP_TITLES, GROUPS
from config import AUDIO_DIR, TOP_K
from search import SearchEngine

st.set_page_config(page_title="Tim kiem doan nhac tuong dong", layout="wide")


@st.cache_resource(show_spinner="Dang nap CSDL...")
def get_engine(source):
    return SearchEngine.load(source)


source = os.getenv("APP_SOURCE", "db")
try:
    eng = get_engine(source)
except Exception as e:                                   # noqa: BLE001
    st.warning(f"Khong nap duoc nguon '{source}' ({type(e).__name__}); chuyen sang CSV.")
    source = "csv"
    eng = get_engine("csv")

st.title("Tim kiem doan nhac tuong dong theo noi dung")
st.caption(f"CSDL: {len(eng.meta)} doan nhac (nguon: {'PostgreSQL' if source == 'db' else 'CSV'}). "
           f"Vector 71 chieu: " + ", ".join(f"bo {g} {GROUP_DIMS[g]}" for g in GROUPS))

with st.sidebar:
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
    try:
        res, raw, zq = eng.search_file(tmp.name, int(k), tuple(w))
    except Exception as e:                               # noqa: BLE001
        st.error(f"Khong xu ly duoc file: {e}")
        st.stop()
    finally:
        os.unlink(tmp.name)

    st.subheader(f"{len(res)} doan giong nhat")
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
    st.subheader("Khoang cach theo tung bo")
    st.bar_chart(res.set_index("file_name")[[f"D_{g}" for g in GROUPS]])
    with st.expander("Gia tri trung gian cua file truy van (71 chieu)"):
        from audio_features import FEATURE_NAMES, GROUP_OF
        st.dataframe(pd.DataFrame({"dac_trung": FEATURE_NAMES, "bo": GROUP_OF, "goc": raw, "z": zq}))
else:
    st.info("Tai len mot file de bat dau.")
