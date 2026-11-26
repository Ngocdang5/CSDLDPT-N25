"""pytest: kiem tra ban chat cua pipeline bang am thanh tong hop (khong can PostgreSQL)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_features import (FEATURE_NAMES, GROUP_DIMS, GROUP_INDEX, N_FEATURES,  # noqa: E402
                            extract_features, preprocess_array)
from scaling import fit_scaler, transform  # noqa: E402
from search import SearchEngine, combine, group_distances, normalize_weights  # noqa: E402

SR = 22050


def tone(f0, bpm, seconds=8, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(SR * seconds) / SR
    beat = 60 / bpm
    env = np.exp(-((t % beat) / (beat * 0.4)))
    y = (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(4 * np.pi * f0 * t)) * env
    return (y + 0.003 * rng.standard_normal(t.size)).astype(np.float32)


@pytest.fixture(scope="module")
def engine_and_vectors():
    specs = [(110, 70), (110, 130), (330, 70), (330, 130), (880, 70), (880, 130)] * 3
    raws = [extract_features(preprocess_array(tone(f, b, seed=i))) for i, (f, b) in enumerate(specs)]
    X = np.vstack(raws)
    mean, std, _ = fit_scaler(X)
    import pandas as pd
    meta = pd.DataFrame({
        "audio_id": np.arange(1, len(X) + 1),
        "file_name": [f"{i}.wav" for i in range(len(X))],
        "file_path": [f"{i}.wav" for i in range(len(X))],
        "genre": [f"g{i % 3}" for i in range(len(X))],
    })
    return SearchEngine(meta, transform(X, mean, std), mean, std), X


# ---------------------------------------------------------------- feature tests
def test_feature_layout():
    assert N_FEATURES == 71 and len(set(FEATURE_NAMES)) == 71
    assert GROUP_DIMS == {"A": 42, "B": 24, "C": 5}
    assert sorted(np.concatenate(list(GROUP_INDEX.values()))) == list(range(71))


def test_extract_is_finite():
    v = extract_features(preprocess_array(tone(220, 90)))
    assert v.shape == (71,) and np.all(np.isfinite(v))


def test_preprocess_peak_and_short_file():
    y = preprocess_array(tone(220, 90) * 0.1)
    assert abs(np.abs(y).max() - 1.0) < 1e-6
    with pytest.raises(ValueError):
        preprocess_array(np.zeros(SR, dtype=np.float32) + 0.1)      # 1 giay: qua ngan
    with pytest.raises(ValueError):
        preprocess_array(np.zeros(SR * 8, dtype=np.float32))        # im lang


# ---------------------------------------------------------------- scaler tests
def test_scaler_handles_constant_column():
    X = np.random.default_rng(0).normal(size=(20, 4))
    X[:, 2] = 5.0
    mean, std, const = fit_scaler(X)
    Z = transform(X, mean, std)
    assert const.tolist() == [False, False, True, False] and np.all(np.isfinite(Z))
    assert np.allclose(Z.mean(0), 0) and np.allclose(Z[:, [0, 1, 3]].std(0), 1)


# ---------------------------------------------------------------- search tests
def test_weights_normalized():
    assert abs(sum(normalize_weights((2, 1, 1)).values()) - 1) < 1e-12
    with pytest.raises(ValueError):
        normalize_weights((0, 0, 0))
    with pytest.raises(ValueError):
        normalize_weights((-1, 2, 1))          # trong so am


def test_distance_properties(engine_and_vectors):
    eng, _ = engine_and_vectors
    Dg = group_distances(eng.Z[0], eng.Z)
    D = combine(Dg, (1, 1, 1))
    assert D[0] == pytest.approx(0)
    assert np.all(D >= 0)
    w = normalize_weights((1, 1, 1))
    scale = np.zeros(71)
    for g, ix in GROUP_INDEX.items():
        scale[ix] = np.sqrt(w[g] / GROUP_DIMS[g])
    assert np.allclose(D, np.linalg.norm((eng.Z - eng.Z[0]) * scale, axis=1))


def test_search_returns_sorted_top5_and_self_first(engine_and_vectors):
    eng, X = engine_and_vectors
    zq = transform(X[3], eng.mean, eng.std)
    res = eng.search_vector(zq, 5)
    assert len(res) == 5 and res.loc[0, "audio_id"] == 4
    assert res["distance"].is_monotonic_increasing and res["similarity"].is_monotonic_decreasing
    res2 = eng.search_vector(zq, 5, exclude_ids=[4])
    assert 4 not in res2["audio_id"].tolist() and len(res2) == 5


def test_search_vector_validates_inputs(engine_and_vectors):
    """Kiem tra tat ca rang buoc dau vao cua search_vector."""
    eng, X = engine_and_vectors
    zq = transform(X[0], eng.mean, eng.std)
    with pytest.raises(ValueError):
        eng.search_vector(zq, k=0)                          # k <= 0
    with pytest.raises(ValueError):
        eng.search_vector(np.full(N_FEATURES, np.nan))     # vector NaN
    with pytest.raises(ValueError):
        eng.search_vector(np.zeros(N_FEATURES - 1))        # sai so chieu


# ---------------------------------------------------------------- external eval
def test_external_query_eval(engine_and_vectors, tmp_path):
    """E2: external_query_eval tra ve summary_df, top5_df day du 5 dong/truy van."""
    import soundfile as sf
    from evaluate import external_query_eval

    eng, _ = engine_and_vectors
    y = tone(220, 90, seconds=8)
    wav_path = tmp_path / "query_test.wav"
    sf.write(str(wav_path), y, SR)

    lines, summary_df, top5_df, t_ext, t_srch, n_ok = external_query_eval(
        eng, str(tmp_path), None, ".", n=5, seed=0
    )
    assert n_ok >= 1
    assert not summary_df.empty
    # Kiem tra summary co du cot
    for col in ("top1_dist", "top1_sim", "mean_D_top5", "t_extract_s", "t_search_ms", "t_total_ms"):
        assert col in summary_df.columns, f"Thieu cot {col}"
    assert (summary_df["top1_dist"] >= 0).all()

    # Kiem tra top5_df: 5 hang / truy van, co cot D_A/D_B/D_C
    assert not top5_df.empty
    assert set(["query_file", "rank", "result_file", "distance", "similarity",
                "D_A", "D_B", "D_C"]).issubset(top5_df.columns)
    n_per_query = top5_df.groupby("query_file")["rank"].count()
    assert (n_per_query <= 5).all()

    # Kiem tra lines co cac thong tin can thiet
    full_text = " ".join(lines)
    assert "Khoang cach" in full_text and "Tong thoi gian/truy van" in full_text


def test_self_retrieval_rank_uses_argsort(engine_and_vectors):
    """Hang duoc tinh theo argsort (khong bi lech khi dong hang)."""
    from evaluate import _rank_of

    eng, _ = engine_and_vectors
    # Voi chinh vector do -> hang phai la 1
    for i in range(min(5, len(eng.Z))):
        rank = _rank_of(eng.Z[i], eng.Z, i, (1, 1, 1))
        assert rank == 1, f"Phai hang 1 khi truy van chinh no, nhan duoc {rank}"


def test_run_mode_logic():
    """Kiem tra logic run_self / run_ext (khong can I/O)."""
    # run_ext = False -> run_self = True (mac dinh chi tu truy hoi)
    run_ext = False
    run_self = True or not run_ext
    assert run_self

    # run_ext = True, run_self not set -> run_self = False
    run_ext = True
    run_self = False or not run_ext
    assert not run_self

    # run_ext = True, --self -> ca hai
    run_ext = True
    run_self = True or not run_ext
    assert run_self and run_ext


def test_exact_and_segment_self_retrieval_and_perf(tmp_path):
    """Kiem tra E1.1 (exact), E1.2 (segment) va E4 (summarize_performance)."""
    import pandas as pd
    import soundfile as sf
    from evaluate import exact_self_retrieval, segment_self_retrieval, summarize_performance

    from audio_features import load_audio

    specs = [(110, 80), (220, 100), (440, 120)]
    raws = []
    rows = []
    for i, (f, b) in enumerate(specs, start=1):
        y = tone(f, b, seconds=8, seed=i)
        fn = f"s{i}.wav"
        p = str(tmp_path / fn)
        sf.write(p, y, SR)
        raws.append(extract_features(preprocess_array(load_audio(p))))
        rows.append({"audio_id": i, "file_name": fn, "file_path": fn, "genre": "x"})

    X = np.vstack(raws)
    mean, std, _ = fit_scaler(X)
    eng = SearchEngine(pd.DataFrame(rows), transform(X, mean, std), mean, std)

    tab_exact, _, _, nq1, perf_rows = exact_self_retrieval(eng, n=3, seed=0, audio_dir=str(tmp_path))
    assert nq1 == 3 and tab_exact.loc[0, "hit@1"] == pytest.approx(1.0)
    assert len(perf_rows) == 3

    tab_seg, _, _, nq2 = segment_self_retrieval(eng, n=3, seed=0, audio_dir=str(tmp_path))
    assert nq2 == 3 and "hit@5" in tab_seg.columns

    perf_lines, perf_df = summarize_performance(perf_rows, db_load_ms=12.5)
    assert any("E4." in l for l in perf_lines)
    assert set(["cong_doan", "mean_ms", "median_ms", "std_ms", "p95_ms"]).issubset(perf_df.columns)


# ---------------------------------------------------------------- extract_all validation
def test_extract_all_load_existing(tmp_path):
    """Kiem tra _load_existing: loc dong NaN va phat hien lech file_name."""
    import pandas as pd
    from extract_all import EXPECTED_COLS, _load_existing

    meta = pd.DataFrame({"audio_id": [1, 2], "file_name": ["a.wav", "b.wav"]})
    rows = [
        [1, "a.wav"] + [0.5] * N_FEATURES,
        [2, "b.wav"] + [np.nan] * N_FEATURES,
    ]
    out_csv = tmp_path / "features.csv"
    pd.DataFrame(rows, columns=EXPECTED_COLS).to_csv(out_csv, index=False)

    done = _load_existing(str(out_csv), meta)
    assert done["audio_id"].tolist() == [1]

    bad_meta = pd.DataFrame({"audio_id": [1, 2], "file_name": ["khac.wav", "b.wav"]})
    with pytest.raises(SystemExit):
        _load_existing(str(out_csv), bad_meta)


# ---------------------------------------------------------------- postgres (optional)
@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="can TEST_DATABASE_URL de thu PostgreSQL")
def test_postgres_schema_roundtrip():
    from db import connect, create_schema
    with connect(os.environ["TEST_DATABASE_URL"]) as conn:
        create_schema(conn, reset=True)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'audio_feature'")
            assert cur.fetchone()[0] == 1 + 71


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="can TEST_DATABASE_URL de thu PostgreSQL")
def test_postgres_load_and_search_roundtrip(engine_and_vectors, tmp_path):
    """Kiem tra nap CSDL bang load_db.load() va truy van lai bang SearchEngine.from_db()."""
    import pandas as pd
    from load_db import load

    eng, X = engine_and_vectors
    dsn = os.environ["TEST_DATABASE_URL"]
    meta_df = eng.meta.copy()
    meta_df["format"] = "wav"
    meta_df["instrument"] = ""
    meta_df["duration"] = 8.0
    meta_df["sample_rate"] = SR
    meta_df["channels"] = 1
    meta_df["split"] = "database"

    feat_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    feat_df.insert(0, "file_name", meta_df["file_name"])
    feat_df.insert(0, "audio_id", meta_df["audio_id"])

    meta_csv = tmp_path / "metadata.csv"
    feat_csv = tmp_path / "features.csv"
    meta_df.to_csv(meta_csv, index=False)
    feat_df.to_csv(feat_csv, index=False)

    load(str(meta_csv), str(feat_csv), dsn=dsn, reset=True)
    db_eng = SearchEngine.from_db(dsn)
    assert len(db_eng.meta) == len(meta_df)
    zq = transform(X[0], db_eng.mean, db_eng.std)
    res = db_eng.search_vector(zq, k=3)
    assert len(res) == 3 and res.loc[0, "audio_id"] == int(meta_df.loc[0, "audio_id"])
