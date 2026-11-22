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
    meta = pd.DataFrame({"audio_id": np.arange(1, len(X) + 1), "file_name": [f"{i}.wav" for i in range(len(X))],
                         "file_path": [f"{i}.wav" for i in range(len(X))], "genre": ["x"] * len(X)})
    return SearchEngine(meta, transform(X, mean, std), mean, std), X


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


def test_scaler_handles_constant_column():
    X = np.random.default_rng(0).normal(size=(20, 4))
    X[:, 2] = 5.0
    mean, std, const = fit_scaler(X)
    Z = transform(X, mean, std)
    assert const.tolist() == [False, False, True, False] and np.all(np.isfinite(Z))
    assert np.allclose(Z.mean(0), 0) and np.allclose(Z[:, [0, 1, 3]].std(0), 1)


def test_weights_normalized():
    assert abs(sum(normalize_weights((2, 1, 1)).values()) - 1) < 1e-12
    with pytest.raises(ValueError):
        normalize_weights((0, 0, 0))


def test_distance_properties(engine_and_vectors):
    eng, _ = engine_and_vectors
    Dg = group_distances(eng.Z[0], eng.Z)
    D = combine(Dg, (1, 1, 1))
    assert D[0] == pytest.approx(0)
    assert np.all(D >= 0)
    # D bang khoang cach Euclid sau khi nhan moi chieu voi sqrt(w_g / d_g)
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


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="can TEST_DATABASE_URL de thu PostgreSQL")
def test_postgres_schema_roundtrip():
    from db import connect, create_schema
    with connect(os.environ["TEST_DATABASE_URL"]) as conn:
        create_schema(conn, reset=True)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'audio_feature'")
            assert cur.fetchone()[0] == 1 + 71
