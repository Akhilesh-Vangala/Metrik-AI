import numpy as np
import pytest

from src.numba_ops import modified_zscore_numba, rolling_mean_numba, flag_anomalies_numba, warmup
from src.anomaly import modified_zscore_vectorized


@pytest.fixture(scope="module", autouse=True)
def jit_warmup():
    warmup()


class TestNumbaZScore:
    def test_matches_numpy_implementation(self):
        rng = np.random.RandomState(42)
        data = rng.randn(50_000).astype(np.float64)

        numba_result = modified_zscore_numba(data)
        numpy_result = modified_zscore_vectorized(data)

        np.testing.assert_allclose(numba_result, numpy_result, atol=1e-4)

    def test_handles_constant_array(self):
        data = np.ones(1000, dtype=np.float64) * 5.0
        result = modified_zscore_numba(data)
        assert np.all(np.isfinite(result))

    def test_handles_single_element(self):
        data = np.array([3.14], dtype=np.float64)
        result = modified_zscore_numba(data)
        assert len(result) == 1


class TestNumbaRolling:
    def test_rolling_mean_basic(self):
        data = np.arange(100, dtype=np.float64)
        result = rolling_mean_numba(data, window=5)
        assert np.isfinite(result[-1])
        expected_last = np.mean(data[-6:-1])
        np.testing.assert_allclose(result[-1], expected_last, atol=1e-10)

    def test_rolling_mean_with_nans(self):
        data = np.array([1.0, np.nan, 3.0, 4.0, 5.0], dtype=np.float64)
        result = rolling_mean_numba(data, window=3)
        assert np.isfinite(result[-1])


class TestNumbaFlags:
    def test_threshold_flagging(self):
        scores = np.array([0.5, 1.0, 3.0, 4.0, -5.0, 2.0], dtype=np.float64)
        flags = flag_anomalies_numba(scores, 3.5)
        assert flags[3] == 1
        assert flags[4] == 1
        assert flags[0] == 0
        assert flags[5] == 0
