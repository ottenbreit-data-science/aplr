"""Compare standard training with results captured from the upstream release."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from aplr import APLRRegressor


_BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_standard.py"
_SPEC = importlib.util.spec_from_file_location("upstream_parity_benchmark", _BENCHMARK_PATH)
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK)


@pytest.mark.parametrize("aggregate", [False, True])
def test_realistic_duplicate_tweedie_matches_upstream(aggregate):
    """Protect full boosting histories and historical interaction-partner scoring."""
    fixture_path = Path(__file__).with_name("fixtures") / "standard_tweedie_duplicates.json"
    fixture = json.loads(fixture_path.read_text())
    recipe = fixture["data"]
    X, y, weights, folds = _BENCHMARK.synthetic_data(
        recipe["rows"], recipe["duplicates"], recipe["loss"],
    )
    model = APLRRegressor(
        **fixture["parameters"], aggregate_identical_rows=aggregate,
    ).fit(X, y, sample_weight=weights, cv_observations=folds)
    np.testing.assert_allclose(
        model.get_validation_error_steps(), fixture["validation_error_steps"],
        rtol=1e-10, atol=1e-12,
    )
    np.testing.assert_allclose(
        model.predict(X[fixture["prediction_indexes"]]), fixture["predictions"], rtol=1e-10, atol=1e-12,
    )
    np.testing.assert_allclose(
        model.get_term_coefficients(), fixture["coefficients"],
        rtol=1e-10, atol=1e-12,
    )
    assert model.get_optimal_m() == fixture["optimal_m"]
    assert model.get_intercept() == pytest.approx(fixture["intercept"], rel=1e-10, abs=1e-12)
    assert model.get_cv_error() == pytest.approx(fixture["cv_error"], rel=1e-10, abs=1e-12)
    assert [
        _BENCHMARK.term_structure(term) for term in model.APLRRegressor.terms
    ] == fixture["selected_terms"]
