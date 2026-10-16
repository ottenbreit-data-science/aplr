"""Regression coverage for standard APLR's optional training-row aggregation."""

import inspect
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aplr import APLRRegressor


def _duplicate_data():
    # Exact duplicates at both extreme predictor values are intentional: split
    # eligibility must count observations, rather than compressed feature rows.
    unique = np.array(
        [[a, b, (a + 2 * b) % 3] for a in range(-2, 3) for b in range(5)],
        dtype=float,
    )
    X = np.repeat(unique, 4, axis=0)
    mean = np.exp(0.25 * unique[:, 0] + 0.1 * unique[:, 1]
                  + 0.12 * unique[:, 0] * unique[:, 1])
    y = np.repeat(mean, 4) * np.tile([0.5, 0.9, 1.1, 1.7], len(unique))
    weights = np.tile([1.0, 2.0, 0.0, 4.0], len(unique))
    # Also retain a predictor group with entirely zero weight.
    weights[8:12] = 0.0
    folds = np.ones((len(y), 2), dtype=int)
    folds[::4, 0] = -1
    folds[1::4, 1] = -1
    return X, y, weights, folds


def _model(loss="poisson", aggregate=True, **parameters):
    defaults = dict(
        m=20, v=0.2, random_state=3, loss_function=loss,
        link_function="log", n_jobs=1, bins=8,
        max_interaction_level=1, min_observations_in_split=0.15,
        early_stopping_rounds=20, preprocess=False,
        aggregate_identical_rows=aggregate,
    )
    defaults.update(parameters)
    return APLRRegressor(**defaults)


def _fit_pair(loss="poisson", parameters=None, fit_parameters=None, X=None):
    matrix, y, weights, folds = _duplicate_data()
    if X is None:
        X = matrix
    fit_args = dict(sample_weight=weights, cv_observations=folds)
    fit_args.update(fit_parameters or {})
    baseline = _model(loss, False, **(parameters or {})).fit(X, y, **fit_args)
    aggregated = _model(loss, True, **(parameters or {})).fit(X, y, **fit_args)
    return baseline, aggregated, X, y, weights, folds


def _assert_equivalent(baseline, aggregated, X):
    assert baseline.get_term_names() == aggregated.get_term_names()
    assert baseline.get_optimal_m() == aggregated.get_optimal_m()
    np.testing.assert_allclose(
        baseline.get_term_coefficients(), aggregated.get_term_coefficients(),
        rtol=1e-10, atol=1e-12,
    )
    assert baseline.get_intercept() == pytest.approx(
        aggregated.get_intercept(), rel=1e-10, abs=1e-12,
    )
    assert baseline.get_cv_error() == pytest.approx(
        aggregated.get_cv_error(), rel=1e-10, abs=1e-12,
    )
    np.testing.assert_allclose(
        baseline.get_validation_error_steps(), aggregated.get_validation_error_steps(),
        rtol=1e-10, atol=1e-12,
    )
    for capped in (False, True):
        np.testing.assert_allclose(
            baseline.predict(X, cap_predictions_to_minmax_in_training=capped),
            aggregated.predict(X, cap_predictions_to_minmax_in_training=capped),
            rtol=1e-10, atol=1e-12,
        )
    for fold in range(baseline.get_num_cv_folds()):
        np.testing.assert_array_equal(
            baseline.get_cv_validation_indexes(fold),
            aggregated.get_cv_validation_indexes(fold),
        )
        np.testing.assert_array_equal(baseline.get_cv_y(fold), aggregated.get_cv_y(fold))
        np.testing.assert_array_equal(
            baseline.get_cv_sample_weight(fold), aggregated.get_cv_sample_weight(fold),
        )
        np.testing.assert_allclose(
            baseline.get_cv_validation_predictions(fold),
            aggregated.get_cv_validation_predictions(fold), rtol=1e-10, atol=1e-12,
        )


@pytest.mark.parametrize("loss", ["poisson", "tweedie"])
@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("split_parameters,fit_parameters", [
    ({"bins": 3, "min_observations_in_split": 0.2}, {}),
    ({"bins": 8, "min_observations_in_split": 12}, {}),
    ({"bins": 100}, {"predictor_min_observations_in_split": [8, 10, 12]}),
])
def test_weighted_aggregation_preserves_folds_and_split_eligibility(
    loss, n_jobs, split_parameters, fit_parameters,
):
    parameters = dict(split_parameters, n_jobs=n_jobs)
    baseline, aggregated, X, y, weights, folds = _fit_pair(
        loss, parameters, fit_parameters,
    )
    _assert_equivalent(baseline, aggregated, X)
    assert aggregated.get_aggregation_diagnostics() == [
        {"fold": fold, "rows_before": 75, "rows_after": 25, "compression_ratio": 3.0}
        for fold in range(2)
    ]
    for fold in range(2):
        indexes = np.flatnonzero(folds[:, fold] == -1)
        np.testing.assert_array_equal(aggregated.get_cv_validation_indexes(fold), indexes)
        np.testing.assert_array_equal(aggregated.get_cv_y(fold), y[indexes])


@pytest.mark.parametrize("loss", ["poisson", "tweedie"])
def test_aggregation_with_preprocessing_and_interactions(loss):
    X, _, _, _ = _duplicate_data()
    frame = pd.DataFrame(X[:, :2], columns=["signal", "level"])
    frame["category"] = np.where(X[:, 2] == 0, "zero", "other")
    frame.loc[frame["signal"] == 0, "signal"] = np.nan
    parameters = dict(
        preprocess=True, max_interaction_level=2, max_interactions=12,
        penalty_for_interactions=0.0, penalty_for_non_linearity=0.03,
        ridge_penalty=0.2, m=35,
    )
    baseline, aggregated, _, _, _, _ = _fit_pair(loss, parameters, X=frame)
    _assert_equivalent(baseline, aggregated, frame)
    assert np.any(aggregated.get_term_interaction_levels() > 0)
    assert all(item["rows_after"] < item["rows_before"]
               for item in aggregated.get_aggregation_diagnostics())


def test_aggregation_preserves_training_response_extrema_for_clipping():
    # Zero-weight extrema still determine capping. After grouping, weighted
    # targets are >= 1, while the fitted log-linear model predicts below 1.
    X = np.repeat(np.arange(-2, 3, dtype=float).reshape(-1, 1), 4, axis=0)
    y = np.repeat([1.0, 1.0, 1.0, 1.0, 3.0], 4)
    weights = np.tile([1.0, 1.0, 0.0, 1.0], 5)
    y[2], y[-2] = 0.01, 100.0
    folds = np.ones((len(y), 1), dtype=int)
    folds[::4] = -1
    arguments = dict(sample_weight=weights, cv_observations=folds)
    parameters = dict(m=30, num_first_steps_with_linear_effects_only=100,
                      min_observations_in_split=2)
    baseline = _model(aggregate=False, **parameters).fit(X, y, **arguments)
    aggregated = _model(**parameters).fit(X, y, **arguments)
    query = np.array([[-100.0], [100.0]])
    _assert_equivalent(baseline, aggregated, query)
    raw = aggregated.predict(query, cap_predictions_to_minmax_in_training=False)
    capped = aggregated.predict(query)
    training_y = y[folds[:, 0] == 1]
    training_predictions = baseline.predict(X[folds[:, 0] == 1], False)
    lower = max(training_predictions.min(), training_y.min())
    upper = min(training_predictions.max(), training_y.max())
    assert lower < 1.0  # The compressed targets alone would incorrectly cap at 1.
    np.testing.assert_allclose(capped, np.clip(raw, lower, upper))
    assert np.any(raw != capped)


def test_aggregation_refit_refreshes_caches_and_diagnostics():
    _, aggregated, X, y, weights, folds = _fit_pair()
    # Change ordering, values, row count, and duplicate ratio in the same object.
    indexes = np.arange(0, len(y), 2)[::-1]
    changed_X = X[indexes].copy()
    changed_X[:, 1] += np.linspace(0.0, 0.5, len(indexes))
    changed_y = y[indexes] * 1.3
    changed_weights = weights[indexes] + 0.1
    changed_folds = np.ones((len(indexes), 1), dtype=int)
    changed_folds[::5] = -1
    arguments = dict(sample_weight=changed_weights, cv_observations=changed_folds)
    aggregated.fit(changed_X, changed_y, **arguments)
    fresh = _model().fit(changed_X, changed_y, **arguments)
    _assert_equivalent(fresh, aggregated, changed_X)
    assert aggregated.get_aggregation_diagnostics() == [
        {"fold": 0, "rows_before": 40, "rows_after": 40, "compression_ratio": 1.0}
    ]
    aggregated.set_params(aggregate_identical_rows=False).fit(changed_X, changed_y, **arguments)
    assert aggregated.get_aggregation_diagnostics()[0]["compression_ratio"] == 1.0


@pytest.mark.parametrize("parameters,fit_parameters,reason", [
    ({"loss_function": "mse", "link_function": "identity"}, {}, "poisson|tweedie"),
    ({"link_function": "identity"}, {}, "log"),
    ({"loss_function": "tweedie", "dispersion_parameter": 1.0}, {}, "dispersion|Tweedie|tweedie"),
    ({"loss_function": "tweedie", "dispersion_parameter": 2.0}, {}, "dispersion|Tweedie|tweedie"),
    ({"validation_tuning_metric": "mse"}, {}, "validation_tuning_metric"),
    ({}, {"group": np.arange(100)}, "group"),
    ({}, {"other_data": np.ones((100, 1))}, "other_data"),
])
def test_aggregation_rejects_unsupported_configurations(parameters, fit_parameters, reason):
    X, y, weights, folds = _duplicate_data()
    with pytest.raises(RuntimeError, match=reason):
        _model(**parameters).fit(
            X, y, sample_weight=weights, cv_observations=folds, **fit_parameters,
        )


@pytest.mark.parametrize("callback", [
    "calculate_custom_validation_error_function",
    "calculate_custom_loss_function",
    "calculate_custom_negative_gradient_function",
    "calculate_custom_hessian_function",
    "calculate_custom_transform_linear_predictor_to_predictions_function",
    "calculate_custom_differentiate_predictions_wrt_linear_predictor_function",
    "calculate_custom_differentiate2_predictions_wrt_linear_predictor_function",
])
def test_aggregation_rejects_custom_numerical_callbacks_before_calling_them(callback):
    X, y, weights, folds = _duplicate_data()

    def forbidden(*arguments):
        pytest.fail("unsupported callback must not run before aggregation rejects it")

    with pytest.raises(RuntimeError, match="custom|callback"):
        _model(**{callback: forbidden}).fit(
            X, y, sample_weight=weights, cv_observations=folds,
        )


def test_aggregation_is_appended_and_supported_by_sklearn_clone():
    clone = pytest.importorskip("sklearn.base").clone
    signature = inspect.signature(APLRRegressor)
    assert list(signature.parameters)[-2:] == ["time_limit", "aggregate_identical_rows"]
    model = _model(time_limit=1e6)
    copied = clone(model)
    assert copied.get_params() == model.get_params()
    assert copied.APLRRegressor.aggregate_identical_rows
    assert copied.get_aggregation_diagnostics() == []
    assert copied.set_params(aggregate_identical_rows=False) is copied
    assert not copied.APLRRegressor.aggregate_identical_rows
    # Preserve every old positional constructor parameter, including time_limit.
    old_arguments = [parameter.default for name, parameter in signature.parameters.items()
                     if name != "aggregate_identical_rows"]
    old_arguments[-1] = 2.5
    positional = APLRRegressor(*old_arguments)
    assert positional.time_limit == 2.5
    assert positional.aggregate_identical_rows is False


def test_pickle_round_trip_and_old_native_and_python_state():
    _, model, X, _, _, _ = _fit_pair(parameters={"time_limit": 1e6})
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    assert restored.aggregate_identical_rows
    assert restored.time_limit == 1e6
    assert restored.get_aggregation_diagnostics() == model.get_aggregation_diagnostics()
    native_state = model.APLRRegressor.__getstate__()
    assert len(native_state) == 62
    assert native_state[58] == 1e6
    assert native_state[59] is True
    assert native_state[60] == [75, 75]
    assert native_state[61] == [25, 25]
    for length in (58, 59):
        native = type(model.APLRRegressor).__new__(type(model.APLRRegressor))
        native.__setstate__(native_state[:length])
        assert native.aggregate_identical_rows is False
        assert native.cv_training_rows_before_aggregation == []
        assert native.cv_training_rows_after_aggregation == []
        if length == 59:
            assert native.time_limit == 1e6
        else:
            assert np.isnan(native.time_limit)
        np.testing.assert_array_equal(native.predict(X, True), model.predict(X))
    old_state = model.__dict__.copy()
    old_state.pop("aggregate_identical_rows")
    old_state["APLRRegressor"] = native
    old_model = APLRRegressor.__new__(APLRRegressor)
    old_model.__setstate__(old_state)
    assert old_model.aggregate_identical_rows is False
    assert old_model.get_aggregation_diagnostics() == []
    np.testing.assert_array_equal(old_model.predict(X), model.predict(X))


def test_late_tweedie_validation_history_matches_upstream():
    # Final predictions alone miss a split-scoring regression after step 60.
    # Keep the full curve produced by genuine upstream, including later steps
    # that were not selected for the final model.
    directory = Path(__file__).resolve().parent
    fixture = json.loads(
        (directory / "fixtures" / "standard_tweedie_validation_history.json").read_text()
    )
    specification = importlib.util.spec_from_file_location(
        "aplr_benchmark_standard", directory.parent / "benchmarks" / "benchmark_standard.py"
    )
    benchmark = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(benchmark)
    X, y, weight, cv = benchmark.synthetic_data(10000, 0.0, "tweedie")
    model = APLRRegressor(**fixture["parameters"])
    model.fit(X, y, sample_weight=weight, cv_observations=cv)
    history = model.get_validation_error_steps()
    assert history.shape == (100, 3)
    np.testing.assert_allclose(
        history, fixture["validation_error_steps"], rtol=1e-10, atol=1e-12,
    )


@pytest.mark.parametrize("loss", ["poisson", "tweedie"])
@pytest.mark.parametrize("updates", [
    pytest.param({"mean_bias_correction": True}, id="mean-bias-correction"),
    pytest.param({"max_terms": 3}, id="round-robin-coefficients"),
])
def test_aggregation_preserves_optional_training_updates(loss, updates):
    parameters = dict(updates, m=35)
    baseline, aggregated, X, _, _, _ = _fit_pair(loss, parameters)
    _assert_equivalent(baseline, aggregated, X)
