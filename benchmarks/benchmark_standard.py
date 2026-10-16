"""Compare independently installed upstream and optimized APLR packages.

Only synthetic data is used. Each measured fit runs in a fresh process; imports
and data generation are outside the fit timer. No timing assertion runs in CI.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time


def synthetic_data(rows, duplicate_fraction, loss):
    import numpy as np

    rng = np.random.default_rng(0)
    unique_rows = max(2, round(rows * (1.0 - duplicate_fraction)))
    unique = rng.normal(size=(unique_rows, 8))
    X = unique[np.arange(rows) % unique_rows].copy()
    rng.shuffle(X)
    eta = -0.5 + 0.35 * X[:, 0] + 0.2 * np.maximum(X[:, 1], 0.0)
    eta += 0.12 * X[:, 2] * (X[:, 3] > 0.0)
    mean = np.exp(eta)
    if loss == "poisson":
        y = rng.poisson(mean).astype(float)
    elif loss == "tweedie":
        count = rng.poisson(2.0 * mean)
        y = rng.gamma(np.maximum(count, 1), 0.5) * (count > 0)
    elif loss in ("binomial", "classifier"):
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    else:
        y = eta + rng.normal(scale=0.15, size=rows)
    weight = rng.uniform(0.5, 2.0, size=rows)
    # Explicit folds fix memberships across implementations. The same feature
    # can occur in both partitions, but aggregation only operates inside train.
    fold = np.arange(rows) % 3
    cv = np.where(fold[:, None] == np.arange(3), -1, 1).astype(np.int32)
    return X, y, weight, cv


def term_structure(term):
    return {
        "base_term": term.base_term,
        "split_point": None if term.split_point != term.split_point else term.split_point,
        "direction_right": term.direction_right,
        "given_terms": [term_structure(given) for given in term.given_terms],
    }


def measure(spec, output):
    import numpy as np
    import aplr
    import aplr_cpp

    X, y, weight, cv = synthetic_data(spec["rows"], spec["duplicates"], spec["loss"])
    params = dict(m=100, bins=100, cv_folds=3, n_jobs=1, v=0.2,
                  random_state=0, preprocess=False, max_interaction_level=1,
                  penalty_for_interactions=0.5)
    if spec["suite"] == "parity":
        params.update(m=30, bins=12)
    if spec["loss"] == "preprocessed":
        import pandas as pd
        X = pd.DataFrame(X, columns=["x" + str(i) for i in range(X.shape[1])])
        X["category"] = np.where(X["x0"] > 0, "high", "low")
        X.loc[np.arange(0, len(X), 19), "x1"] = np.nan
        params["preprocess"] = True
    if spec["loss"] == "classifier":
        model = aplr.APLRClassifier(**params)
        y = np.where(y, "positive", "negative")
    else:
        loss = "mse" if spec["loss"] == "preprocessed" else spec["loss"]
        params.update(loss_function=loss, link_function="log" if loss in ("poisson", "tweedie") else "identity")
        if loss == "binomial":
            params["link_function"] = "logit"
        elif loss == "custom":
            params.update(
                loss_function="custom_function", link_function="custom_function",
                calculate_custom_loss_function=lambda y, p, w, g, d: float(np.average((y - p) ** 2, weights=w)),
                calculate_custom_negative_gradient_function=lambda y, p, g, d: y - p,
                calculate_custom_hessian_function=lambda y, p, g, d: np.ones_like(y),
                calculate_custom_transform_linear_predictor_to_predictions_function=lambda z: z,
                calculate_custom_differentiate_predictions_wrt_linear_predictor_function=lambda z: np.ones_like(z),
                calculate_custom_differentiate2_predictions_wrt_linear_predictor_function=lambda z: np.zeros_like(z),
            )
        if spec["variant"] != "upstream":
            params["aggregate_identical_rows"] = spec["variant"] == "aggregated"
        model = aplr.APLRRegressor(**params)
    started = time.perf_counter()
    cpu_started = time.process_time()
    model.fit(X, y, sample_weight=weight, cv_observations=cv)
    fit_cpu_seconds = time.process_time() - cpu_started
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    cpu_started = time.process_time()
    predictions = model.predict_proba(X) if spec["loss"] == "classifier" else model.predict(X)
    predict_cpu_seconds = time.process_time() - cpu_started
    predict_seconds = time.perf_counter() - started
    legacy_equal = None
    if spec.get("legacy_pickle"):
        import pickle
        legacy_path = Path(spec["legacy_pickle"])
        if spec["variant"] == "upstream":
            model.set_params(time_limit=123.5)
            legacy_path.write_bytes(pickle.dumps(model))
        else:
            restored = pickle.loads(legacy_path.read_bytes())
            assert restored.time_limit == 123.5
            if isinstance(restored, aplr.APLRRegressor):
                assert restored.aggregate_identical_rows is False
            restored_predictions = restored.predict_proba(X) if spec["loss"] == "classifier" else restored.predict(X)
            np.testing.assert_array_equal(restored_predictions, np.load(spec["legacy_arrays"])["predictions"])
            legacy_equal = True
    arrays = {"predictions": predictions, "validation_error_steps": model.get_validation_error_steps(),
              "cv_error": np.array(model.get_cv_error())}
    if spec["loss"] == "classifier":
        regressors = [model.get_logit_model(category) for category in model.get_categories()]
    else:
        regressors = [model]
    structure = []
    for index, regressor in enumerate(regressors):
        structure.append([term_structure(term) for term in regressor.APLRRegressor.terms])
        arrays["coefficients_" + str(index)] = regressor.get_term_coefficients()
        arrays["intercept_" + str(index)] = np.array(regressor.get_intercept())
        arrays["optimal_m_" + str(index)] = np.array(regressor.get_optimal_m())
        arrays["feature_importance_" + str(index)] = regressor.get_feature_importance()
        arrays["term_importance_" + str(index)] = regressor.get_term_importance()
        sample = X.iloc[:50] if hasattr(X, "iloc") else X[:50]
        arrays["local_feature_contribution_" + str(index)] = regressor.calculate_local_feature_contribution(sample)
        for fold in range(regressor.get_num_cv_folds()):
            arrays["cv_predictions_%s_%s" % (index, fold)] = regressor.get_cv_validation_predictions(fold)
            arrays["cv_indexes_%s_%s" % (index, fold)] = regressor.get_cv_validation_indexes(fold)
            arrays["cv_y_%s_%s" % (index, fold)] = regressor.get_cv_y(fold)
            arrays["cv_weights_%s_%s" % (index, fold)] = regressor.get_cv_sample_weight(fold)
    np.savez(output.with_suffix(".npz"), **arrays)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mib = rss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    except ImportError:
        peak_rss_mib = None
    result = dict(spec, fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                  fit_cpu_seconds=fit_cpu_seconds, predict_cpu_seconds=predict_cpu_seconds,
                  peak_rss_mib=peak_rss_mib, selected_terms=structure,
                  legacy_pickle_predictions_equal=legacy_equal,
                  parameters={key: value if not callable(value) else "custom callback"
                              for key, value in params.items()},
                  aggregation_diagnostics=model.get_aggregation_diagnostics() if spec["variant"] == "aggregated" else [])
    result.pop("legacy_pickle", None)
    result.pop("legacy_arrays", None)
    result["environment"] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "packages": {name: importlib.metadata.version(name) for name in ("aplr", "numpy", "pandas")},
        "extension_sha256": hashlib.sha256(Path(aplr_cpp.__file__).read_bytes()).hexdigest(),
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                result["environment"]["cpu_model"] = line.split(":", 1)[1].strip()
                break
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def compare(reference, candidate):
    import numpy as np

    left, right = json.loads(reference.read_text()), json.loads(candidate.read_text())
    def compare_terms(a, b):
        assert len(a) == len(b), "Selected term counts differ"
        for term_a, term_b in zip(a, b):
            assert term_a["base_term"] == term_b["base_term"], "Selected predictors differ"
            assert term_a["direction_right"] == term_b["direction_right"], "Selected hinge directions differ"
            if term_a["split_point"] is None or term_b["split_point"] is None:
                assert term_a["split_point"] is term_b["split_point"], "Selected term types differ"
            else:
                np.testing.assert_allclose(term_a["split_point"], term_b["split_point"], rtol=1e-10, atol=1e-12)
            compare_terms(term_a["given_terms"], term_b["given_terms"])
    assert len(left["selected_terms"]) == len(right["selected_terms"])
    for terms_a, terms_b in zip(left["selected_terms"], right["selected_terms"]):
        compare_terms(terms_a, terms_b)
    maxima = {}
    with np.load(reference.with_suffix(".npz")) as baseline, np.load(candidate.with_suffix(".npz")) as optimized:
        if set(baseline.files) != set(optimized.files):
            raise AssertionError("Result arrays differ")
        for key in baseline.files:
            a, b = baseline[key], optimized[key]
            if a.dtype.kind in "iub":
                np.testing.assert_array_equal(a, b, err_msg=key)
            else:
                np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True, err_msg=key)
            finite = np.isfinite(a) & np.isfinite(b)
            maxima[key] = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
    return maxima


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--suite", choices=("benchmark", "parity"), default="benchmark")
    parser.add_argument("--rows", default="10000,50000")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("standard-results.json"))
    parser.add_argument("--worker-spec")
    args = parser.parse_args()
    if args.worker_spec:
        measure(json.loads(args.worker_spec), args.output)
        return
    if not args.baseline_python or not args.candidate_python:
        parser.error("--baseline-python and --candidate-python are required")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    script = Path(__file__).resolve()
    if args.suite == "parity":
        cases = [(600, 0.0, loss) for loss in ("mse", "poisson", "tweedie", "binomial", "classifier", "custom", "preprocessed")]
        repeats = 1
    else:
        cases = [(int(rows), duplicates, loss) for rows in args.rows.split(",")
                 for duplicates in (0.0, 0.75) for loss in ("poisson", "tweedie")]
        repeats = args.repeats
    results = []
    def revision(directory):
        process = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                 capture_output=True, text=True)
        return process.stdout.strip() if process.returncode == 0 else None
    source_root = script.parent.parent
    compiled_sources = [source_root / path for path in (
        "setup.py", "aplr/aplr.py", "cpp/APLRRegressor.h", "cpp/pythonbinding.cpp",
        "cpp/term.h", "cpp/prefix_split_search.h", "cpp/row_aggregation.h")]
    source_hash = hashlib.sha256()
    for path in compiled_sources:
        source_hash.update(path.relative_to(source_root).as_posix().encode())
        source_hash.update(path.read_bytes())
    report = {
        "suite": args.suite,
        "baseline_revision": revision(args.baseline_python.absolute().parent),
        "candidate_revision": revision(source_root),
        "candidate_sources_sha256": source_hash.hexdigest(),
        "comparison_tolerance": {"rtol": 1e-10, "atol": 1e-12},
        "results": results,
    }
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="aplr-comparison-") as directory:
        for case_index, (rows, duplicates, loss) in enumerate(cases):
            reference = None
            variants = ("upstream", "optimized", "aggregated") if args.suite == "benchmark" else ("upstream", "optimized")
            for variant in variants:
                for repeat in range(repeats):
                    spec = dict(suite=args.suite, rows=rows, duplicates=duplicates, loss=loss,
                                variant=variant, repeat=repeat)
                    destination = Path(directory) / ("%s-%s-%s.json" % (case_index, variant, repeat))
                    if args.suite == "parity" and loss != "custom":
                        spec["legacy_pickle"] = str(Path(directory) / (str(case_index) + ".pickle"))
                        if reference:
                            spec["legacy_arrays"] = str(reference.with_suffix(".npz"))
                    executable = args.baseline_python if variant == "upstream" else args.candidate_python
                    subprocess.run([str(executable.absolute()), str(script), "--worker-spec", json.dumps(spec),
                                    "--output", str(destination)], cwd=directory, env=env, check=True)
                    result = json.loads(destination.read_text())
                    if reference is None:
                        reference = destination
                    try:
                        result["parity_max_absolute_differences"] = compare(reference, destination)
                    except AssertionError:
                        failure_directory = args.output.with_suffix(".failure")
                        failure_directory.mkdir(parents=True, exist_ok=True)
                        for path in (reference, destination, reference.with_suffix(".npz"), destination.with_suffix(".npz")):
                            shutil.copy2(path, failure_directory / path.name)
                        print("Comparison evidence retained in " + str(failure_directory), flush=True)
                        raise
                    result.pop("selected_terms")
                    results.append(result)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print("[%s of %s] %s %s rows=%s duplicates=%s fit=%.3fs parity=passed" % (
                        len(results), len(cases) * len(variants) * repeats, loss, variant, rows,
                        duplicates, result["fit_seconds"]), flush=True)


if __name__ == "__main__":
    main()
