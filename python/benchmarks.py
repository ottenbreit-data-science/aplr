# This code is based on https://github.com/interpretml/interpret/blob/develop/docs/benchmarks/ebm-benchmark.ipynb

import joblib

try:
    completed_so_far = joblib.load("completed_so_far.zip")
except:
    completed_so_far = set()


def trial_filter(task):
    min_samples = 1
    max_samples = 1000000000000

    if task.scalar_measure("n_rows") < min_samples:
        return []

    if max_samples < task.scalar_measure("n_rows"):
        return []

    if task.origin == "pmlb":
        if task.problem == "binary":
            pass
        elif task.problem == "multiclass":
            pass
        elif task.problem == "regression":
            pass
        else:
            return []

        exclude_set = set()
        if task.name in exclude_set:
            return []
    else:
        return []

    exclude_set = completed_so_far.copy()
    if task.name in exclude_set:
        return []

    return [
        # "xgboost-base",
        # "ebm-base",
        "aplr-01",
        "aplr-03",
        "aplr-base",
    ]


def trial_runner(trial):
    seed = 42
    extra_params = {}
    # extra_params = {"interactions":0, "max_rounds":5}

    from xgboost import XGBClassifier, XGBRegressor
    from interpret.glassbox import (
        ExplainableBoostingClassifier,
        ExplainableBoostingRegressor,
    )
    from aplr import APLRClassifier, APLRRegressor
    from sklearn.metrics import roc_auc_score, r2_score, log_loss
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from time import time
    import warnings

    X, y, meta = trial.task.data(["X", "y", "meta"])

    # TODO: move this into powerlift
    import pandas as pd

    for col_name in X.columns:
        col = X[col_name]
        if col.dtype.name == "object":
            X[col_name] = col.astype(pd.CategoricalDtype(ordered=False))
        elif col.dtype.name == "category" and col.cat.ordered:
            X[col_name] = col.cat.as_unordered()
    import numpy as np

    _, y = np.unique(y, return_inverse=True)

    stratification = None
    if trial.task.problem in ["binary", "multiclass"]:
        # use stratefied, otherwise eval can fail if one of the classes is not in the training set
        stratification = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=stratification, random_state=seed
    )

    # Build preprocessor
    is_cat = meta["categorical_mask"]
    cat_cols = [idx for idx in range(X.shape[1]) if is_cat[idx]]
    num_cols = [idx for idx in range(X.shape[1]) if not is_cat[idx]]
    cat_ohe_step = ("ohe", OneHotEncoder(sparse_output=True, handle_unknown="ignore"))
    cat_pipe = Pipeline([cat_ohe_step])
    num_pipe = Pipeline([("identity", FunctionTransformer())])
    transformers = [("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)]
    ct = Pipeline(
        [
            ("ct", ColumnTransformer(transformers=transformers, sparse_threshold=0)),
            (
                "missing",
                SimpleImputer(add_indicator=True, strategy="most_frequent"),
            ),
        ]
    )

    # Specify method
    if trial.task.problem in ["binary", "multiclass"]:
        if trial.method.name == "xgboost-base":
            est = XGBClassifier(enable_categorical=True)
        elif trial.method.name == "ebm-base":
            est = ExplainableBoostingClassifier(**extra_params)
        elif trial.method.name == "aplr-base":
            est = Pipeline(
                [
                    ("ct", ct),
                    (
                        "est",
                        APLRClassifier(),
                    ),
                ]
            )
            y_train = y_train.astype(str).ravel()
            y_test = y_test.astype(str).ravel()
        elif trial.method.name == "aplr-01":
            est = Pipeline(
                [
                    ("ct", ct),
                    (
                        "est",
                        APLRClassifier(v=0.1),
                    ),
                ]
            )
            y_train = y_train.astype(str).ravel()
            y_test = y_test.astype(str).ravel()
        elif trial.method.name == "aplr-03":
            est = Pipeline(
                [
                    ("ct", ct),
                    (
                        "est",
                        APLRClassifier(v=0.3),
                    ),
                ]
            )
            y_train = y_train.astype(str).ravel()
            y_test = y_test.astype(str).ravel()
        else:
            raise RuntimeError(f"Method unavailable for {trial.method.name}")

        predict_fn = est.predict_proba
    elif trial.task.problem == "regression":
        if trial.method.name == "xgboost-base":
            est = XGBRegressor(enable_categorical=True)
        elif trial.method.name == "ebm-base":
            est = ExplainableBoostingRegressor(**extra_params)
        elif trial.method.name == "aplr-base":
            est = Pipeline(
                [
                    ("ct", ct),
                    (
                        "est",
                        APLRRegressor(),
                    ),
                ]
            )
        elif trial.method.name == "aplr-01":
            est = Pipeline(
                [
                    ("ct", ct),
                    (
                        "est",
                        APLRRegressor(v=0.1),
                    ),
                ]
            )
        elif trial.method.name == "aplr-03":
            est = Pipeline(
                [
                    ("ct", ct),
                    (
                        "est",
                        APLRRegressor(v=0.3),
                    ),
                ]
            )
        else:
            raise RuntimeError(f"Method unavailable for {trial.method.name}")

        predict_fn = est.predict
    else:
        raise Exception(f"Unrecognized task problem {trial.task.problem}")

    global global_counter
    try:
        global_counter += 1
    except NameError:
        global_counter = 0

    # Train
    print(
        f"FIT: {global_counter}, {trial.task.origin}, {trial.task.name}, {trial.method.name}, ",
        end="",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        start_time = time()
        est.fit(X_train, y_train)
        elapsed_time = time() - start_time
    trial.log("fit_time", elapsed_time)

    # Predict
    start_time = time()
    predictions = predict_fn(X_test)
    elapsed_time = time() - start_time
    trial.log("predict_time", elapsed_time)

    if trial.task.problem == "binary":
        predictions = predictions[:, 1]

        eval_score = roc_auc_score(y_test, predictions)
        trial.log("auc", eval_score)

        eval_score2 = log_loss(y_test, predictions)
        trial.log("log_loss", eval_score2)
    elif trial.task.problem == "multiclass":
        eval_score = roc_auc_score(
            y_test, predictions, average="weighted", multi_class="ovo"
        )
        trial.log("multi_auc", eval_score)

        eval_score2 = log_loss(y_test, predictions)
        trial.log("cross_entropy", eval_score2)
    elif trial.task.problem == "regression":
        eval_score = r2_score(y_test, predictions)
        trial.log("rsqr", eval_score)
    else:
        raise Exception(f"Unrecognized task problem {trial.task.problem}")

    if trial.method.name == "aplr-base":
        trial.log("rows", X_train.shape[0])
        trial.log("columns", X_train.shape[1])
        trial.log("columns_transformed", ct.transform(X_train).shape[1])
        completed_so_far.add(trial._task.name)
        joblib.dump(completed_so_far, "completed_so_far.zip", 9)
        try:
            benchmark_results = joblib.load("benchmark_results_so_far.zip")
            benchmark_results = pd.concat(
                [benchmark_results, benchmark.results()], ignore_index=True
            ).drop_duplicates(subset=["method", "task", "name"])
        except:
            benchmark_results = benchmark.results()
        joblib.dump(benchmark_results, "benchmark_results_so_far.zip", 9)
        try:
            benchmark_results.to_excel("benchmark_results_so_far.xlsx")
        except:
            pass

    print(eval_score)


force_recreate = False
exist_ok = True

import uuid

experiment_name = "myexperiment" + "__" + str(uuid.uuid4())
print("Experiment name: " + str(experiment_name))

from powerlift.bench import retrieve_pmlb
from powerlift.bench import Benchmark, Store, populate_with_datasets
from powerlift.executors import LocalMachine
from itertools import chain
import os

# Initialize database (if needed).
store = Store(f"sqlite:///{os.getcwd()}/powerlift.db", force_recreate=force_recreate)

cache_dir = "~/.powerlift"
data_retrieval = chain(retrieve_pmlb(cache_dir=cache_dir))

# This downloads datasets once and feeds into the database.
populate_with_datasets(store, data_retrieval, exist_ok=exist_ok)

# Run experiment
benchmark = Benchmark(store, name=experiment_name)
benchmark.run(trial_runner, trial_filter, executor=LocalMachine(store, debug_mode=True))

benchmark.wait_until_complete()

status_df = benchmark.status()
for errmsg in status_df["errmsg"]:
    if errmsg is not None:
        print("ERROR: " + str(errmsg))
print(status_df["status"].value_counts().to_string(index=True, header=False))

# reload if analyzing later
results_df = joblib.load("benchmark_results_so_far.zip")
averages = (
    results_df.groupby(["method", "name"])["num_val"].mean().unstack().reset_index()
)
averages.to_excel("averages.xlsx")
