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

    if task.origin == "openml":
        exclude_set = set()
        #        exclude_set = set(['isolet', 'Devnagari-Script', 'CIFAR_10'])
        exclude_set = set(
            [
                "Fashion-MNIST",
                "mfeat-pixel",
                "Bioresponse",
                "mfeat-factors",
                "isolet",
                "cnae-9",
                "Internet-Advertisements",
                "har",
                "Devnagari-Script",
                "mnist_784",
                "CIFAR_10",
                "letter",
            ]
        )
        if task.name in exclude_set:
            return []
        return []
    elif task.origin == "pmlb":
        if task.problem == "binary":
            return []
        elif task.problem == "multiclass":
            return []
        elif task.problem == "regression":
            pass  # include PMLB regression datasets
        else:
            raise Exception(f"Unrecognized task problem {task.problem}")

        exclude_set = set(["1595_poker"])
        if task.name in exclude_set:
            return []
    else:
        raise Exception(f"Unrecognized task origin {task.origin}")

    exclude_set = completed_so_far.copy()
    if task.name in exclude_set:
        return []

    return [
        # "lightgbm",
        "aplr",
    ]


def trial_runner(trial):
    seed = 42

    from lightgbm import LGBMClassifier, LGBMRegressor
    from aplr import APLRTuner
    from sklearn.metrics import roc_auc_score, root_mean_squared_error, r2_score, log_loss
    from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
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

    # Parameter grids
    lightgbm_parameters = {
        "n_estimators": [10, 100, 500, 1000, 2000, 3000, 5000, 10000, 20000],
        "num_leaves": [2, 3, 4],
        "verbosity": [0],
    }
    aplr_parameters = {
        "max_interaction_level": [0, 1],
        "min_observations_in_split": [2, 10, 20, 100, 500, 1000],
        "v": [0.1],
        "max_terms": [2000],
    }

    # Specify method
    if trial.task.problem in ["binary", "multiclass"]:
        if trial.method.name == "lightgbm":
            est = GridSearchCV(
                estimator=LGBMClassifier(random_state=seed),
                param_grid=lightgbm_parameters,
            )
        elif trial.method.name == "aplr":
            est = Pipeline(
                [
                    ("ct", ct),
                    (
                        "est",
                        APLRTuner(parameters=aplr_parameters, is_regressor=False),
                    ),
                ]
            )
            y_train = y_train.astype(str)
            y_test = y_test.astype(str)
        else:
            raise RuntimeError(f"Method unavailable for {trial.method.name}")

        predict_fn = est.predict_proba
    elif trial.task.problem == "regression":
        if trial.method.name == "lightgbm":
            est = GridSearchCV(
                estimator=LGBMRegressor(random_state=seed),
                param_grid=lightgbm_parameters,
            )
        elif trial.method.name == "aplr":
            est = Pipeline(
                [
                    ("ct", ct),
                    (
                        "est",
                        APLRTuner(parameters=aplr_parameters, is_regressor=True),
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
    if trial.method.name == "aplr":
        model: APLRTuner = est[1]
        best_results = model.get_cv_results()[0]
        for param in aplr_parameters:
            trial.log(param, best_results[param])
        trial.log("rows", X_train.shape[0])
        trial.log("columns", X_train.shape[1])
        trial.log("columns_transformed", ct.transform(X_train).shape[1])

    # Predict
    start_time = time()
    predictions = predict_fn(X_test)
    predictions_train = predict_fn(X_train)
    elapsed_time = time() - start_time
    trial.log("predict_time", elapsed_time)

    if trial.task.problem == "binary":
        predictions = predictions[:, 1]
        predictions_train = predictions_train[:, 1]

        eval_score = roc_auc_score(y_test, predictions)
        trial.log("auc", eval_score)

        eval_score_train = roc_auc_score(y_train, predictions_train)
        trial.log("auc_train", eval_score_train)

        eval_score2 = log_loss(y_test, predictions)
        trial.log("log_loss", eval_score2)
    elif trial.task.problem == "multiclass":
        eval_score = roc_auc_score(
            y_test, predictions, average="weighted", multi_class="ovo"
        )
        trial.log("multi_auc", eval_score)

        eval_score_train = roc_auc_score(
            y_train, predictions_train, average="weighted", multi_class="ovo"
        )
        trial.log("multi_auc_train", eval_score_train)

        eval_score2 = log_loss(y_test, predictions)
        trial.log("cross_entropy", eval_score2)
    elif trial.task.problem == "regression":
        # Use NRMSE-IQR (normalized root mean square error by the interquartile range)
        # so that datasets with large predicted values do not dominate the benchmark
        # and the range is not sensitive to outliers. The rank is identical to RMSE.
        # https://en.wikipedia.org/wiki/Root_mean_square_deviation

        q75, q25 = np.percentile(y_train, [75, 25])
        interquartile_range = q75 - q25

        eval_score = root_mean_squared_error(y_test, predictions) / interquartile_range
        trial.log("nrmse", eval_score)

        rsqr = r2_score(y_test, predictions)
        trial.log("rsqr", rsqr)

        rsqr_train = r2_score(y_train, predictions_train)
        trial.log("rsqr_train", rsqr_train)
    else:
        raise Exception(f"Unrecognized task problem {trial.task.problem}")

    if trial.method.name == "aplr":
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

from powerlift.bench import retrieve_openml, retrieve_pmlb, retrieve_catboost_50k
from powerlift.bench import Benchmark, Store, populate_with_datasets
from powerlift.executors import LocalMachine
from itertools import chain
import os

# Initialize database (if needed).
store = Store(f"sqlite:///{os.getcwd()}/powerlift.db", force_recreate=force_recreate)

cache_dir = "~/.powerlift"
data_retrieval = chain(
    # retrieve_catboost_50k(cache_dir=cache_dir),
    retrieve_pmlb(cache_dir=cache_dir),
    retrieve_openml(cache_dir=cache_dir),
)

# This downloads datasets once and feeds into the database.
populate_with_datasets(store, data_retrieval, exist_ok=exist_ok)

# Run experiment
benchmark = Benchmark(store, name=experiment_name)
benchmark.run(trial_runner, trial_filter, executor=LocalMachine(store, debug_mode=True))

benchmark.wait_until_complete()

# re-establish connection
# benchmark = Benchmark(conn_str, name=experiment_name)

status_df = benchmark.status()
for errmsg in status_df["errmsg"]:
    if errmsg is not None:
        print("ERROR: " + str(errmsg))
print(status_df["status"].value_counts().to_string(index=True, header=False))

# reload if analyzing later
results_df = joblib.load("benchmark_results_so_far.zip")

# %%
import pandas as pd

averages = (
    results_df.groupby(["method", "name"])["num_val"].mean().unstack().reset_index()
)

metric_ranks = results_df.pivot_table("num_val", ["task", "name"], "method")
metric_ranks = metric_ranks.rank(axis=1, ascending=True, method="min")
metric_ranks = metric_ranks.groupby("name").mean().transpose()
metric_ranks.columns = [f"{col}_RANK" for col in metric_ranks.columns]
metric_ranks = metric_ranks.reset_index()

overall_rank = results_df[
    results_df["name"].isin(["log_loss", "cross_entropy", "nrmse"])
]
overall_rank = overall_rank.pivot_table("num_val", "task", "method")
overall_rank = overall_rank.rank(axis=1, ascending=True, method="min")
overall_rank = overall_rank.mean()
overall_rank = overall_rank.to_frame(name="RANK").reset_index()

desired_columns = [
    "method",
    "RANK",
    "auc",
    "multi_auc",
    "nrmse",
    "log_loss_RANK",
    "cross_entropy_RANK",
    "nrmse_RANK",
    "fit_time",
    "predict_time",
]
combined_df = averages.merge(metric_ranks, on="method").merge(overall_rank, on="method")
combined_df = combined_df.reindex(columns=desired_columns)
combined_df = combined_df.sort_values(by="RANK")

print(combined_df.to_string(index=False))
combined_df.to_excel("combined_df.xlsx")
