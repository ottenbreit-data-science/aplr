import pickle

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aplr import APLRClassifier, APLRRegressor


def _regression_data():
    X = np.column_stack([np.linspace(0.0, 1.0, 100), np.linspace(1.0, 2.0, 100)])
    y = 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 1]
    return X, y


def _classification_data():
    X = np.column_stack([np.linspace(0.0, 1.0, 100), np.linspace(1.0, 2.0, 100)])
    y = np.where(X[:, 0] > 0.5, "A", "B")
    return X, y


def test_regressor_clone_and_fit_contract():
    X, y = _regression_data()
    estimator = APLRRegressor(m=10, cv_folds=2, preprocess=False)

    cloned = clone(estimator)

    assert cloned is not estimator
    assert cloned.get_params() == estimator.get_params()
    assert cloned.fit(X, y) is cloned
    assert cloned.n_features_in_ == X.shape[1]


def test_classifier_clone_and_fit_contract():
    X, y = _classification_data()
    estimator = APLRClassifier(m=10, cv_folds=2, preprocess=False)

    cloned = clone(estimator)

    assert cloned is not estimator
    assert cloned.get_params() == estimator.get_params()
    assert cloned.fit(X, y) is cloned
    assert cloned.n_features_in_ == X.shape[1]
    assert np.array_equal(cloned.classes_, np.array(["A", "B"]))
    assert cloned.predict_proba(X[:5]).shape == (5, 2)
    assert np.allclose(cloned.predict_proba(X[:5]).sum(axis=1), 1.0)


def test_regressor_score_uses_r2():
    X, y = _regression_data()
    model = APLRRegressor(m=10, cv_folds=2, preprocess=False).fit(X, y)

    score = model.score(X, y)

    assert np.isfinite(score)
    assert score > 0.9


def test_classifier_score_uses_accuracy():
    X, y = _classification_data()
    model = APLRClassifier(m=10, cv_folds=2, preprocess=False).fit(X, y)

    score = model.score(X, y)

    assert score == pytest.approx(1.0, abs=0.05)
    assert np.array_equal(model.classes_, np.array(["A", "B"]))


def test_predictions_require_fitting():
    X, _ = _regression_data()
    regressor = APLRRegressor(m=10, preprocess=False)
    classifier = APLRClassifier(m=10, preprocess=False)

    with pytest.raises(
        RuntimeError, match=r"must be trained with fit\(\)|not fitted yet"
    ):
        regressor.predict(X)
    with pytest.raises(
        RuntimeError, match=r"must be trained with fit\(\)|not fitted yet"
    ):
        classifier.predict(X)
    with pytest.raises(
        RuntimeError, match=r"must be trained with fit\(\)|not fitted yet"
    ):
        classifier.predict_proba(X)


def test_legacy_model_pickle_still_predicts():
    X, y = _regression_data()
    regressor = APLRRegressor(m=10, cv_folds=2, preprocess=False)
    regressor.fit(X, y)
    regressor.__dict__.pop("n_features_in_", None)

    payload = pickle.dumps(regressor)
    restored = pickle.loads(payload)
    prediction = restored.predict(X[:5])

    assert prediction.shape == (5,)

    Xc, yc = _classification_data()
    classifier = APLRClassifier(m=10, cv_folds=2, preprocess=False)
    classifier.fit(Xc, yc)
    classifier.__dict__.pop("n_features_in_", None)
    classifier.__dict__.pop("classes_", None)

    payload = pickle.dumps(classifier)
    restored = pickle.loads(payload)
    prediction = restored.predict(Xc[:5])

    assert len(prediction) == 5


def test_regressor_works_in_pipeline():
    X, y = _regression_data()
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", APLRRegressor(m=10, cv_folds=2, preprocess=False)),
        ]
    )

    assert pipeline.fit(X, y) is pipeline
    assert pipeline.predict(X).shape == y.shape
