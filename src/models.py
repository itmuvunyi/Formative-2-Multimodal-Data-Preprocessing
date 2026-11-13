"""Task 4: the three models and their evaluation.

Face and voice share one open-set identifier: a closed-set classifier always names
a member, even for a stranger, so it is paired with a similarity check that can say
"not one of ours".

Evaluation never lets an augmented copy of a test recording into training: whole
expressions (face) or whole phrases (voice) are held out together.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE
from .tabular import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


def score(y_true, proba, classes) -> dict:
    classes = np.asarray(classes)
    pred = classes[np.argmax(proba, axis=1)]
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, proba, labels=classes)),
    }


def _identity_candidates(n_train: int) -> dict[str, Pipeline]:
    components = max(2, min(20, n_train - 2))

    def build(clf):
        return Pipeline([("scale", StandardScaler()),
                         ("pca", PCA(n_components=components, random_state=RANDOM_STATE)),
                         ("clf", clf)])

    return {
        "logistic_regression": build(LogisticRegression(C=0.1, max_iter=5000)),
        "random_forest": build(RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                      random_state=RANDOM_STATE, n_jobs=-1)),
    }


def _verifier(n_train: int, n_classes: int) -> Pipeline:
    """Discriminative embedding used only for the "is this anyone we know?" check."""
    components = max(1, min(20, n_train - n_classes - 1))
    return Pipeline([("scale", StandardScaler()),
                     ("pca", PCA(n_components=components, random_state=RANDOM_STATE)),
                     ("lda", LinearDiscriminantAnalysis())])


def _similarity(verifier: Pipeline, X, predicted, X_gallery, y_gallery) -> np.ndarray:
    """Closeness (0 to 1) to the nearest enrolled sample of the predicted person."""
    query, gallery = verifier.transform(X), verifier.transform(X_gallery)
    distances = euclidean_distances(query, gallery)
    y_gallery = np.asarray(y_gallery)
    return np.array([1 / (1 + distances[i, y_gallery == name].min()) for i, name in enumerate(predicted)])


class OpenSetIdentifier:
    """Names an enrolled person, or returns None when the sample looks like nobody enrolled.

    Two gates must both pass: the classifier's confidence, and similarity to that
    person's enrolled samples. Known strangers, when supplied, sharpen the similarity
    gate; each stranger is held out in turn so its threshold reflects unseen people.
    """

    def __init__(self, modality: str):
        self.modality = modality

    def _fit_verifier(self, X, y, X_imp=None) -> Pipeline:
        if X_imp is not None and len(X_imp):
            X_all = np.vstack([X, X_imp])
            y_all = np.concatenate([y, np.full(len(X_imp), "__stranger__")])
        else:
            X_all, y_all = X, y
        return _verifier(len(X_all), len(np.unique(y_all))).fit(X_all, y_all)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            X_impostor: np.ndarray | None = None, impostor_ids: np.ndarray | None = None) -> "OpenSetIdentifier":
        self.classes_ = np.unique(y)
        splitter = LeaveOneGroupOut()
        folds_idx = list(splitter.split(X, y, groups))
        n_train = min(len(train) for train, _ in folds_idx)
        results, oof = {}, {}
        for name, candidate in _identity_candidates(n_train).items():
            proba = np.zeros((len(y), len(self.classes_)))
            folds = []
            for train, test in folds_idx:
                model = clone(candidate).fit(X[train], y[train])
                proba[test] = pd.DataFrame(model.predict_proba(X[test]), columns=model.classes_) \
                    .reindex(columns=self.classes_, fill_value=0.0).to_numpy()
                folds.append(score(y[test], proba[test], self.classes_) | {"held_out": str(groups[test][0])})
            results[name] = score(y, proba, self.classes_) | {"folds": folds}
            oof[name] = proba
        self.cv_results_ = results
        self.model_name_ = min(results, key=lambda k: (-results[k]["f1_macro"], results[k]["log_loss"]))
        proba = oof[self.model_name_]
        predicted = self.classes_[proba.argmax(1)]
        correct = predicted == y
        self.confusion_ = confusion_matrix(y, predicted, labels=self.classes_).tolist()

        has_impostors = X_impostor is not None and len(X_impostor) > 0
        identities = np.unique(impostor_ids) if has_impostors else [None]
        genuine_sim, impostor_sim, impostor_conf = np.zeros(len(y)), [], []
        for identity in identities:
            mask = (impostor_ids == identity) if has_impostors else None
            for train, test in folds_idx:
                verifier = self._fit_verifier(X[train], y[train], X_impostor[~mask] if has_impostors else None)
                genuine_sim[test] += _similarity(verifier, X[test], predicted[test], X[train], y[train])
                if has_impostors:
                    model = clone(_identity_candidates(n_train)[self.model_name_]).fit(X[train], y[train])
                    p = model.predict_proba(X_impostor[mask])
                    guesses = model.classes_[p.argmax(1)]
                    impostor_sim += list(_similarity(verifier, X_impostor[mask], guesses, X[train], y[train]))
                    impostor_conf += list(p.max(1))
        genuine_sim /= len(identities)

        genuine_conf = proba.max(1)[correct] if correct.any() else proba.max(1)
        reference = genuine_sim[correct] if correct.any() else genuine_sim
        self.min_confidence_ = float(max(1.0 / len(self.classes_), np.percentile(genuine_conf, 5) * 0.9))
        self.min_similarity_ = float(np.percentile(reference, 5) * 0.95)
        far = None
        if has_impostors:
            imp_sim, imp_conf = np.array(impostor_sim), np.array(impostor_conf)
            candidates = np.unique(np.concatenate([reference, imp_sim]))
            balanced = [(np.mean(reference >= t) + np.mean(imp_sim < t)) / 2 for t in candidates]
            ties = candidates[np.isclose(balanced, np.max(balanced))]
            self.min_similarity_ = float(ties[len(ties) // 2])
            far = float(((imp_conf >= self.min_confidence_) & (imp_sim >= self.min_similarity_)).mean())
        accepted = (proba.max(1) >= self.min_confidence_) & (genuine_sim >= self.min_similarity_)
        self.open_set_ = {
            "false_reject_rate": float(1 - (accepted & correct).sum() / max(correct.sum(), 1)),
            "false_accept_rate_held_out_strangers": far,
            "wrong_person_accepted_rate": float((accepted & ~correct).mean()),
        }
        self.model_ = clone(_identity_candidates(len(y))[self.model_name_]).fit(X, y)
        self.verifier_ = self._fit_verifier(X, y, X_impostor if has_impostors else None)
        self.X_gallery_, self.y_gallery_ = X, y
        return self

    def identify(self, features: np.ndarray) -> dict:
        X = np.asarray(features, dtype=float).reshape(1, -1)
        proba = self.model_.predict_proba(X)[0]
        closest = self.model_.classes_[proba.argmax()]
        sim = _similarity(self.verifier_, X, [closest], self.X_gallery_, self.y_gallery_)[0]
        accepted = proba.max() >= self.min_confidence_ and sim >= self.min_similarity_
        return {"member": str(closest) if accepted else None, "closest_member": str(closest),
                "confidence": float(proba.max()), "similarity": float(sim),
                "min_confidence": self.min_confidence_, "min_similarity": self.min_similarity_}

    def summary(self) -> dict:
        return {"selected_model": self.model_name_, "members": self.classes_.tolist(),
                "cv": self.cv_results_, "confusion_matrix": self.confusion_,
                "thresholds": {"min_confidence": self.min_confidence_, "min_similarity": self.min_similarity_},
                "open_set": self.open_set_}


def train_identity(df: pd.DataFrame, meta: list[str], group_col: str, modality: str) -> OpenSetIdentifier:
    feature_cols = [c for c in df.columns if c not in meta]
    enrolled = df[df["member"] != "unauthorized"]
    impostors = df[df["member"] == "unauthorized"]
    X_imp, ids = None, None
    if len(impostors):
        X_imp = impostors[feature_cols].to_numpy(float)
        ids = impostors["source_file"].str.extract(r"^(unknown_\d+)", expand=False) \
            .fillna(impostors["source_file"]).to_numpy(str)
    model = OpenSetIdentifier(modality).fit(enrolled[feature_cols].to_numpy(float), enrolled["member"].to_numpy(str),
                                            enrolled[group_col].to_numpy(str), X_imp, ids)
    model.feature_cols_ = feature_cols
    return model


def _product_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


def product_candidates() -> dict[str, Pipeline]:
    classifiers = {
        "baseline_class_frequency": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=80,
                                                            random_state=RANDOM_STATE),
    }
    return {k: Pipeline([("prep", _product_preprocessor()), ("clf", v)]) for k, v in classifiers.items()}


def _cv_proba(model: Pipeline, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int, classes) -> np.ndarray:
    proba = np.zeros((len(y), len(classes)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
            fitted = clone(model).fit(X.iloc[train], y[train])
            proba[test] = pd.DataFrame(fitted.predict_proba(X.iloc[test]), columns=fitted.classes_) \
                .reindex(columns=classes, fill_value=1e-6).to_numpy()
    return proba


def train_product(merged: pd.DataFrame, n_repeats: int = 5, n_permutations: int = 100) -> tuple[Pipeline, dict]:
    """Customer-grouped repeated CV, a time-ordered holdout, and a label-shuffle test."""
    X, y = merged[FEATURES], merged[TARGET].to_numpy(str)
    groups = merged["customer_id"].to_numpy()
    classes = np.unique(y)
    results = {}
    for name, model in product_candidates().items():
        runs = [score(y, _cv_proba(model, X, y, groups, s, classes), classes) for s in range(n_repeats)]
        results[name] = {m: float(np.mean([r[m] for r in runs])) for m in runs[0]} | \
                        {"accuracy_std": float(np.std([r["accuracy"] for r in runs]))}
    real = [k for k in results if not k.startswith("baseline")]
    best = min(real, key=lambda k: results[k]["log_loss"])

    ordered = merged.sort_values("purchase_date")
    cut = int(len(ordered) * 0.8)
    train_part, test_part = ordered.iloc[:cut], ordered.iloc[cut:]
    holdout = {}
    for name in ("baseline_class_frequency", best):
        fitted = clone(product_candidates()[name]).fit(train_part[FEATURES], train_part[TARGET].to_numpy(str))
        proba = pd.DataFrame(fitted.predict_proba(test_part[FEATURES]), columns=fitted.classes_) \
            .reindex(columns=classes, fill_value=1e-6).to_numpy()
        holdout[name] = score(test_part[TARGET].to_numpy(str), proba, classes)
    holdout["test_rows"] = len(test_part)
    holdout["test_starts"] = str(test_part["purchase_date"].min().date())

    # Is the accuracy better than the same model on shuffled labels?
    rng = np.random.default_rng(RANDOM_STATE)
    observed = score(y, _cv_proba(product_candidates()[best], X, y, groups, 0, classes), classes)["accuracy"]
    fast = clone(product_candidates()[best])
    if best == "random_forest":
        fast.set_params(clf__n_estimators=100)
    null = [score(s := rng.permutation(y), _cv_proba(fast, X, s, groups, 0, classes), classes)["accuracy"]
            for _ in range(n_permutations)]
    p_value = (1 + sum(v >= observed for v in null)) / (n_permutations + 1)

    final_proba = _cv_proba(product_candidates()[best], X, y, groups, 0, classes)
    final = clone(product_candidates()[best]).fit(X, y)
    report = {
        "selected_model": best, "n_rows": len(merged), "n_customers": int(merged["customer_id"].nunique()),
        "classes": classes.tolist(), "cv_customer_grouped": results, "time_holdout": holdout,
        "permutation_test": {"observed_accuracy": observed, "null_mean_accuracy": float(np.mean(null)),
                             "null_95th_percentile": float(np.percentile(null, 95)),
                             "p_value": float(p_value), "n_permutations": n_permutations},
        "confusion_matrix": confusion_matrix(y, classes[final_proba.argmax(1)], labels=classes).tolist(),
    }
    return final, report
