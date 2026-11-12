"""Training-only feature-budget controls and the proposed QUANT extension."""
from time import perf_counter
import warnings
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif


def forest(seed, n_jobs=8, trees=200):
    return ExtraTreesClassifier(n_estimators=trees, max_features=0.1,
                                criterion="entropy", random_state=seed, n_jobs=n_jobs)


def evenly_spaced(indices, count):
    count = min(len(indices), max(1, int(count)))
    return np.asarray(indices)[np.linspace(0, len(indices)-1, count, dtype=int)]


def pilot_importance(x, y, seed, n_jobs=8):
    started = perf_counter()
    pilot = forest(seed + 10000, n_jobs, trees=32)
    pilot.fit(x, y)
    importance = pilot.feature_importances_
    seconds = perf_counter() - started
    return importance, seconds


def ranked_with_coverage(importance, budget):
    f = len(importance)
    budget = min(f, max(1, int(budget)))
    ranking = np.argsort(-importance, kind="stable")
    core_count = max(1, int(np.ceil(0.8 * budget)))
    core = ranking[:core_count]
    remaining = np.setdiff1d(np.arange(f), core, assume_unique=True)
    reserve_count = budget - core_count
    reserve = evenly_spaced(remaining, reserve_count) if reserve_count else np.empty(0, dtype=int)
    return np.sort(np.concatenate([core, reserve]))


def select(method, x, y, seed, n_jobs=8, mass=0.95, pilot=None):
    """Return column indices and selection time, charging a reused pilot fully."""
    started = perf_counter()
    f = x.shape[1]
    pilot_seconds = 0.0
    extra = {}
    if method == "baseline":
        columns = np.arange(f)
    elif method == "uniform_half":
        columns = evenly_spaced(np.arange(f), int(np.ceil(f/2)))
    elif method == "fisher_half":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            scores, _ = f_classif(x, y)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(scores.dtype).max, neginf=-np.inf)
        columns = np.sort(np.argsort(-scores, kind="stable")[:int(np.ceil(f/2))])
    elif method in ("pilot_half", "adaptive_pilot"):
        if pilot is None:
            importance, pilot_seconds = pilot_importance(x, y, seed, n_jobs)
            # The elapsed time below already includes this pilot.
            charged_pilot = 0.0
        else:
            importance, pilot_seconds = pilot
            charged_pilot = pilot_seconds
        ranking = np.argsort(-importance, kind="stable")
        cumulative = np.cumsum(importance[ranking])
        raw_count = int(np.searchsorted(cumulative, mass)) + 1 if cumulative[-1] > 0 else f
        if method == "pilot_half":
            budget = int(np.ceil(f/2))
        else:
            budget = int(np.clip(np.ceil(raw_count/0.8), np.ceil(f/4), np.ceil(3*f/4)))
        columns = ranked_with_coverage(importance, budget)
        extra = {"pilot_seconds": pilot_seconds, "importance_prefix_count": raw_count,
                 "retained_importance": float(importance[columns].sum())}
        return columns, perf_counter()-started+charged_pilot, extra
    else:
        raise ValueError(method)
    return columns, perf_counter()-started, extra


def tree_array_bytes(model):
    total = 0
    for estimator in model.estimators_:
        state = estimator.tree_.__getstate__()
        total += state["nodes"].nbytes + state["values"].nbytes
    return total
