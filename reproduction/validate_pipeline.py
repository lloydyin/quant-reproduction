"""Check loader and cached/batched features against the author's execution path."""
import ast
import importlib.util
import json
import re
from types import SimpleNamespace
import run_reproduction as run
from quant import Quant
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier


def main():
    torch.set_num_threads(1)
    tree = ast.parse((run.PROJECT / "quant/code/run.py").read_text())
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    namespace = {"np": np, "re": re}
    exec(compile(functions, "author_run.py", "exec"), namespace)
    reference_path = run.ROOT / "reference/quant_2023.py"
    spec = importlib.util.spec_from_file_location("quant_2023", reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    checks = []
    for name in ["Coffee", "GunPoint", "ItalyPowerDemand", "Chinatown"]:
        paths = run.data_files(name)
        for path in paths:
            x, y = run.load_ts(path)
            old_x, old_y = namespace["load_from_tsfile"](str(path))
            np.testing.assert_array_equal(x, old_x.astype(np.float32))
            np.testing.assert_array_equal(y, old_y)
        cached, labels, meta = run.prepare(name, paths, SimpleNamespace(batch_size=64), run.digest(run.PROJECT / "quant/code/quant.py"))
        x = np.concatenate([run.load_ts(p)[0] for p in paths])
        full = Quant().fit_transform(torch.from_numpy(x), None).numpy()
        original = reference.Quant().fit_transform(torch.from_numpy(x), None).numpy()
        np.testing.assert_array_equal(full, original)
        # PyTorch's FFT/reductions can differ at float32 roundoff across batch
        # sizes; require tight numerical agreement and identical predictions.
        np.testing.assert_allclose(full, cached, rtol=1e-5, atol=2e-5)
        train, test, _ = run.read_indices(name, 1, "java", meta["n_train"], len(labels), labels)
        q = Quant()
        direct_train = q.fit_transform(torch.from_numpy(x[train]), labels[train]).numpy()
        direct_test = q.transform(torch.from_numpy(x[test])).numpy()
        np.testing.assert_allclose(direct_train, cached[train], rtol=1e-5, atol=2e-5)
        np.testing.assert_allclose(direct_test, cached[test], rtol=1e-5, atol=2e-5)
        options = dict(n_estimators=200, max_features=0.1, criterion="entropy", random_state=1, n_jobs=1)
        direct = ExtraTreesClassifier(**options).fit(direct_train, labels[train]).predict(direct_test)
        reused = ExtraTreesClassifier(**options).fit(cached[train], labels[train]).predict(cached[test])
        np.testing.assert_array_equal(direct, reused)
        checks.append({"dataset": name, "loader_equal": True, "2023_2025_features_equal": True,
                       "cached_batched_features_allclose": True,
                       "max_cache_difference": float(np.max(np.abs(full - cached))),
                       "resample_predictions_equal": True})
        print(f"PASS {name}: loader, 2023 code, batching, cache, resample predictions", flush=True)
    (run.ROOT / "validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
