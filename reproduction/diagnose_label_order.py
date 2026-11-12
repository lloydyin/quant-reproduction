"""Separate historical-split diagnostic; never replaces the primary results.

Run after the paired efficiency experiment, not concurrently with its timing.
The author linked a class-wise Python resampler. String and numeric label orders
can allocate the same RNG stream to different classes. Test this one structural
alternative on EVERY additional dataset whose class orders differ.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from run_reproduction import read_indices, digest

ROOT = Path(__file__).resolve().parent


def generate(labels, n_train, order, seed):
    rng = np.random.RandomState(seed)
    parts = []
    for label in order:
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        parts.append(indices[:np.sum(labels[:n_train] == label)])
    train = np.concatenate(parts)
    return train, np.setdiff1d(np.arange(len(labels)), train)


def main():
    out = ROOT / "label_order_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    names = (ROOT.parent / "quant/code/ucr142.txt").read_text().splitlines()[112:]
    audit, affected = [], []
    for name in names:
        cache = ROOT / "features" / name
        meta = json.loads((cache / "metadata.json").read_text())
        labels = np.load(cache / "labels.npy")
        lexical = list(range(len(meta["classes"])))
        try:
            numeric = sorted(lexical, key=lambda i: float(meta["classes"][i]))
        except ValueError:
            # Named classes have no numerical ordering; still audit their
            # current official membership against lexical generation.
            numeric = lexical
        if numeric != lexical: affected.append(name)
        for resample in range(1,30):
            official, _, _ = read_indices(name, resample, "python", meta["n_train"], len(labels), labels)
            lex_train, _ = generate(labels, meta["n_train"], lexical, resample)
            num_train, _ = generate(labels, meta["n_train"], numeric, resample)
            same_set = np.array_equal(np.sort(official), np.sort(lex_train))
            if not same_set:
                raise ValueError(f"Current official indices do not match lexical generation: {name}/{resample}")
            audit.append({"dataset":name,"resample":resample,"official_equals_lexical_set":bool(same_set),
                          "official_equals_lexical_order":bool(np.array_equal(official,lex_train)),
                          "numeric_changes_membership":bool(not np.array_equal(np.sort(official),np.sort(num_train))),
                          "train_overlap_fraction":len(np.intersect1d(official,num_train))/len(official)})
    pd.DataFrame(audit).to_csv(out / "index_audit.csv",index=False)
    config={"datasets":affected,"resamples":list(range(30)),"classifier_seed":"resample index",
            "classifier_label_encoding":"unchanged lexical encoding from the main run; only resampler class iteration changes",
            "source":"https://github.com/angus924/quant/issues/1#issuecomment-1809307377",
            "historical_resampler_commit":"5dac9ce9955bfb4596735ef117438c4c2a299acc",
            "purpose":"diagnostic sensitivity analysis; no historical dtype is asserted as proven",
            "python_indices_sha256":digest(ROOT / "downloads/python_indices.zip")}
    (out / "configuration.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    fields=["dataset","resample","numeric_accuracy","official_accuracy","author_accuracy","fit_seconds"]
    for position,name in enumerate(affected,1):
        cache=ROOT / "features" / name
        meta=json.loads((cache / "metadata.json").read_text())
        labels=np.load(cache / "labels.npy")
        features=np.load(cache / "features.npy",mmap_mode="r")
        order=sorted(range(len(meta["classes"])),key=lambda i:float(meta["classes"][i]))
        reference=pd.read_csv(ROOT / "results" / f"{name}.csv").set_index("resample")
        path=out / f"{name}.csv"
        done=set(pd.read_csv(path)["resample"]) if path.exists() else set()
        for resample in range(30):
            if resample in done: continue
            if resample==0:
                accuracy=float(reference.loc[resample,"accuracy"]); seconds=0.0
            else:
                train,test=generate(labels,meta["n_train"],order,resample)
                x=np.asarray(features[train],order="F")
                start=perf_counter()
                classifier=ExtraTreesClassifier(n_estimators=200,max_features=.1,criterion="entropy",n_jobs=8,random_state=resample)
                classifier.fit(x,labels[train]); seconds=perf_counter()-start
                predictions=np.concatenate([classifier.predict(features[test[i:i+1024]]) for i in range(0,len(test),1024)])
                accuracy=float(np.mean(predictions==labels[test]))
                del x,classifier
            row=dict(dataset=name,resample=resample,numeric_accuracy=accuracy,
                     official_accuracy=float(reference.loc[resample,"accuracy"]),
                     author_accuracy=float(reference.loc[resample,"paper_accuracy"]),fit_seconds=seconds)
            new=not path.exists()
            with path.open("a",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=fields)
                if new: writer.writeheader()
                writer.writerow(row)
            print(f"[{position}/{len(affected)}] {name} r={resample}: numeric={accuracy:.6f} official={row['official_accuracy']:.6f} author={row['author_accuracy']:.6f}",flush=True)
    results=[]
    for name in affected:
        df=pd.read_csv(out / f"{name}.csv")
        if set(df["resample"]) != set(range(30)) or len(df)!=30: raise ValueError("Incomplete diagnostic")
        shuffled=df[df["resample"]>0]
        results.append({"dataset":name,"official_correlation_r1_r29":shuffled.official_accuracy.corr(shuffled.author_accuracy),
                        "numeric_correlation_r1_r29":shuffled.numeric_accuracy.corr(shuffled.author_accuracy),
                        "official_mean":df.official_accuracy.mean(),"numeric_mean":df.numeric_accuracy.mean(),
                        "author_mean":df.author_accuracy.mean(),
                        "official_mean_absolute_error":np.mean(np.abs(shuffled.official_accuracy-shuffled.author_accuracy)),
                        "numeric_mean_absolute_error":np.mean(np.abs(shuffled.numeric_accuracy-shuffled.author_accuracy))})
    pd.DataFrame(results).to_csv(out / "summary.csv",index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__=="__main__": main()
