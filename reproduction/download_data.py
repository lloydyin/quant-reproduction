"""Download the public benchmark files linked by the QUANT authors.

Run from any working directory. Archives and SHA256 provenance remain local.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DRIVE = {
    "java_indices": "1FsG7Fp74y_TpaPhJ7U066ot8A07BPhr3",
    "python_indices": "1aLBP_nhnoqz075puKg30zuF3F_QBOXYM",
    "new_datasets": "1vuh6mgNrNKjHr9MMRQP0J0_gGA4dE7E3",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive path: {info.filename}")
        z.extractall(destination)


def download_dataset(name):
    archive = ROOT / "downloads" / f"{name}.zip"
    url = f"https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        temporary = archive.with_suffix(".part")
        request = urllib.request.Request(url, headers={"User-Agent": "QUANT-reproduction/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if not zipfile.is_zipfile(temporary):
            raise ValueError(f"Not a ZIP: {name}")
        temporary.replace(archive)
    # Copy only the canonical splits, regardless of the ZIP's directory layout.
    destination = ROOT / "data" / name
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for split in ("TRAIN", "TEST"):
            filename = f"{name}_{split}.ts"
            matches = [p for p in z.namelist() if Path(p).name.lower() == filename.lower()]
            if len(matches) != 1:
                raise ValueError(f"Expected one {filename}, got {matches}")
            with z.open(matches[0]) as source, (destination / filename).open("wb") as output:
                shutil.copyfileobj(source, output)
    return {"dataset": name, "url": url, "archive_sha256": sha256(archive), "bytes": archive.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--drive", nargs="*", choices=list(DRIVE), default=[])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    (ROOT / "downloads").mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "downloads" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for key in args.drive:
        import gdown
        archive = ROOT / "downloads" / f"{key}.zip"
        if not archive.exists() or not zipfile.is_zipfile(archive):
            gdown.download(id=DRIVE[key], output=str(archive), quiet=False, use_cookies=False)
        if not zipfile.is_zipfile(archive):
            raise ValueError(f"Download failed: {key}")
        if key == "new_datasets":
            unpack(archive, ROOT / "data_extra" / key)
        manifest[key] = {"url": f"https://drive.google.com/file/d/{DRIVE[key]}/view", "archive_sha256": sha256(archive), "bytes": archive.stat().st_size}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"READY {key}: {archive.stat().st_size:,} bytes", flush=True)
    names = args.datasets
    if names is None:
        names = (PROJECT / "quant" / "code" / "ucr142.txt").read_text().splitlines()[:112]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_dataset, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                manifest[name] = future.result()
                print(f"READY {name}", flush=True)
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            except Exception as exc:
                failures.append(name)
                print(f"FAILED {name}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"Retry these datasets: {' '.join(failures)}")


if __name__ == "__main__":
    main()
