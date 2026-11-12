"""Package code and recorded results without manuscripts, raw data or caches."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    if json.loads((ROOT / "verification.json").read_text())["status"] != "PASS":
        raise SystemExit("Run verify_artifacts.py before packaging")
    out = ROOT / "deliverables"
    out.mkdir(exist_ok=True)
    excluded = {"__pycache__", "data", "data_extra", "downloads", "indices", "features",
                "baseline_results", "smoke"}
    suffixes = {".py", ".md", ".csv", ".json", ".txt", ".pdf", ".png"}
    files = [ROOT / n for n in ["README.md", "UPLOAD_GUIDE.md",
             "verify_artifacts.py", "package_project.py", "verification.json", ".gitignore"]]
    for folder in ["quant", "reproduction", "experiments"]:
        for path in (ROOT / folder).rglob("*"):
            relative = path.relative_to(ROOT / folder)
            if not path.is_file() or any(p in excluded for p in relative.parts):
                continue
            if path.suffix not in suffixes or path.name in {"main.pdf", "thread_check_progress.json", "paper_text.txt", "example_structure.txt"}:
                continue
            files.append(path)
    files = sorted(set(files))
    manifest = {}
    with zipfile.ZipFile(out / "QUANT_Code_and_Results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            archive.write(path, name)
            manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        archive.writestr("PACKAGE_SHA256.json", json.dumps(manifest, indent=2))
    (out / "PACKAGE_SHA256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Packaged {len(files)} files; manuscripts, raw data, feature caches and reference PDFs are excluded.")
    for path in out.glob("*.zip"):
        print(path.name, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
