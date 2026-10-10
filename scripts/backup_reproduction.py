"""Archive source, data, reports, and completed run artifacts outside the workspace."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]


def files():
    selected = set()
    for entry in ROOT.iterdir():
        if entry.name in (".venv", ".venv_gpu", ".git", "runs", "__pycache__", ".pytest_cache"):
            continue
        candidates = entry.rglob("*") if entry.is_dir() else (entry,)
        for path in candidates:
            if path.is_file() and not any(part in ("__pycache__", ".pytest_cache") for part in path.parts):
                selected.add(path)
    runs = ROOT / "runs"
    for directory in (runs / "audit_recovered", runs / "preprocessing_screen",
                      runs / "recovered_legacy_1_to_2"):
        if directory.exists():
            selected.update(path for path in directory.rglob("*") if path.is_file())
    revised = runs / "revised"
    if revised.exists():
        for result in revised.rglob("result.json"):
            selected.update(path for path in result.parent.rglob("*") if path.is_file())
        for name in ("summary.json", "tasks.csv", "audit.json", "audit_partial.json"):
            selected.update(path for path in revised.rglob(name) if path.is_file())
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.is_relative_to(ROOT):
        raise ValueError("backup must be outside the project workspace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    chosen = files()
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "project_root": str(ROOT), "file_count": len(chosen),
                "completed_revised_runs": sum(path.name == "result.json" and
                                              "revised" in path.parts for path in chosen)}
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in chosen:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("BACKUP_MANIFEST.json", json.dumps(manifest, indent=2))
    with ZipFile(destination) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise OSError(f"backup CRC verification failed: {bad}")
    print(json.dumps({**manifest, "destination": str(destination),
                      "bytes": destination.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
