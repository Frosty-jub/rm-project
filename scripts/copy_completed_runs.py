"""Copy only hash-verified complete task folders into a fresh suite output."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("source and destination must be separate folders")
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    copied, skipped = [], []
    for folder in sorted(source.glob("batch*_to_*")):
        result_path, model_path = folder / "result.json", folder / "model.pt"
        if not result_path.is_file() or not model_path.is_file():
            skipped.append(folder.name)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != result["model_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {folder}")
        target = destination / folder.name
        if target.exists():
            existing = target / "result.json"
            if not existing.is_file() or json.loads(existing.read_text(encoding="utf-8"))["model_sha256"] != result["model_sha256"]:
                raise ValueError(f"destination contains different or incomplete run: {target}")
            continue
        shutil.copytree(folder, target)
        if hashlib.sha256((target / "model.pt").read_bytes()).hexdigest() != result["model_sha256"]:
            raise IOError(f"copied checkpoint hash mismatch: {target}")
        copied.append(folder.name)
    print(json.dumps({"copied": copied, "skipped_incomplete": skipped}, indent=2))


if __name__ == "__main__":
    main()
