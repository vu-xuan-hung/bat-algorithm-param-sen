"""Download the pinned UCI benchmark CSVs into the local datasets directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

DATASETS = {
    "rice": (545, "a0d9b19693f0bde9a59a6a6e886df0456008fab1c3a406178a76e7c4440e95b3"),
    "dry_bean": (602, "c91a086bb4357e69409d23287ec22be1be9512145cead1d8ca988c918be38fcb"),
    "htru2": (372, "f2ad10d1da05ebd2fa6b4e610e314284e646d56663cc64b889506e3a1c59c814"),
    "letter_recognition": (
        59,
        "9ec44e47110540d997fc2e247464a75ed3da0ad9dfcd49ded7e68fcaab04a9f7",
    ),
}
DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"


def fetch_dataset(name: str) -> None:
    dataset_id, expected_sha256 = DATASETS[name]
    destination = DATASET_DIR / f"{name}.csv"
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError(
                f"{destination} differs from the pinned UCI CSV. Move it aside to retry."
            )
        print(f"Verified {destination}")
        return

    url = f"https://archive.ics.uci.edu/static/public/{dataset_id}/data.csv"
    with urlopen(url, timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"The SHA-256 for {url} differs from the pinned UCI CSV.")

    with destination.open("xb") as handle:
        handle.write(payload)
    print(f"Downloaded {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        for name in DATASETS:
            fetch_dataset(name)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
