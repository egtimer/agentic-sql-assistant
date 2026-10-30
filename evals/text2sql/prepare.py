"""Download, verify and extract BIRD Mini-Dev, and write the fixed dev/test split.

Usage: ``python -m evals.text2sql.prepare`` (or ``make bird-data``).
"""

import hashlib
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import httpx

from evals.text2sql.data import (
    BIRD_DIR,
    DIFFICULTIES,
    QUESTIONS_PATH,
    RAW_DIR,
    SPLITS_DIR,
    Question,
    load_questions,
)

# Pinned to the dataset revision the reported runs used (2026-01-18). On 2026-10-26 the
# dataset's main branch replaced this file with a reviewed update, so "main" 404s here.
QUESTIONS_REVISION = "f65faf4ae3b638c1fa6df1d3370c8d92c8366301"
QUESTIONS_URL = (
    f"https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/{QUESTIONS_REVISION}/"
    "data/mini_dev_sqlite-00000-of-00001.json"
)
QUESTIONS_SHA256 = "88ceb0710163cae46a256ecea8f0a8c98286599530b60587fda5c3cfe57d45d2"
ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
ZIP_SHA256 = "cc48ba16838204e4e214512030cb572eeb5f7bcdd999bae4b9b6ff12ec13b92f"
ZIP_PATH = RAW_DIR / "minidev.zip"

DB_MEMBER_PREFIX = ("minidev", "MINIDEV", "dev_databases")
N_DEV = 150
SEED = 0


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (through a .part file so partial downloads are not kept)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with open(part, "wb") as f:
            for chunk in response.iter_bytes(1 << 20):
                f.write(chunk)
    part.replace(dest)


def ensure_file(path: Path, url: str, expected_sha256: str) -> None:
    """Download ``path`` if missing, then verify its sha256."""
    if not path.exists():
        print(f"Downloading {url}")
        download(url, path)
    actual = sha256_of(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"sha256 mismatch for {path}: expected {expected_sha256}, got {actual}. "
            "Delete the file and run again."
        )
    print(f"OK {path.name} sha256 {actual[:12]}...")


def _is_unsafe(name: str) -> bool:
    path = PurePosixPath(name)
    return path.is_absolute() or ".." in path.parts or "\\" in name


def extract_databases(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract only ``minidev/MINIDEV/dev_databases/<db>/<file>.sqlite`` into ``dest_dir``.

    The result is ``dest_dir/dev_databases/<db>/<file>.sqlite``. Archives with absolute
    paths or ``..`` components are rejected entirely.
    """
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for info in members:
            if _is_unsafe(info.filename):
                raise ValueError(f"Unsafe path in archive: {info.filename!r}")
        for info in members:
            parts = PurePosixPath(info.filename).parts
            if (
                info.is_dir()
                or len(parts) != len(DB_MEMBER_PREFIX) + 2
                or parts[: len(DB_MEMBER_PREFIX)] != DB_MEMBER_PREFIX
                or not parts[-1].endswith(".sqlite")
            ):
                continue
            target = dest_dir / "dev_databases" / parts[-2] / parts[-1]
            if target.exists() and target.stat().st_size == info.file_size:
                extracted.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted.append(target)
    return extracted


def make_split(
    questions: list[Question], n_dev: int = N_DEV, seed: int = SEED
) -> tuple[list[int], list[int]]:
    """Stratified (by difficulty) dev/test split; returns sorted question ids.

    The dev size per difficulty is proportional, rounded with the largest-remainder method.
    """
    by_difficulty: dict[str, list[int]] = defaultdict(list)
    for q in questions:
        by_difficulty[q.difficulty].append(q.question_id)
    total = len(questions)
    if not 0 <= n_dev <= total:
        raise ValueError(f"n_dev must be between 0 and {total}")

    quotas = {d: n_dev * len(by_difficulty[d]) / total for d in DIFFICULTIES}
    sizes = {d: int(quotas[d]) for d in DIFFICULTIES}
    leftover = n_dev - sum(sizes.values())
    by_remainder = sorted(DIFFICULTIES, key=lambda d: (-(quotas[d] - sizes[d]), d))
    for d in by_remainder[:leftover]:
        sizes[d] += 1

    rng = random.Random(seed)
    dev: list[int] = []
    test: list[int] = []
    for d in DIFFICULTIES:
        ids = sorted(by_difficulty[d])
        rng.shuffle(ids)
        dev.extend(ids[: sizes[d]])
        test.extend(ids[sizes[d] :])
    return sorted(dev), sorted(test)


def write_split(
    splits_dir: Path | None = None,
    questions: list[Question] | None = None,
    n_dev: int = N_DEV,
) -> bool:
    """Write dev.json and test.json if missing. Returns True when files were written."""
    out = splits_dir or SPLITS_DIR
    dev_path, test_path = out / "dev.json", out / "test.json"
    if dev_path.exists() and test_path.exists():
        return False
    dev, test = make_split(questions if questions is not None else load_questions(), n_dev)
    out.mkdir(parents=True, exist_ok=True)
    for path, ids in ((dev_path, dev), (test_path, test)):
        path.write_text(json.dumps(ids) + "\n", encoding="utf-8")
    return True


def main() -> None:
    ensure_file(QUESTIONS_PATH, QUESTIONS_URL, QUESTIONS_SHA256)
    ensure_file(ZIP_PATH, ZIP_URL, ZIP_SHA256)
    databases = extract_databases(ZIP_PATH, BIRD_DIR)
    print(f"{len(databases)} databases available in {BIRD_DIR / 'dev_databases'}")
    if write_split():
        print(f"Wrote split files to {SPLITS_DIR}")
    else:
        print("Split files already exist, left unchanged")


if __name__ == "__main__":
    main()
