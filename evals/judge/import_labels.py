"""Import the human labels of the judge set.

``labels.json`` (exported by the labeling page) has this shape:

    {"labeler": "name", "labels": {"j-01": {"label": "correct", "note": "..."}, ...}}

``label`` is ``correct``, ``incorrect`` or ``unsure`` (unsure items are left out of the
agreement and counted); ``note`` is optional free text. The file is validated against an
items run: every id must be a known item, and (unless ``--allow-partial``) every item needs a
label. The result is ``labels.jsonl`` in the items run, one line per item.

    uv run python -m evals.judge.import_labels --items evals/results/judge/<items run> \\
        --labels path/to/labels.json
"""

import argparse
import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

from evals.judge.build_set import write_jsonl
from evals.memory.common import read_jsonl

ALLOWED_LABELS = ("correct", "incorrect", "unsure")


def validate_labels(
    data: Any, item_ids: Collection[str], *, allow_partial: bool = False
) -> list[dict[str, Any]]:
    """Records ``{item_id, label, note, labeler}`` sorted by id; ValueError lists all problems."""
    problems: list[str] = []
    if not isinstance(data, dict):
        raise ValueError("labels file must be a JSON object")
    labeler = data.get("labeler")
    if not isinstance(labeler, str) or not labeler.strip():
        problems.append("'labeler' must be a non-empty string")
    labels = data.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("; ".join([*problems, "'labels' must be an object keyed by item id"]))
    known = set(item_ids)
    records = []
    for item_id, entry in labels.items():
        if item_id not in known:
            problems.append(f"unknown item id {item_id!r}")
            continue
        if not isinstance(entry, dict):
            problems.append(f"{item_id}: entry must be an object")
            continue
        label, note = entry.get("label"), entry.get("note", "")
        if label not in ALLOWED_LABELS:
            problems.append(f"{item_id}: label {label!r} is not one of {ALLOWED_LABELS}")
            continue
        if not isinstance(note, str):
            problems.append(f"{item_id}: note must be a string")
            continue
        records.append(
            {"item_id": item_id, "label": label, "note": note, "labeler": str(labeler).strip()}
        )
    if not allow_partial:
        missing = sorted(known - set(labels))
        if missing:
            problems.append(f"{len(missing)} items without a label: {', '.join(missing[:10])}")
    if problems:
        raise ValueError("; ".join(problems))
    return sorted(records, key=lambda r: r["item_id"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--items", type=Path, required=True, help="items run directory")
    parser.add_argument("--labels", type=Path, required=True, help="labels.json from the page")
    parser.add_argument("--out", type=Path, help="default: <items run>/labels.jsonl")
    parser.add_argument("--allow-partial", action="store_true", help="not every item needs a label")
    parser.add_argument("--force", action="store_true", help="overwrite an existing labels.jsonl")
    args = parser.parse_args(argv)
    items = read_jsonl(args.items / "items.jsonl") or read_jsonl(args.items / "items_public.jsonl")
    if not items:
        raise SystemExit(f"no items in {args.items}")
    out = args.out or args.items / "labels.jsonl"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; use --force to overwrite")
    try:
        data = json.loads(args.labels.read_text(encoding="utf-8"))
        records = validate_labels(
            data, [i["item_id"] for i in items], allow_partial=args.allow_partial
        )
    except (ValueError, OSError) as e:  # JSONDecodeError is a ValueError
        raise SystemExit(f"{args.labels}: {e}") from e
    write_jsonl(out, records)
    counts = {label: sum(1 for r in records if r["label"] == label) for label in ALLOWED_LABELS}
    print(f"Wrote {len(records)} labels by {records[0]['labeler'] if records else '?'} to {out}")
    print(
        f"correct {counts['correct']}, incorrect {counts['incorrect']}, unsure {counts['unsure']}"
    )


if __name__ == "__main__":
    main()
