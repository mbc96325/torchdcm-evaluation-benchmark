from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import compare_real_nested_logit_battery as nested


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_CASES = [
    "swissmetro",
    "lpmc",
    "nhts",
    "parking",
    "airline",
    "mlogit_catsup",
    "mlogit_cracker",
    "mlogit_electricity",
    "mlogit_fishing",
    "mlogit_hc",
    "mlogit_heating",
    "mlogit_mode",
]


def run_case(case: str) -> tuple[dict | None, str]:
    with tempfile.TemporaryDirectory(prefix=f"torchdcm_nested_{case}_") as tmp:
        tmp_path = Path(tmp)
        json_path = tmp_path / "result.json"
        md_path = tmp_path / "result.md"
        command = [
            sys.executable,
            str(Path(nested.__file__).resolve()),
            "--case",
            case,
            "--json-output",
            str(json_path),
            "--md-output",
            str(md_path),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        message = (completed.stderr or completed.stdout).strip()[-2000:]
        if completed.returncode != 0 or not json_path.exists():
            return None, message or f"child exit code {completed.returncode}"
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        return (rows[0] if rows else None), message


def write_outputs(rows: list[dict], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path.write_text(nested.render_markdown(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--json-output", type=Path, default=RESULTS / "nested_real_battery_single_core.json")
    parser.add_argument("--md-output", type=Path, default=RESULTS / "nested_real_battery_single_core.md")
    args = parser.parse_args()

    rows: list[dict] = []
    for case in args.cases:
        print(f"[nested-isolated] running {case}", flush=True)
        row, message = run_case(case)
        if row is None:
            print(f"[nested-isolated] {case}: failed: {message}", flush=True)
            continue
        rows.append(row)
        write_outputs(rows, args.json_output, args.md_output)
        backends = {item["backend"]: item for item in row["backends"]}
        print(
            f"[nested-isolated] {case}: torch={nested.fmt_time(backends.get('torchdcm'))} "
            f"biogeme={nested.fmt_time(backends.get('biogeme'))} "
            f"apollo={nested.fmt_time(backends.get('apollo'))} consistent={row['consistent']}",
            flush=True,
        )
    write_outputs(rows, args.json_output, args.md_output)
    print(f"json: {args.json_output}", flush=True)
    print(f"markdown: {args.md_output}", flush=True)


if __name__ == "__main__":
    main()
