from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import compare_real_mixed_logit_battery as mixed


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated"
DEFAULT_CASES = [
    "swissmetro",
    "airline",
    "parking",
    "telephone",
    "lpmc",
    "mlogit_catsup",
    "mlogit_cracker",
    "mlogit_electricity",
    "mlogit_fishing",
    "mlogit_hc",
    "mlogit_heating",
    "mlogit_mode",
    "mlogit_modecanada",
    "mlogit_nox",
    "mlogit_risky_transport",
    "mlogit_train",
]


def run_case(case: str, args: argparse.Namespace) -> tuple[dict | None, str]:
    with tempfile.TemporaryDirectory(prefix=f"torchdcm_mixed_{case}_") as tmp:
        tmp_path = Path(tmp)
        json_path = tmp_path / "result.json"
        md_path = tmp_path / "result.md"
        command = [
            sys.executable,
            str(Path(mixed.__file__).resolve()),
            "--datasets",
            case,
            "--n-draws",
            str(args.n_draws),
            "--max-iter",
            str(args.max_iter),
            "--sigma",
            str(args.sigma),
            "--seed",
            str(args.seed),
            "--backend-timeout",
            str(args.backend_timeout),
            "--torch-device",
            "cpu",
            "--json-output",
            str(json_path),
            "--md-output",
            str(md_path),
        ]
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=args.case_timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"case exceeded {args.case_timeout:.0f}s worker limit"
        message = (completed.stderr or completed.stdout).strip()[-2000:]
        if completed.returncode != 0 or not json_path.exists():
            return None, message or f"child exit code {completed.returncode}"
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        return (rows[0] if rows else None), message


def write_outputs(rows: list[dict], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path.write_text(mixed.render_markdown(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--n-draws", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--case-timeout", type=float, default=300.0)
    parser.add_argument("--backend-timeout", type=float, default=300.0)
    parser.add_argument("--json-output", type=Path, default=GENERATED / "mixed_real_battery_single_core.json")
    parser.add_argument("--md-output", type=Path, default=GENERATED / "mixed_real_battery_single_core.md")
    args = parser.parse_args()

    rows: list[dict] = []
    for case in args.cases:
        print(f"[mixed-isolated] running {case}", flush=True)
        row, message = run_case(case, args)
        if row is None:
            print(f"[mixed-isolated] {case}: failed: {message}", flush=True)
            rows.append({"case": case, "status": "skipped", "message": message})
            write_outputs(rows, args.json_output, args.md_output)
            continue
        rows.append(row)
        write_outputs(rows, args.json_output, args.md_output)
        if row.get("status") == "skipped":
            print(f"[mixed-isolated] {case}: skipped: {row.get('message')}", flush=True)
            continue
        torch_row = mixed.backend(row, "torchdcm")
        biogeme_row = mixed.backend(row, "biogeme")
        apollo_row = mixed.backend(row, "apollo")
        print(
            f"[mixed-isolated] {case}: torch={mixed.fmt(torch_row.get('total_s'))} "
            f"biogeme={mixed.fmt(biogeme_row.get('total_s'))} "
            f"apollo={mixed.fmt(apollo_row.get('total_s'))} consistent={row['consistent']}",
            flush=True,
        )
    write_outputs(rows, args.json_output, args.md_output)
    print(f"json: {args.json_output}", flush=True)
    print(f"markdown: {args.md_output}", flush=True)


if __name__ == "__main__":
    main()
