"""Profile riddfmb3d NVML power/temperature for time-step/modulus pairs.

The script creates one derived config and one Nsight Systems report per run. It
never edits examples/config.txt, so each report remains reproducible.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import math
import re
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Iterable


AVERAGE_POWER_JSON_QUERY = """
SELECT ROUND(AVG(CAST(json_extract(data, '$.Power') AS REAL)), 3) AS avg_power_w
FROM GENERIC_EVENTS;
"""

AVERAGE_POWER_NORMALIZED_QUERY = """
SELECT ROUND(
    AVG(COALESCE(event_data.doubleVal, event_data.floatVal,
                 event_data.intVal, event_data.uintVal)),
    3
) AS avg_power_w
FROM GENERIC_EVENTS AS events
JOIN GENERIC_EVENT_DATA AS event_data
  ON event_data.genericEventId = events.genericEventId
JOIN GENERIC_EVENT_TYPE_FIELDS AS fields
  ON fields.typeId = events.typeId
 AND fields.fieldIdx = event_data.fieldIdx
JOIN StringIds AS field_names ON field_names.id = fields.fieldNameId
WHERE field_names.value = 'Power';
"""

SQLITE_EXPORT_ATTEMPTS = 10
SQLITE_EXPORT_RETRY_SECONDS = 1.0


def read_average_power(database: sqlite3.Connection) -> float | None:
    """Read power from either the legacy JSON or current normalized schema."""
    row = database.execute(AVERAGE_POWER_JSON_QUERY).fetchone()
    if row is not None and row[0] is not None:
        return row[0]

    table_names = {
        row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    normalized_tables = {
        "GENERIC_EVENT_DATA",
        "GENERIC_EVENT_TYPE_FIELDS",
        "StringIds",
    }
    if not normalized_tables.issubset(table_names):
        return None

    row = database.execute(AVERAGE_POWER_NORMALIZED_QUERY).fetchone()
    return None if row is None else row[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta-t-inv",
        type=int,
        nargs="+",
        default=[900,1020,1200,1500,1800,2400,900,1020,1200,1500,1800,2400,900,1020,1200,1500,1800,2400,900,1020,1200,1500,1800,2400,900,1020,1200,1500,1800,2400,900,1020,1200,1500,1800,2400,900,1020,1200,1500,1800,2400,900,1020,1200,1500,1800,2400,900,1020,1200,1500,1800,2400,900,1020,1200,1500,1800,2400],
        metavar="VALUE",
        help="deltaTInv values to profile.",
    )
    parser.add_argument(
        "--youngs-modulus",
        type=float,
        nargs="+",
        metavar="PA",
        default=[3335000
,2912700
,2597000
,2346900
,2231600
,2127600
,5774000
,4832100
,4157300
,3661000
,3432800
,3229400
,9280500
,7275500
,5983400
,5103000
,4714100
,4380300
,14435000
,10396000
,8095700
,6663700
,6066600
,5557400
,39824000
,20314000
,13561000
,10265000
,9023500
,8029500
,2.95E+09
,74942000
,27837000
,17264000
,14248000
,12055000
,8.00E+09
,1.23E+09
,47231000
,23573000
,18391000
,15006000
,4.76E+09
,5.09E+09
,164580000
,37353000
,26049000
,19846000
,2.88E+10
,1.08E+10
,3.58E+09
,51841000
,32435000
,23429000
,9.88E+09
,7.60E+09
,1.85E+10
,91345000
,44826000
,29375000
],
        help=(
            "Young's modulus in Pa for each --delta-t-inv, in the same order. "
            "If omitted, the youngsModulus value from --base-config is used for all runs."
        ),
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=10,
        help="NVML collection duration for each run (default: 10).",
    )
    parser.add_argument(
        "--nvml-interval-ms",
        type=int,
        default=100,
        help="NVML sampling interval in milliseconds (default: 100).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated measurements per deltaTInv (default: 1).",
    )
    parser.add_argument(
        "--gpu-devices",
        default="0",
        help="NVML GPU IDs, for example '0' or '0,1' (default: 0).",
    )
    parser.add_argument(
        "--nsys",
        type=Path,
        help="Path to nsys.exe. The newest installed Nsight Systems is used if omitted.",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        help="Path to riddfmb3d.exe (default: build/bin/riddfmb3d.exe).",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        help=(
            "Base riddfmb3d config "
            "(default: examples/bunny_ymax.config.txt)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Parent directory for reports (default: measurements/riddfmb3d-nvml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without creating files or launching profiling.",
    )
    return parser.parse_args()


def version_key(path: Path) -> tuple[int, ...]:
    match = re.fullmatch(r"Nsight Systems (\d+(?:\.\d+)*)", path.name)
    if match is None:
        return ()
    return tuple(int(component) for component in match.group(1).split("."))


def find_nsys() -> Path | None:
    install_root = Path(r"C:\Program Files\NVIDIA Corporation")
    if not install_root.is_dir():
        return None

    candidates = []
    for directory in install_root.glob("Nsight Systems *"):
        candidate = directory / "target-windows-x64" / "nsys.exe"
        if candidate.is_file() and version_key(directory):
            candidates.append(candidate)
    return max(candidates, key=lambda path: version_key(path.parent.parent), default=None)


def validate_positive(values: Iterable[int], name: str) -> None:
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")


def validate_finite_positive(values: Iterable[float], name: str) -> None:
    for value in values:
        if value <= 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and positive, got {value}.")


def config_float_value(config_path: Path, key: str) -> float:
    matches = []
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^#\s]+)")
    for line in config_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            matches.append(match.group(1))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {key} entry in {config_path}, found {len(matches)}."
        )
    try:
        value = float(matches[0])
    except ValueError as error:
        raise ValueError(f"Invalid {key} value in {config_path}: {matches[0]!r}.") from error
    validate_finite_positive([value], key)
    return value


def value_file_token(value: float) -> str:
    """Return a decimal-point-free, readable token for report filenames."""
    return format(value, ".9g").replace(".", "p").replace("+", "")


def write_derived_config(
    base_config: Path, destination: Path, delta_t_inv: int, youngs_modulus: float
) -> None:
    lines = base_config.read_text(encoding="utf-8").splitlines()
    replacement_counts = {"deltaTInv": 0, "youngsModulus": 0}
    updated_lines = []
    for line in lines:
        if re.match(r"^\s*deltaTInv\s*=", line):
            updated_lines.append(f"deltaTInv = {delta_t_inv}")
            replacement_counts["deltaTInv"] += 1
        elif re.match(r"^\s*youngsModulus\s*=", line):
            updated_lines.append(f"youngsModulus = {format(youngs_modulus, '.9g')}")
            replacement_counts["youngsModulus"] += 1
        else:
            updated_lines.append(line)

    for key, count in replacement_counts.items():
        if count != 1:
            raise RuntimeError(
                f"Expected exactly one {key} entry in {base_config}, found {count}."
            )
    destination.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    fieldnames = [
        "delta_t_inv",
        "delta_t",
        "youngs_modulus",
        "repeat",
        "duration_seconds",
        "nvml_interval_ms",
        "gpu_devices",
        "started_at",
        "finished_at",
        "return_code",
        "export_return_code",
        "config_path",
        "report_path",
        "database_path",
        "average_power_w",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def export_sqlite_and_read_average_power(
    nsys_path: Path,
    report_path: Path,
    database_path: Path,
    working_directory: Path,
) -> tuple[int, float | None]:
    export_command = [
        str(nsys_path),
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        f"--output={database_path}",
        str(report_path),
    ]
    last_error = "unknown error"
    last_return_code = 0
    for _ in range(SQLITE_EXPORT_ATTEMPTS):
        completed = subprocess.run(export_command, cwd=working_directory, check=False)
        last_return_code = completed.returncode
        if completed.returncode == 0:
            try:
                # Open read-only so a transient/empty export cannot create or
                # modify a database file before the next retry.
                database_uri = f"{database_path.resolve().as_uri()}?mode=ro"
                with sqlite3.connect(database_uri, uri=True) as database:
                    average_power = read_average_power(database)
                return completed.returncode, average_power
            except (OSError, sqlite3.Error) as error:
                last_error = str(error)
        else:
            last_error = f"nsys export exited with code {completed.returncode}"
        time.sleep(SQLITE_EXPORT_RETRY_SECONDS)

    raise RuntimeError(
        f"Could not export a queryable SQLite database from {report_path} after "
        f"{SQLITE_EXPORT_ATTEMPTS} attempts (last exit code {last_return_code}: "
        f"{last_error})."
    )


def main() -> int:
    args = parse_args()
    validate_positive(args.delta_t_inv, "deltaTInv")
    if args.youngs_modulus is not None:
        validate_finite_positive(args.youngs_modulus, "youngsModulus")
        if len(args.youngs_modulus) != len(args.delta_t_inv):
            raise ValueError(
                "--youngs-modulus must provide exactly one value for each "
                "--delta-t-inv value."
            )
    validate_positive(
        [args.duration_seconds, args.nvml_interval_ms, args.repeats], "measurement parameter"
    )

    repository_root = Path(__file__).resolve().parent.parent
    nsys_path = args.nsys or find_nsys()
    executable_path = args.executable or repository_root / "build" / "bin" / "riddfmb3d.exe"
    base_config_path = (
        args.base_config
        or repository_root / "examples" / "bunny_ymax.config.txt"
    )
    output_directory = (
        args.output_dir or repository_root / "measurements" / "riddfmb3d-nvml"
    )

    for label, path in (
        ("Nsight Systems CLI", nsys_path),
        ("riddfmb3d executable", executable_path),
        ("base config", base_config_path),
    ):
        if path is None or not path.is_file():
            raise FileNotFoundError(f"{label} was not found: {path}")

    youngs_moduli = args.youngs_modulus
    if youngs_moduli is None:
        youngs_moduli = [config_float_value(base_config_path, "youngsModulus")] * len(
            args.delta_t_inv
        )

    run_directory = output_directory / f"run-{datetime.now():%Y%m%d-%H%M%S}"
    nvml_plugin = (
        f"nvml_metrics,--gpu-devices={args.gpu_devices},"
        f"--interval={args.nvml_interval_ms}"
    )

    planned_runs = [
        (delta_t_inv, youngs_modulus, repeat)
        for delta_t_inv, youngs_modulus in zip(args.delta_t_inv, youngs_moduli)
        for repeat in range(1, args.repeats + 1)
    ]

    if args.dry_run:
        for delta_t_inv, youngs_modulus, repeat in planned_runs:
            run_name = (
                f"riddfmb3d_deltaTInv_{delta_t_inv}_"
                f"youngsModulus_{value_file_token(youngs_modulus)}_run_{repeat}"
            )
            report_base = run_directory / run_name
            report_path = report_base.with_suffix(".nsys-rep")
            database_path = report_base.with_suffix(".sqlite")
            command = [
                str(nsys_path),
                "profile",
                f"--output={report_base}",
                "--trace=none",
                f"--enable={nvml_plugin}",
                f"--duration={args.duration_seconds}",
                "--kill=true",
                str(executable_path),
                "--config",
                str(run_directory / f"{run_name}.config.txt"),
            ]
            print(subprocess.list2cmdline(command))
            export_command = [
                str(nsys_path),
                "export",
                "--type=sqlite",
                "--force-overwrite=true",
                f"--output={database_path}",
                str(report_path),
            ]
            print(subprocess.list2cmdline(export_command))
            print(AVERAGE_POWER_JSON_QUERY.strip())
            print("-- Falls back to the normalized schema when the JSON column is empty.")
            print(AVERAGE_POWER_NORMALIZED_QUERY.strip())
        return 0

    run_directory.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    manifest_path = run_directory / "manifest.csv"

    try:
        for delta_t_inv, youngs_modulus, repeat in planned_runs:
            run_name = (
                f"riddfmb3d_deltaTInv_{delta_t_inv}_"
                f"youngsModulus_{value_file_token(youngs_modulus)}_run_{repeat}"
            )
            config_path = run_directory / f"{run_name}.config.txt"
            report_base = run_directory / run_name
            report_path = report_base.with_suffix(".nsys-rep")
            database_path = report_base.with_suffix(".sqlite")
            write_derived_config(base_config_path, config_path, delta_t_inv, youngs_modulus)

            command = [
                str(nsys_path),
                "profile",
                f"--output={report_base}",
                "--trace=none",
                f"--enable={nvml_plugin}",
                f"--duration={args.duration_seconds}",
                "--kill=true",
                str(executable_path),
                "--config",
                str(config_path),
            ]

            print(
                f"Profiling deltaTInv={delta_t_inv}, "
                f"youngsModulus={format(youngs_modulus, '.9g')}, "
                f"run {repeat}/{args.repeats} ..."
            )
            started_at = datetime.now()
            completed = subprocess.run(command, cwd=repository_root, check=False)
            finished_at = datetime.now()
            record: dict[str, object] = {
                "delta_t_inv": delta_t_inv,
                "delta_t": 1.0 / delta_t_inv,
                "youngs_modulus": youngs_modulus,
                "repeat": repeat,
                "duration_seconds": args.duration_seconds,
                "nvml_interval_ms": args.nvml_interval_ms,
                "gpu_devices": args.gpu_devices,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "return_code": completed.returncode,
                "export_return_code": None,
                "config_path": config_path,
                "report_path": report_path,
                "database_path": database_path,
                "average_power_w": None,
            }
            records.append(record)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Nsight Systems failed for deltaTInv={delta_t_inv} "
                    f"with exit code {completed.returncode}."
                )

            print(f"Exporting {report_path.name} to SQLite ...")
            export_return_code, average_power_w = export_sqlite_and_read_average_power(
                nsys_path, report_path, database_path, repository_root
            )
            record["export_return_code"] = export_return_code
            record["average_power_w"] = average_power_w
            if export_return_code != 0:
                raise RuntimeError(
                    f"Nsight Systems could not export {report_path} "
                    f"to SQLite (exit code {export_return_code})."
                )
            print(f"Average power: {average_power_w} W")
    finally:
        write_manifest(manifest_path, records)
        print(f"Measurement manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
