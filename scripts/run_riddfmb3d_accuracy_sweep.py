#!/usr/bin/env python3
"""批量运行 riddfmb3d，保存逐帧网格并与 FEM 结果比较精度。

用法：
    python scripts/run_riddfmb3d_accuracy_sweep.py --config scripts/config.json

每一个 case 会生成：
  <outputRoot>/<case>/mesh/RIDdfmb_0.vtk ... RIDdfmb_<N>.vtk
  <outputRoot>/<case>/accuracy_by_frame.csv
  <outputRoot>/<case>/accuracy_summary.csv
  <outputRoot>/<case>/rid.log
  <outputRoot>/<case>/comparison.log

全部 case 完成后，会在 <outputRoot>/batch_accuracy_summary.csv 写入可直接
用 Excel 打开的汇总表。脚本不会覆盖已有 case；只有显式传入 --overwrite
才会删除并重新生成对应目录。

RID 程序需包含以下批处理选项（本仓库 riddfmb3d 已实现）：
  --vtk-output-dir <dir>
  --vtk-frame-limit <positive integer>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
COMPARE_SCRIPT = SCRIPT_PATH.with_name("compare_vtk_mesh_accuracy.py")
OUTPUT_FPS = 60.0
CASE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ConfigError(ValueError):
    """配置不完整或数值非法。"""


class CaseError(RuntimeError):
    """单组 RID 运行或精度比较失败。"""


@dataclass(frozen=True)
class Case:
    name: str
    delta_t_inv: float
    youngs_modulus: float
    fem_reference_dir: Path


@dataclass(frozen=True)
class SweepConfig:
    executable: Path
    working_directory: Path
    base_config: Path
    timeout_seconds: float
    frame_count: int
    output_root: Path
    simulation: Mapping[str, Any]
    reference_pattern: str
    test_pattern: str
    rest_frame: int
    comparison_frames: str
    key_displacement_threshold: float
    cases: tuple[Case, ...]


def required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise ConfigError(f"{context}.{key} 必填，当前为空。")
    return value


def as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} 必须是 JSON 对象。")
    return value


def finite_positive(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{context} 必须是正数。")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context} 必须是正数。") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ConfigError(f"{context} 必须是有限正数。")
    return parsed


def nonnegative_integer(value: Any, context: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context} 必须是整数。")
    if value < 0 or (positive and value == 0):
        relation = "正整数" if positive else "非负整数"
        raise ConfigError(f"{context} 必须是{relation}。")
    return value


def resolve_project_path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} 必须是非空路径字符串。")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as config_file:
            content = json.load(config_file)
    except FileNotFoundError as exc:
        raise ConfigError(f"找不到配置文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"配置文件不是有效 JSON：第 {exc.lineno} 行、第 {exc.colno} 列：{exc.msg}"
        ) from exc
    return as_mapping(content, "根对象")


def parse_cases(value: Any) -> tuple[Case, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError("cases 必须是至少包含一个对象的数组。")

    parsed_cases: list[Case] = []
    names: set[str] = set()
    for index, raw_case in enumerate(value):
        context = f"cases[{index}]"
        case = as_mapping(raw_case, context)
        name = required(case, "name", context)
        if not isinstance(name, str) or not CASE_NAME_RE.fullmatch(name):
            raise ConfigError(
                f"{context}.name 必须仅使用英文字母、数字、下划线或连字符，且以字母或数字开头。"
            )
        if name in names:
            raise ConfigError(f"{context}.name 与前面的 case 重名：{name}")
        names.add(name)

        delta_t_inv = finite_positive(required(case, "deltaTInv", context), f"{context}.deltaTInv")
        ratio = delta_t_inv / OUTPUT_FPS
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-6):
            raise ConfigError(
                f"{context}.deltaTInv={delta_t_inv:g} 不是 {OUTPUT_FPS:g} 的整数倍。"
                "当前 RID 每个输出帧固定为 1/60 秒；请使用 60 的整数倍，"
                "避免与 FEM 同帧的物理时刻错位。"
            )

        parsed_cases.append(Case(
            name=name,
            delta_t_inv=delta_t_inv,
            youngs_modulus=finite_positive(
                required(case, "youngsModulus", context), f"{context}.youngsModulus"
            ),
            fem_reference_dir=resolve_project_path(
                required(case, "femReferenceDir", context), f"{context}.femReferenceDir"
            ),
        ))
    return tuple(parsed_cases)


def parse_csv_positive_integer(value: Any, context: str) -> int:
    if value is None:
        raise ConfigError(f"{context} 不能为空。")
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context} 必须是正整数。") from exc
    if not math.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ConfigError(f"{context} 必须是正整数。")
    return int(numeric)


def parse_cases_from_csv(value: Any) -> tuple[Case, ...]:
    source = as_mapping(value, "caseSource")
    csv_path = resolve_project_path(required(source, "path", "caseSource"), "caseSource.path")
    case_column = str(required(source, "caseColumn", "caseSource"))
    substeps_column = str(required(source, "substepsColumn", "caseSource"))
    modulus_column = str(required(source, "youngsModulusColumn", "caseSource"))
    dir_column = str(required(source, "dirColumn", "caseSource"))
    fem_reference_dir_prefix = str(
        required(source, "femReferenceDirPrefix", "caseSource")
    )
    if not csv_path.is_file():
        raise ConfigError(f"caseSource CSV 不存在：{csv_path}")

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = set(reader.fieldnames or [])
            required_columns = {case_column, substeps_column, modulus_column, dir_column}
            missing_columns = sorted(required_columns - fieldnames)
            if missing_columns:
                raise ConfigError(
                    f"caseSource CSV 缺少列：{', '.join(missing_columns)}；"
                    f"实际列：{', '.join(reader.fieldnames or [])}"
                )

            parsed_cases: list[Case] = []
            names: set[str] = set()
            for row_number, row in enumerate(reader, start=2):
                if not any(value and value.strip() for value in row.values()):
                    continue
                context = f"caseSource CSV 第 {row_number} 行"
                name = (row.get(case_column) or "").strip()
                if not CASE_NAME_RE.fullmatch(name):
                    raise ConfigError(
                        f"{context} 的 {case_column!r} 必须仅使用英文字母、数字、下划线或连字符。"
                    )
                if name in names:
                    raise ConfigError(f"{context} 的 case 名称重复：{name}")
                names.add(name)

                substeps = parse_csv_positive_integer(row.get(substeps_column), context)
                directory_number = parse_csv_positive_integer(
                    row.get(dir_column), f"{context} 的 {dir_column!r}"
                )
                parsed_cases.append(Case(
                    name=name,
                    delta_t_inv=substeps * OUTPUT_FPS,
                    youngs_modulus=finite_positive(
                        row.get(modulus_column), f"{context} 的 {modulus_column!r}"
                    ),
                    fem_reference_dir=resolve_project_path(
                        f"{fem_reference_dir_prefix}{directory_number}",
                        f"{context} 的 FEM 参考目录",
                    ),
                ))
    except OSError as exc:
        raise ConfigError(f"无法读取 caseSource CSV {csv_path}：{exc}") from exc

    if not parsed_cases:
        raise ConfigError(f"caseSource CSV 没有有效数据行：{csv_path}")
    return tuple(parsed_cases)


def parse_configuration(path: Path) -> SweepConfig:
    root = parse_json(path)
    program = as_mapping(required(root, "program", "根对象"), "program")
    export = as_mapping(required(root, "export", "根对象"), "export")
    simulation = as_mapping(required(root, "simulation", "根对象"), "simulation")
    comparison = as_mapping(required(root, "comparison", "根对象"), "comparison")

    timeout_seconds = finite_positive(
        required(program, "timeoutSeconds", "program"), "program.timeoutSeconds"
    )
    frame_count = nonnegative_integer(
        required(export, "frameCount", "export"), "export.frameCount", positive=True
    )
    rest_frame = nonnegative_integer(
        required(comparison, "restFrame", "comparison"), "comparison.restFrame"
    )
    if rest_frame > frame_count:
        raise ConfigError("comparison.restFrame 不能大于 export.frameCount。")

    comparison_frames = required(comparison, "frames", "comparison")
    if not isinstance(comparison_frames, str) or not comparison_frames.strip():
        raise ConfigError("comparison.frames 必须是非空字符串，例如 \"1:250\"。")

    return SweepConfig(
        executable=resolve_project_path(required(program, "executable", "program"), "program.executable"),
        working_directory=resolve_project_path(
            required(program, "workingDirectory", "program"), "program.workingDirectory"
        ),
        base_config=resolve_project_path(required(program, "baseConfig", "program"), "program.baseConfig"),
        timeout_seconds=timeout_seconds,
        frame_count=frame_count,
        output_root=resolve_project_path(required(export, "outputRoot", "export"), "export.outputRoot"),
        simulation=simulation,
        reference_pattern=str(required(comparison, "referenceFilenamePattern", "comparison")),
        test_pattern=str(required(comparison, "testFilenamePattern", "comparison")),
        rest_frame=rest_frame,
        comparison_frames=comparison_frames.strip(),
        key_displacement_threshold=finite_positive(
            required(comparison, "keyDisplacementThreshold", "comparison"),
            "comparison.keyDisplacementThreshold",
        ),
        cases=(
            parse_cases_from_csv(root["caseSource"])
            if root.get("caseSource") is not None
            else parse_cases(required(root, "cases", "根对象"))
        ),
    )


def verify_inputs(config: SweepConfig) -> None:
    if not config.executable.is_file():
        raise ConfigError(f"RID 可执行程序不存在：{config.executable}")
    if not config.working_directory.is_dir():
        raise ConfigError(f"RID 工作目录不存在：{config.working_directory}")
    if not config.base_config.is_file():
        raise ConfigError(f"RID 基础配置不存在：{config.base_config}")
    if not COMPARE_SCRIPT.is_file():
        raise ConfigError(f"找不到精度比较脚本：{COMPARE_SCRIPT}")

    try:
        from compare_vtk_mesh_accuracy import find_frames, parse_frame_selection
    except ImportError as exc:  # pragma: no cover - only guards broken checkout
        raise ConfigError(f"无法加载精度比较工具：{exc}") from exc

    try:
        selected_frames = parse_frame_selection(config.comparison_frames)
    except ValueError as exc:
        raise ConfigError(f"comparison.frames 不合法：{exc}") from exc
    if selected_frames is not None:
        unavailable_test_frames = sorted(
            frame for frame in selected_frames if frame > config.frame_count
        )
        if unavailable_test_frames:
            raise ConfigError(
                "comparison.frames 请求了未导出的 RID 帧："
                + ", ".join(map(str, unavailable_test_frames[:20]))
                + (" …" if len(unavailable_test_frames) > 20 else "")
            )

    for case in config.cases:
        if not case.fem_reference_dir.is_dir():
            raise ConfigError(f"case {case.name} 的 FEM 目录不存在：{case.fem_reference_dir}")
        try:
            reference_frames = find_frames(case.fem_reference_dir, config.reference_pattern)
        except ValueError as exc:
            raise ConfigError(f"case {case.name} 的 FEM 文件名模式无效：{exc}") from exc
        if config.rest_frame not in reference_frames:
            raise ConfigError(
                f"case {case.name} 的 FEM 目录缺少静止帧 {config.rest_frame}："
                f"{case.fem_reference_dir / config.reference_pattern.replace('{frame}', str(config.rest_frame))}"
            )
        if selected_frames is not None:
            missing_reference_frames = sorted(selected_frames - reference_frames.keys())
            if missing_reference_frames:
                raise ConfigError(
                    f"case {case.name} 的 FEM 目录缺少 comparison.frames 所需帧："
                    + ", ".join(map(str, missing_reference_frames[:20]))
                    + (" …" if len(missing_reference_frames) > 20 else "")
                )


def value_to_config_text(value: Any, context: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ConfigError(f"simulation.{context} 必须是有限数值。")
        return format(numeric, ".9g")
    if isinstance(value, str):
        if not value:
            raise ConfigError(f"simulation.{context} 不能为空。")
        return value
    if isinstance(value, list):
        if len(value) != 4:
            raise ConfigError(f"simulation.{context} 必须是包含 4 个数值的数组。")
        return " ".join(value_to_config_text(component, context) for component in value)
    raise ConfigError(f"simulation.{context} 的类型不受支持。")


def make_case_config(base_config: Path, simulation: Mapping[str, Any], case: Case) -> str:
    expected_keys = (
        "modelPath", "numSolverIterations", "density", "damping", "gravity",
        "poissonRatio", "fixedPlaneEnabled", "fixedSelector",
        "fixedRelativeThickness", "fixedPlaneNormal", "fixedPlaneOffset",
        "fixedPlaneTolerance", "groundEnabled", "groundHeight",
        "groundRestitution",
    )
    missing = [key for key in expected_keys if key not in simulation]
    if missing:
        raise ConfigError("simulation 缺少字段：" + ", ".join(missing))

    try:
        base_text = base_config.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"无法读取基础配置 {base_config}：{exc}") from exc

    lines = [
        "",
        "# Generated by scripts/run_riddfmb3d_accuracy_sweep.py. Do not edit.",
        f"# case = {case.name}",
    ]
    for key in expected_keys:
        lines.append(f"{key} = {value_to_config_text(simulation[key], key)}")
    lines.extend([
        f"deltaTInv = {format(case.delta_t_inv, '.9g')}",
        f"youngsModulus = {format(case.youngs_modulus, '.9g')}",
        "",
    ])
    return base_text.rstrip() + "\n" + "\n".join(lines)


def case_directory(config: SweepConfig, case: Case) -> Path:
    path = (config.output_root / case.name).resolve()
    try:
        path.relative_to(config.output_root.resolve())
    except ValueError as exc:  # pragma: no cover - name parser already prevents this
        raise ConfigError(f"case 输出目录越出 outputRoot：{path}") from exc
    return path


def remove_case_directory(path: Path, output_root: Path) -> None:
    try:
        path.relative_to(output_root.resolve())
    except ValueError as exc:  # defensive check before recursive deletion
        raise CaseError(f"拒绝删除 outputRoot 外的目录：{path}") from exc
    shutil.rmtree(path)


def run_command(command: Sequence[str], *, cwd: Path, log_path: Path, timeout: float) -> None:
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        log_file.write("Command:\n")
        log_file.write(" ".join(command) + "\n\n")
        log_file.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise CaseError(
                f"RID 超过 {timeout:g} 秒仍未结束；详见日志 {log_path}。"
            ) from exc
    if completed.returncode != 0:
        raise CaseError(
            f"命令退出码为 {completed.returncode}；详见日志 {log_path}。"
        )


def verify_mesh_export(mesh_directory: Path, frame_count: int) -> None:
    missing_or_empty: list[str] = []
    for frame in range(frame_count + 1):
        path = mesh_directory / f"RIDdfmb_{frame}.vtk"
        if not path.is_file() or path.stat().st_size == 0:
            missing_or_empty.append(str(frame))
    if missing_or_empty:
        preview = ", ".join(missing_or_empty[:20])
        suffix = " …" if len(missing_or_empty) > 20 else ""
        raise CaseError(
            f"RID 未完整导出 0..{frame_count} 帧；缺失或空文件的帧：{preview}{suffix}。"
        )


def read_summary(path: Path) -> dict[str, str]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as summary_file:
            rows = csv.DictReader(summary_file)
            return {row["metric"]: row["value"] for row in rows if row.get("metric")}
    except (OSError, KeyError) as exc:
        raise CaseError(f"无法读取精度汇总表 {path}：{exc}") from exc


SUMMARY_COLUMNS = (
    "candidate_common_frames",
    "compared_frames",
    "valid_point_pairs",
    "key_displacement_threshold",
    "mean_absolute_error",
    "rms_absolute_error",
    "max_absolute_error",
    "mean_relative_error",
    "rms_relative_error",
    "max_relative_error",
    "key_mean_relative_error",
    "key_rms_relative_error",
    "key_max_relative_error",
    "relative_l2_error",
)


def write_batch_summary(path: Path, rows: Sequence[dict[str, str]]) -> None:
    fieldnames = [
        "case", "deltaTInv", "deltaT", "youngsModulus", "meshDirectory",
        "femReferenceDirectory",
        *SUMMARY_COLUMNS,
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_case(config: SweepConfig, case: Case, *, overwrite: bool, dry_run: bool) -> dict[str, str] | None:
    output_directory = case_directory(config, case)
    mesh_directory = output_directory / "mesh"
    generated_config = output_directory / "riddfmb3d_config.txt"
    rid_log = output_directory / "rid.log"
    frame_csv = output_directory / "accuracy_by_frame.csv"
    summary_csv = output_directory / "accuracy_summary.csv"
    comparison_log = output_directory / "comparison.log"

    rid_command = [
        str(config.executable), "--config", str(generated_config), "--write-vtk",
        "--vtk-output-dir", str(mesh_directory), "--vtk-frame-limit", str(config.frame_count),
        "--console",
    ]
    comparison_command = [
        sys.executable, str(COMPARE_SCRIPT),
        "--reference-dir", str(case.fem_reference_dir),
        "--test-dir", str(mesh_directory),
        "--reference-pattern", config.reference_pattern,
        "--test-pattern", config.test_pattern,
        "--frames", config.comparison_frames,
        "--rest-frame", str(config.rest_frame),
        "--key-displacement-threshold", format(config.key_displacement_threshold, ".9g"),
        "--csv", str(frame_csv),
        "--summary-csv", str(summary_csv),
    ]

    print(f"\n=== case: {case.name} ===")
    print("RID:", " ".join(rid_command))
    print("比较:", " ".join(comparison_command))
    if dry_run:
        return None

    if output_directory.exists():
        if not overwrite:
            raise CaseError(
                f"输出目录已存在：{output_directory}。如确认要删除该 case 的旧结果并重跑，请加 --overwrite。"
            )
        remove_case_directory(output_directory, config.output_root)
    output_directory.mkdir(parents=True, exist_ok=False)
    generated_config.write_text(
        make_case_config(config.base_config, config.simulation, case), encoding="utf-8"
    )

    run_command(
        rid_command, cwd=config.working_directory, log_path=rid_log,
        timeout=config.timeout_seconds,
    )
    verify_mesh_export(mesh_directory, config.frame_count)
    run_command(
        comparison_command, cwd=PROJECT_ROOT, log_path=comparison_log,
        timeout=config.timeout_seconds,
    )
    summary = read_summary(summary_csv)
    return {
        "case": case.name,
        "deltaTInv": format(case.delta_t_inv, ".9g"),
        "deltaT": format(1.0 / case.delta_t_inv, ".9g"),
        "youngsModulus": format(case.youngs_modulus, ".9g"),
        "meshDirectory": str(mesh_directory),
        "femReferenceDirectory": str(case.fem_reference_dir),
        **{column: summary.get(column, "") for column in SUMMARY_COLUMNS},
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量运行 riddfmb3d、保存每帧 VTK 网格并输出 FEM 精度对比总表。"
    )
    parser.add_argument(
        "--config", type=Path, default=SCRIPT_PATH.with_name("config.json"),
        help="批处理 JSON 配置；默认 scripts/config.json",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="删除 outputRoot 中同名 case 的旧结果后重新运行。",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只校验配置并打印命令，不启动 RID、不写文件。",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    config_path = args.config.resolve()
    try:
        config = parse_configuration(config_path)
        verify_inputs(config)
        if not args.dry_run:
            config.output_root.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, str]] = []
        for case in config.cases:
            result = run_case(config, case, overwrite=args.overwrite, dry_run=args.dry_run)
            if result is not None:
                rows.append(result)

        if args.dry_run:
            print("\n配置校验完成；未启动 RID，也未写入文件。")
            return 0

        batch_summary = config.output_root / "batch_accuracy_summary.csv"
        write_batch_summary(batch_summary, rows)
        print(f"\n全部 case 完成。批量精度汇总表：{batch_summary}")
        return 0
    except (ConfigError, CaseError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
