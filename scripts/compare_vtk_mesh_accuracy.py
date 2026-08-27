#!/usr/bin/env python3
"""比较两个 ASCII Legacy VTK 网格序列的相对位移误差。

默认比较：
  FEM:  C:\\zjucad\\Stiff-GIPC\\Output\\tetmesh_frame_<frame>.vtk
  GPU:  C:\\zjucad\\Vulkan\\output\\RIDdfmb_<frame>.vtk

以第 0 帧（可通过 --rest-frame 修改）的未变形网格为基准，先计算每个点的
位移，再按 VTK 中的顶点编号配对：

    u_i = x_i(frame) - x_i(rest_frame)
    d_i = ||u_test_i - u_ref_i||_2

脚本输出四类统计量：
  1. 相对位移的绝对误差 d_i；
  2. 相对位移的逐点相对误差 d_i / ||u_ref_i||；
  3. 关键点逐点相对误差，仅统计 ||u_ref_i|| >= key_displacement_threshold
     的点；小于阈值的点视为非关键点，不计入该项；
  4. 整体相对 L2 误差 sqrt(sum_i d_i^2 / sum_i ||u_ref_i||^2)。

第 2 项对参考位移为零的点没有定义，这些点会被明确计数并从相对误差平均值
中排除，而不是设置任意分母下限。第 4 项若 FEM 在一个帧内完全静止，同样会
标记为未定义。默认用 FEM 作为参考解。

注意：该脚本假设两套网格的顶点编号一一对应。若网格分辨率或顶点排序不
相同，不能直接作逐点误差；脚本会跳过顶点数不同的帧并在报告中列出原因。

示例：
  python scripts/compare_vtk_mesh_accuracy.py
  python scripts/compare_vtk_mesh_accuracy.py --frames 104:221
  python scripts/compare_vtk_mesh_accuracy.py --key-displacement-threshold 0.2
  python scripts/compare_vtk_mesh_accuracy.py --frames 1:250 --key-displacement-threshold 0.05
  python scripts/compare_vtk_mesh_accuracy.py --csv results/error.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_REFERENCE_DIR = Path(r"C:\zjucad\Stiff-GIPC\Output")
DEFAULT_TEST_DIR = Path(r"C:\zjucad\Vulkan\output")
DEFAULT_REFERENCE_PATTERN = "tetmesh_frame_{frame}.vtk"
DEFAULT_TEST_PATTERN = "RIDdfmb_{frame}.vtk"


@dataclass(frozen=True)
class FrameResult:
    """一个成功比较帧的统计量。"""

    frame: int
    point_count: int
    valid_count: int
    invalid_count: int
    zero_reference_displacement_count: int
    relative_error_count: int
    key_point_count: int
    mean_absolute_error: float | None
    rms_absolute_error: float | None
    max_absolute_error: float | None
    mean_relative_error: float | None
    rms_relative_error: float | None
    max_relative_error: float | None
    key_mean_relative_error: float | None
    key_rms_relative_error: float | None
    key_max_relative_error: float | None
    reference_rms: float | None
    relative_l2_error: float | None


def parse_ascii_vtk_points(path: Path) -> list[tuple[float, float, float]]:
    """读取 Legacy ASCII VTK 文件的 POINTS 坐标，不依赖 vtk/numpy。"""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as vtk_file:
            lines = iter(vtk_file)
            for line in lines:
                tokens = line.split()
                if len(tokens) >= 3 and tokens[0].upper() == "POINTS":
                    point_count = int(tokens[1])
                    coordinates: list[float] = []
                    needed = 3 * point_count

                    while len(coordinates) < needed:
                        try:
                            coordinate_line = next(lines)
                        except StopIteration as exc:
                            raise ValueError(
                                f"POINTS 段不完整：声明 {point_count} 个点，"
                                f"仅读取到 {len(coordinates) // 3} 个点"
                            ) from exc
                        try:
                            coordinates.extend(float(value) for value in coordinate_line.split())
                        except ValueError as exc:
                            raise ValueError(
                                f"POINTS 段含有非数值内容：{coordinate_line.strip()!r}"
                            ) from exc

                    if len(coordinates) != needed:
                        raise ValueError(
                            f"POINTS 段数据量异常：需要 {needed} 个数，读取到 "
                            f"{len(coordinates)} 个数"
                        )
                    return [
                        (coordinates[index], coordinates[index + 1], coordinates[index + 2])
                        for index in range(0, needed, 3)
                    ]
    except OSError as exc:
        raise ValueError(f"无法读取文件：{exc}") from exc

    raise ValueError("未找到 POINTS 段；该脚本只支持 Legacy ASCII VTK 网格")


def pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """把 ``foo_{frame}.vtk`` 转为提取帧编号的正则表达式。"""
    marker = "{frame}"
    if pattern.count(marker) != 1:
        raise ValueError(f"文件名模式必须且只能包含一次 {marker!r}：{pattern!r}")
    escaped = re.escape(pattern)
    return re.compile("^" + escaped.replace(re.escape(marker), r"(?P<frame>\d+)") + "$")


def find_frames(directory: Path, pattern: str) -> dict[int, Path]:
    """按文件名模式返回 ``{frame: vtk_path}``。"""
    if not directory.is_dir():
        raise ValueError(f"目录不存在或不是目录：{directory}")

    expression = pattern_to_regex(pattern)
    frames: dict[int, Path] = {}
    for candidate in directory.iterdir():
        if not candidate.is_file():
            continue
        match = expression.fullmatch(candidate.name)
        if match is None:
            continue
        frame = int(match.group("frame"))
        if frame in frames:
            raise ValueError(f"帧 {frame} 存在重复文件：{frames[frame]} 和 {candidate}")
        frames[frame] = candidate
    return frames


def parse_frame_selection(text: str | None) -> set[int] | None:
    """解析 ``1,3,8:20``；区间两端均包含。空值表示不筛选。"""
    if text is None:
        return None

    selected: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_text, end_text = (value.strip() for value in part.split(":", 1))
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"帧区间起点不能大于终点：{part!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected:
        raise ValueError("--frames 不能为空")
    return selected


def vector_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def subtract_vectors(
    values: list[tuple[float, float, float]],
    rest_values: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """返回相对静止帧的位移。"""
    if len(values) != len(rest_values):
        raise ValueError(f"当前帧有 {len(values)} 个点，静止帧有 {len(rest_values)} 个点")
    return [
        (value[0] - rest[0], value[1] - rest[1], value[2] - rest[2])
        for value, rest in zip(values, rest_values)
    ]


def compare_frame(
    frame: int,
    reference_points: list[tuple[float, float, float]],
    test_points: list[tuple[float, float, float]],
    key_displacement_threshold: float,
) -> FrameResult:
    """计算单帧四类相对位移误差。非有限数值会从统计中排除。"""
    if len(reference_points) != len(test_points):
        raise ValueError("点数不一致")

    absolute_error_sum = 0.0
    error_square_sum = 0.0
    reference_square_sum = 0.0
    max_absolute_error = 0.0
    valid_count = 0
    invalid_count = 0
    zero_reference_displacement_count = 0
    relative_error_sum = 0.0
    relative_error_square_sum = 0.0
    max_relative_error = 0.0
    relative_error_count = 0
    key_relative_error_sum = 0.0
    key_relative_error_square_sum = 0.0
    key_max_relative_error = 0.0
    key_point_count = 0

    for reference, test in zip(reference_points, test_points):
        if not all(math.isfinite(value) for value in (*reference, *test)):
            invalid_count += 1
            continue

        difference = (test[0] - reference[0], test[1] - reference[1], test[2] - reference[2])
        absolute_error = vector_norm(difference)
        absolute_error_sum += absolute_error
        error_square_sum += absolute_error * absolute_error
        reference_norm = vector_norm(reference)
        reference_square_sum += reference_norm * reference_norm
        max_absolute_error = max(max_absolute_error, absolute_error)
        if reference_norm == 0.0:
            zero_reference_displacement_count += 1
        else:
            relative_error = absolute_error / reference_norm
            relative_error_sum += relative_error
            relative_error_square_sum += relative_error * relative_error
            max_relative_error = max(max_relative_error, relative_error)
            relative_error_count += 1
            if reference_norm >= key_displacement_threshold:
                key_relative_error_sum += relative_error
                key_relative_error_square_sum += relative_error * relative_error
                key_max_relative_error = max(key_max_relative_error, relative_error)
                key_point_count += 1
        valid_count += 1

    if valid_count == 0:
        return FrameResult(
            frame=frame,
            point_count=len(reference_points),
            valid_count=0,
            invalid_count=invalid_count,
            zero_reference_displacement_count=0,
            relative_error_count=0,
            key_point_count=0,
            mean_absolute_error=None,
            rms_absolute_error=None,
            max_absolute_error=None,
            mean_relative_error=None,
            rms_relative_error=None,
            max_relative_error=None,
            key_mean_relative_error=None,
            key_rms_relative_error=None,
            key_max_relative_error=None,
            reference_rms=None,
            relative_l2_error=None,
        )

    return FrameResult(
        frame=frame,
        point_count=len(reference_points),
        valid_count=valid_count,
        invalid_count=invalid_count,
        zero_reference_displacement_count=zero_reference_displacement_count,
        relative_error_count=relative_error_count,
        key_point_count=key_point_count,
        mean_absolute_error=absolute_error_sum / valid_count,
        rms_absolute_error=math.sqrt(error_square_sum / valid_count),
        max_absolute_error=max_absolute_error,
        mean_relative_error=(relative_error_sum / relative_error_count
                             if relative_error_count else None),
        rms_relative_error=(math.sqrt(relative_error_square_sum / relative_error_count)
                            if relative_error_count else None),
        max_relative_error=(max_relative_error if relative_error_count else None),
        key_mean_relative_error=(key_relative_error_sum / key_point_count
                                 if key_point_count else None),
        key_rms_relative_error=(math.sqrt(key_relative_error_square_sum / key_point_count)
                                if key_point_count else None),
        key_max_relative_error=(key_max_relative_error if key_point_count else None),
        reference_rms=math.sqrt(reference_square_sum / valid_count),
        relative_l2_error=(
            math.sqrt(error_square_sum / reference_square_sum)
            if reference_square_sum > 0.0
            else None
        ),
    )


def write_csv(path: Path, results: Iterable[FrameResult]) -> None:
    """保存逐帧结果，便于用 Excel、Matlab 或 Python 绘图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "frame",
                "point_count",
                "valid_count",
                "invalid_count",
                "zero_reference_displacement_count",
                "relative_error_count",
                "key_point_count",
                "mean_absolute_error",
                "rms_absolute_error",
                "max_absolute_error",
                "mean_relative_error",
                "rms_relative_error",
                "max_relative_error",
                "key_mean_relative_error",
                "key_rms_relative_error",
                "key_max_relative_error",
                "reference_rms",
                "relative_l2_error",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.frame,
                    result.point_count,
                    result.valid_count,
                    result.invalid_count,
                    result.zero_reference_displacement_count,
                    result.relative_error_count,
                    result.key_point_count,
                    result.mean_absolute_error,
                    result.rms_absolute_error,
                    result.max_absolute_error,
                    result.mean_relative_error,
                    result.rms_relative_error,
                    result.max_relative_error,
                    result.key_mean_relative_error,
                    result.key_rms_relative_error,
                    result.key_max_relative_error,
                    result.reference_rms,
                    result.relative_l2_error,
                ]
            )


def write_summary_csv(path: Path, rows: Iterable[tuple[object, ...]]) -> None:
    """保存命令行最终汇总，使用 UTF-8 BOM 以便 Excel 直接打开。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value", "percent", "unit", "sample_count", "description"])
        writer.writerows(rows)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按顶点编号比较两个 Legacy ASCII VTK 网格序列的相对位移误差。"
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR,
                        help=f"参考（FEM）VTK 所在目录；默认：{DEFAULT_REFERENCE_DIR}")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR,
                        help=f"待比较 VTK 所在目录；默认：{DEFAULT_TEST_DIR}")
    parser.add_argument("--reference-pattern", default=DEFAULT_REFERENCE_PATTERN,
                        help=f"参考文件名模式；默认：{DEFAULT_REFERENCE_PATTERN}")
    parser.add_argument("--test-pattern", default=DEFAULT_TEST_PATTERN,
                        help=f"待比较文件名模式；默认：{DEFAULT_TEST_PATTERN}")
    parser.add_argument("--frames", metavar="LIST",
                        help="只比较指定帧，例如 104:221 或 1,3,8:20；默认比较所有共同帧")
    parser.add_argument("--rest-frame", type=int, default=0,
                        help="未变形参考帧编号，默认：0")
    parser.add_argument("--key-displacement-threshold", type=float, required=True,
                        help="关键点的 FEM 位移阈值（模型长度单位）；小于它的点不计入关键点误差")
    parser.add_argument("--csv", type=Path, default=Path("mesh_accuracy_report.csv"),
                        help="逐帧 CSV 报告路径；默认：mesh_accuracy_report.csv")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="最终汇总 CSV 路径；默认：在 --csv 同目录生成 <csv名称>_summary.csv",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if (not math.isfinite(args.key_displacement_threshold)
            or args.key_displacement_threshold <= 0.0):
        print("错误：--key-displacement-threshold 必须是有限正数。", file=sys.stderr)
        return 2
    summary_csv = args.summary_csv or args.csv.with_name(f"{args.csv.stem}_summary.csv")

    try:
        selected_frames = parse_frame_selection(args.frames)
        reference_frames = find_frames(args.reference_dir, args.reference_pattern)
        test_frames = find_frames(args.test_dir, args.test_pattern)
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    common_frames = sorted(reference_frames.keys() & test_frames.keys())
    if selected_frames is not None:
        common_frames = [frame for frame in common_frames if frame in selected_frames]

    if not common_frames:
        print("错误：没有可比较的共同帧。请检查目录、文件名模式和 --frames。", file=sys.stderr)
        return 2

    try:
        reference_rest_path = reference_frames[args.rest_frame]
        test_rest_path = test_frames[args.rest_frame]
    except KeyError:
        print(
            f"错误：位移比较需要两边都有静止帧 {args.rest_frame}。"
            "请先导出未运动的第 0 帧，或指定 --rest-frame。",
            file=sys.stderr,
        )
        return 2
    try:
        reference_rest_points = parse_ascii_vtk_points(reference_rest_path)
        test_rest_points = parse_ascii_vtk_points(test_rest_path)
    except ValueError as exc:
        print(f"错误：无法读取静止帧：{exc}", file=sys.stderr)
        return 2
    if len(reference_rest_points) != len(test_rest_points):
        print(
            f"错误：静止帧顶点数不同（FEM {len(reference_rest_points)}，"
            f"待比较 {len(test_rest_points)}），无法按编号配对。",
            file=sys.stderr,
        )
        return 2

    print(f"位移：u = x(frame) - x({args.rest_frame})")
    print("整体相对 L2 误差：sqrt(sum ||u_test - u_FEM||_2^2 / sum ||u_FEM||_2^2)")
    print(
        "关键点：||u_FEM||_2 >= "
        f"{args.key_displacement_threshold:.8e}（小于该值的点不计入关键点误差）"
    )
    print(f"FEM 目录：{args.reference_dir}")
    print(f"待比较目录：{args.test_dir}")
    print(f"候选共同帧：{len(common_frames)}")

    results: list[FrameResult] = []
    skipped: list[str] = []
    for frame in common_frames:
        try:
            reference_points = parse_ascii_vtk_points(reference_frames[frame])
            test_points = parse_ascii_vtk_points(test_frames[frame])
        except ValueError as exc:
            skipped.append(f"frame {frame}: 读取失败（{exc}）")
            continue

        if len(reference_points) != len(test_points):
            skipped.append(
                f"frame {frame}: 顶点数不同（FEM {len(reference_points)}，"
                f"待比较 {len(test_points)}）"
            )
            continue

        try:
            reference_values = subtract_vectors(reference_points, reference_rest_points)
            test_values = subtract_vectors(test_points, test_rest_points)
        except ValueError as exc:
            skipped.append(f"frame {frame}: 无法计算位移（{exc}）")
            continue

        results.append(compare_frame(
            frame, reference_values, test_values, args.key_displacement_threshold
        ))

    if results:
        write_csv(args.csv, results)
        valid_results = [result for result in results if result.valid_count > 0]
        total_valid = sum(result.valid_count for result in valid_results)
        total_invalid = sum(result.invalid_count for result in results)
        total_zero_reference = sum(
            result.zero_reference_displacement_count for result in valid_results
        )
        total_relative = sum(result.relative_error_count for result in valid_results)
        total_key_points = sum(result.key_point_count for result in valid_results)
        summary_rows: list[tuple[object, ...]] = [
            ("candidate_common_frames", len(common_frames), "", "frames", "", "匹配到文件名的共同帧数"),
            ("compared_frames", len(results), "", "frames", "", "成功读取且顶点数一致的帧数"),
            ("valid_point_pairs", total_valid, "", "point-frame pairs", "", "参与绝对误差和 L2 统计的点对数"),
            ("invalid_point_pairs", total_invalid, "", "point-frame pairs", "", "因 NaN/Inf 跳过的点对数"),
            ("key_displacement_threshold", args.key_displacement_threshold, "", "model length", "", "FEM 位移达到此阈值的点为关键点"),
            ("zero_reference_displacement_pairs", total_zero_reference, "", "point-frame pairs", "", "逐点相对误差未定义、未参与第 2 项的点对数"),
            ("relative_error_point_pairs", total_relative, "", "point-frame pairs", "", "参与第 2 项逐点相对误差的点对数"),
            ("key_point_pairs", total_key_points, "", "point-frame pairs", "", "参与第 3 项关键点相对误差的点对数"),
        ]

        print(f"成功读取且顶点数一致的帧：{len(results)}")
        print(f"逐帧报告：{args.csv.resolve()}")
        print(f"最终汇总表：{summary_csv.resolve()}")
        print(f"有效点对：{total_valid}；含 NaN/Inf 而跳过的点对：{total_invalid}")

        if total_valid:
            global_mean_absolute = sum(
                result.mean_absolute_error * result.valid_count
                for result in valid_results
                if result.mean_absolute_error is not None
            ) / total_valid
            global_rms_absolute = math.sqrt(sum(
                result.rms_absolute_error ** 2 * result.valid_count
                for result in valid_results
                if result.rms_absolute_error is not None
            ) / total_valid)
            global_reference_square_sum = sum(
                result.reference_rms ** 2 * result.valid_count
                for result in valid_results
                if result.reference_rms is not None
            )
            global_max_absolute = max(
                result.max_absolute_error
                for result in valid_results
                if result.max_absolute_error is not None
            )
            print(f"所有有效对应点的平均绝对误差：{global_mean_absolute:.8e}")
            print(f"所有有效对应点的 RMS 绝对误差：{global_rms_absolute:.8e}")
            print(f"所有有效对应点的最大绝对误差：{global_max_absolute:.8e}")
            summary_rows.extend([
                ("mean_absolute_error", global_mean_absolute, "", "model length", total_valid, "相对位移绝对误差的平均值"),
                ("rms_absolute_error", global_rms_absolute, "", "model length", total_valid, "相对位移绝对误差的均方根"),
                ("max_absolute_error", global_max_absolute, "", "model length", total_valid, "相对位移绝对误差的最大值"),
            ])

            print(
                "逐点相对误差的有效点对："
                f"{total_relative}；FEM 位移为零而未定义的点对：{total_zero_reference}"
            )
            if total_relative:
                global_mean_relative = sum(
                    result.mean_relative_error * result.relative_error_count
                    for result in valid_results
                    if result.mean_relative_error is not None
                ) / total_relative
                global_rms_relative = math.sqrt(sum(
                    result.rms_relative_error ** 2 * result.relative_error_count
                    for result in valid_results
                    if result.rms_relative_error is not None
                ) / total_relative)
                global_max_relative = max(
                    result.max_relative_error
                    for result in valid_results
                    if result.max_relative_error is not None
                )
                print(
                    "所有 FEM 位移非零点的平均相对误差："
                    f"{global_mean_relative:.8e} ({global_mean_relative * 100:.6f}%)"
                )
                print(
                    "所有 FEM 位移非零点的 RMS 相对误差："
                    f"{global_rms_relative:.8e} ({global_rms_relative * 100:.6f}%)"
                )
                print(
                    "所有 FEM 位移非零点的最大相对误差："
                    f"{global_max_relative:.8e} ({global_max_relative * 100:.6f}%)"
                )
                summary_rows.extend([
                    ("mean_relative_error", global_mean_relative, global_mean_relative * 100, "ratio", total_relative, "FEM 位移非零点的逐点相对误差平均值"),
                    ("rms_relative_error", global_rms_relative, global_rms_relative * 100, "ratio", total_relative, "FEM 位移非零点的逐点相对误差均方根"),
                    ("max_relative_error", global_max_relative, global_max_relative * 100, "ratio", total_relative, "FEM 位移非零点的逐点相对误差最大值"),
                ])
            else:
                print("所有 FEM 位移均为零，逐点相对误差未定义。")
                summary_rows.append(("relative_error_status", "undefined", "", "", 0, "所有 FEM 位移均为零"))

            print(
                "关键点数："
                f"{total_key_points}；非关键点数：{total_valid - total_key_points}"
            )
            if total_key_points:
                global_key_mean_relative = sum(
                    result.key_mean_relative_error * result.key_point_count
                    for result in valid_results
                    if result.key_mean_relative_error is not None
                ) / total_key_points
                global_key_rms_relative = math.sqrt(sum(
                    result.key_rms_relative_error ** 2 * result.key_point_count
                    for result in valid_results
                    if result.key_rms_relative_error is not None
                ) / total_key_points)
                global_key_max_relative = max(
                    result.key_max_relative_error
                    for result in valid_results
                    if result.key_max_relative_error is not None
                )
                print(
                    "所有关键点的平均相对误差："
                    f"{global_key_mean_relative:.8e} ({global_key_mean_relative * 100:.6f}%)"
                )
                print(
                    "所有关键点的 RMS 相对误差："
                    f"{global_key_rms_relative:.8e} ({global_key_rms_relative * 100:.6f}%)"
                )
                print(
                    "所有关键点的最大相对误差："
                    f"{global_key_max_relative:.8e} ({global_key_max_relative * 100:.6f}%)"
                )
                summary_rows.extend([
                    ("key_mean_relative_error", global_key_mean_relative, global_key_mean_relative * 100, "ratio", total_key_points, "关键点逐点相对误差平均值"),
                    ("key_rms_relative_error", global_key_rms_relative, global_key_rms_relative * 100, "ratio", total_key_points, "关键点逐点相对误差均方根"),
                    ("key_max_relative_error", global_key_max_relative, global_key_max_relative * 100, "ratio", total_key_points, "关键点逐点相对误差最大值"),
                ])
            else:
                print("没有 FEM 位移达到阈值的关键点，关键点相对误差未定义。")
                summary_rows.append(("key_relative_error_status", "undefined", "", "", 0, "没有 FEM 位移达到关键点阈值"))

            if global_reference_square_sum > 0.0:
                global_relative_l2 = math.sqrt(
                    (global_rms_absolute ** 2 * total_valid) / global_reference_square_sum
                )
                print(
                    "所有有效对应点的整体相对 L2 误差："
                    f"{global_relative_l2:.8e} ({global_relative_l2 * 100:.6f}%)"
                )
                summary_rows.append(
                    ("relative_l2_error", global_relative_l2, global_relative_l2 * 100, "ratio", total_valid, "整体相对 L2 误差")
                )
            else:
                print("所有 FEM 参考量均为零，只能报告绝对误差，整体相对 L2 误差未定义。")
                summary_rows.append(("relative_l2_error", "undefined", "", "", total_valid, "所有 FEM 参考位移均为零"))
        else:
            print("没有有效点对，无法计算误差。请检查待比较结果是否含 NaN/Inf。", file=sys.stderr)
            summary_rows.append(("comparison_status", "no_valid_point_pairs", "", "", 0, "没有有效点对，无法计算误差"))

        write_summary_csv(summary_csv, summary_rows)

    if skipped:
        print(f"跳过的帧：{len(skipped)}")
        for message in skipped:
            print(f"  - {message}")

    if not results:
        print("错误：所有候选帧都无法比较。", file=sys.stderr)
        return 1
    return 0 if any(result.valid_count > 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
