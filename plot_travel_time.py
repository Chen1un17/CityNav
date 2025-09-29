#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Any, Optional

import matplotlib.pyplot as plt


def parse_timestamp(value: Any) -> Any:
    """
    将 timestamp 尝试解析为数值或 datetime，便于 Matplotlib 绘制。
    - 数值（int/float/数字字符串）保持为数值
    - ISO 时间字符串尽量解析为 datetime
    - 其他原样返回（Matplotlib 也可按分类轴渲染）
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        v = value.strip()
        # 尝试数字
        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            pass
        # 尝试 ISO-8601
        try:
            # 兼容 "2025-09-27 17:26:45" 或 "2025-09-27T17:26:45"
            v_norm = v.replace("T", " ")
            return datetime.fromisoformat(v_norm)
        except Exception:
            return v
    return value


def read_series(jsonl_path: Path,
                x_key: str = "timestamp",
                y_key: str = "average_travel_time") -> Tuple[List[Any], List[float]]:
    """
    从 JSONL 文件读取 (timestamp, average_travel_time) 序列。
    跳过缺失或无法解析的行。
    """
    xs: List[Any] = []
    ys: List[float] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if x_key not in obj or y_key not in obj:
                continue
            x_val = parse_timestamp(obj.get(x_key))
            y_val = obj.get(y_key)
            if x_val is None or y_val is None:
                continue
            # 仅接受可绘制的 y 数值
            try:
                y_num = float(y_val)
            except (TypeError, ValueError):
                continue
            xs.append(x_val)
            ys.append(y_num)
    # 按 x 排序，保持曲线单调前进
    if len(xs) > 1:
        try:
            pairs = sorted(zip(xs, ys), key=lambda p: p[0])
            xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        except TypeError:
            # x 类型不可比较（混合类型）时，不排序
            pass
    return xs, ys


def main():
    parser = argparse.ArgumentParser(
        description="对比绘制 JSONL 中 timestamp vs average_travel_time 曲线"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="一个或多个 JSONL 文件的绝对路径，用于对比绘制"
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="各曲线标签；数量与 --files 对应。不提供则使用文件名"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图片绝对路径（如 /path/to/plot.png）。未提供则直接展示窗口"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Average Travel Time over Timestamp",
        help="图标题"
    )
    parser.add_argument(
        "--xkey",
        type=str,
        default="timestamp",
        help="横轴字段名（默认 timestamp）"
    )
    parser.add_argument(
        "--ykey",
        type=str,
        default="average_travel_time",
        help="纵轴字段名（默认 average_travel_time）"
    )
    args = parser.parse_args()

    paths = [Path(p).expanduser().resolve() for p in args.files]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {p}")

    labels: Optional[List[str]] = None
    if args.labels and len(args.labels) == len(paths):
        labels = args.labels
    else:
        labels = [p.stem for p in paths]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

    any_curve = False
    for path, label in zip(paths, labels):
        xs, ys = read_series(path, x_key=args.xkey, y_key=args.ykey)
        if not xs:
            print(f"警告：{path} 未解析出有效数据（跳过）。")
            continue
        ax.plot(xs, ys, label=label, linewidth=1.8)
        any_curve = True

    if not any_curve:
        raise RuntimeError("未能从任何文件中解析出可绘制的数据。")

    ax.set_title(args.title)
    ax.set_xlabel(args.xkey)
    ax.set_ylabel(args.ykey)
    ax.legend()
    fig.tight_layout()

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"已保存: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()