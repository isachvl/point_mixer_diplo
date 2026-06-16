from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUNS = [
    {
        "label": "19%",
        "name": "pv004 focal+lovasz from scratch",
        "log": Path(r"E:\сохры\метрики хорошие паноптика и лоссы\2026-05-05_18-50-33__scannetpp__pointmixer_panoptic_3060\log_train_2026-05-05 1852.log"),
        "best_csv": Path(r"E:\pointmixer_outputs\scannetpp_val_epoch010_19percent_class_eval\class_metrics\val_class_metrics_epoch_000.csv"),
        "checkpoint": Path(r"E:\pointmixer_outputs\PointMixerScanNetPP_panoptic_rgb_pv004_focal_lovasz_from_scratch_ep30\2026-05-05_18-50-33__scannetpp__pointmixer_panoptic_3060\epoch=010--mIoU_val=0.1855--.ckpt"),
        "best_epoch": 10,
        "best_ckpt_miou": 0.1855,
        "color": (45, 128, 115),
    },
    {
        "label": "17%",
        "name": "pv003 focal+lovasz resume",
        "log": Path(r"E:\сохры\метрики хорошие паноптика и лоссы 17проц\2026-05-10_17-47-13__scannetpp__pointmixer_panoptic_3060\log_train_2026-05-10 1751.log"),
        "best_csv": Path(r"E:\сохры\метрики хорошие паноптика и лоссы 17проц\2026-05-10_17-47-13__scannetpp__pointmixer_panoptic_3060\class_metrics\val_class_metrics_epoch_014.csv"),
        "checkpoint": Path(r"E:\pointmixer_outputs\PointMixerScanNetPP_panoptic_rgb_pv003_focal_lovasz_ep20_resume\2026-05-10_17-47-13__scannetpp__pointmixer_panoptic_3060\epoch=014--mIoU_val=0.1759--.ckpt"),
        "best_epoch": 14,
        "best_ckpt_miou": 0.1759,
        "color": (70, 112, 168),
    },
    {
        "label": "5%",
        "name": "dense005 focal+lovasz",
        "log": Path(r"E:\сохры\метрики хорошие паноптика и лоссы 5проц\2026-05-08_19-31-36__scannetpp__pointmixer_panoptic_3060\log_train_2026-05-08 2012.log"),
        "best_csv": Path(r"E:\сохры\метрики хорошие паноптика и лоссы 5проц\2026-05-08_19-31-36__scannetpp__pointmixer_panoptic_3060\class_metrics\val_class_metrics_epoch_000.csv"),
        "checkpoint": Path(r"E:\pointmixer_outputs\PointMixerScanNetPP_panoptic_rgb_dense005_focal_lovasz_ep20\2026-05-08_19-31-36__scannetpp__pointmixer_panoptic_3060\epoch=000--mIoU_val=0.0070--.ckpt"),
        "best_epoch": 0,
        "best_ckpt_miou": 0.0070,
        "color": (190, 90, 76),
    },
]


VAL_RE = re.compile(
    r"val : epoch\[(?P<epoch>\d+)\], steps\[(?P<steps>\d+)\] lr\[(?P<lr>[0-9.eE+-]+)\] \| "
    r".*?mIoU_val\[(?P<miou>[0-9.]+)\], mAcc_val\[(?P<macc>[0-9.]+)\], allAcc_val\[(?P<allacc>[0-9.]+)\]"
)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def parse_history(log_path: Path) -> list[dict]:
    history = []
    if not log_path.exists():
        return history
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = VAL_RE.search(line)
        if not match:
            continue
        item = {key: match.group(key) for key in match.groupdict()}
        history.append({
            "epoch": int(item["epoch"]),
            "steps": int(item["steps"]),
            "lr": float(item["lr"]),
            "miou": float(item["miou"]),
            "macc": float(item["macc"]),
            "allacc": float(item["allacc"]),
        })
    return history


def read_class_csv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "rank_by_iou": int(row["rank_by_iou"]),
                "class_index": int(row["class_index"]),
                "class_name": row["class_name"],
                "iou": float(row["iou"]),
                "accuracy": float(row["accuracy"]),
                "intersection": float(row["intersection"]),
                "union": float(row["union"]),
                "target": float(row["target"]),
            })
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def draw_best_bars(runs: list[dict], out: Path) -> None:
    width, height = 1100, 520
    img = Image.new("RGB", (width, height), (250, 248, 243))
    draw = ImageDraw.Draw(img)
    title = load_font(34, True)
    normal = load_font(20)
    small = load_font(17)

    draw.text((36, 28), "Сравнение лучших запусков PointMixer panoptic", fill=(31, 38, 44), font=title)
    draw.text((38, 76), "mIoU по validation, проценты", fill=(92, 99, 105), font=normal)

    left, top, chart_w, bar_h, gap = 260, 155, 720, 54, 42
    max_v = 20.0
    for tick in range(0, 21, 5):
        x = left + int(chart_w * tick / max_v)
        draw.line((x, top - 18, x, top + 3 * (bar_h + gap) - gap + 20), fill=(224, 220, 212), width=1)
        draw.text((x - 8, top - 48), str(tick), fill=(118, 124, 130), font=small)
    draw.text((left + chart_w + 20, top - 48), "%", fill=(118, 124, 130), font=small)

    for idx, run in enumerate(runs):
        y = top + idx * (bar_h + gap)
        value = run["best_miou_percent"]
        draw.text((36, y + 13), f"{run['label']}  {run['name']}", fill=(45, 52, 59), font=normal)
        bar_w = int(chart_w * value / max_v)
        draw.rounded_rectangle((left, y, left + bar_w, y + bar_h), radius=12, fill=run["color"])
        draw.text((left + bar_w + 14, y + 13), f"{value:.2f}%", fill=(31, 38, 44), font=normal)
        draw.text((left + 4, y + bar_h + 4), f"best epoch {run['best_epoch']}", fill=(105, 111, 117), font=small)

    img.save(out)


def draw_history(runs: list[dict], out: Path) -> None:
    width, height = 1450, 780
    img = Image.new("RGB", (width, height), (250, 248, 243))
    draw = ImageDraw.Draw(img)
    title = load_font(34, True)
    normal = load_font(19)
    small = load_font(16)

    draw.text((36, 28), "Динамика mIoU по эпохам", fill=(31, 38, 44), font=title)
    draw.text((38, 76), "По логам обучения; значения в логах округлены до двух знаков", fill=(92, 99, 105), font=normal)

    left, top, right, bottom = 92, 140, 80, 90
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_epoch = max(max([h["epoch"] for h in run["history"]] or [0]) for run in runs)
    max_y = 0.20

    for tick in range(0, 21, 5):
        y = top + chart_h - int(chart_h * (tick / 100.0) / max_y)
        draw.line((left, y, width - right, y), fill=(224, 220, 212), width=1)
        draw.text((34, y - 10), f"{tick}%", fill=(105, 111, 117), font=small)
    for epoch in range(0, max_epoch + 1, max(1, max_epoch // 8 or 1)):
        x = left + int(chart_w * epoch / max(1, max_epoch))
        draw.line((x, top, x, top + chart_h), fill=(232, 228, 220), width=1)
        draw.text((x - 8, top + chart_h + 14), str(epoch), fill=(105, 111, 117), font=small)

    draw.line((left, top, left, top + chart_h), fill=(90, 96, 102), width=2)
    draw.line((left, top + chart_h, width - right, top + chart_h), fill=(90, 96, 102), width=2)

    for run in runs:
        pts = []
        for item in run["history"]:
            x = left + int(chart_w * item["epoch"] / max(1, max_epoch))
            y = top + chart_h - int(chart_h * item["miou"] / max_y)
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=run["color"], width=4)
        for x, y in pts:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=run["color"])

    legend_x, legend_y = width - 470, 40
    for idx, run in enumerate(runs):
        y = legend_y + idx * 30
        draw.rounded_rectangle((legend_x, y + 4, legend_x + 24, y + 20), radius=4, fill=run["color"])
        draw.text((legend_x + 34, y), f"{run['label']} {run['name']}", fill=(45, 52, 59), font=small)

    img.save(out)


def draw_top_classes(runs: list[dict], out: Path) -> None:
    # Use classes from the strongest run, then compare the same IDs where available.
    strongest = max(runs, key=lambda r: r["best_miou_percent"])
    selected = sorted(strongest["classes"], key=lambda r: r["iou"], reverse=True)[:15]
    class_ids = [row["class_index"] for row in selected]

    width, height = 1550, 860
    img = Image.new("RGB", (width, height), (250, 248, 243))
    draw = ImageDraw.Draw(img)
    title = load_font(32, True)
    normal = load_font(18)
    small = load_font(15)

    draw.text((36, 26), "Топ-классы: сравнение IoU между 19/17/5%", fill=(31, 38, 44), font=title)
    draw.text((38, 70), "Классы взяты из топа лучшего запуска; значения в процентах", fill=(92, 99, 105), font=normal)

    left, top, chart_w = 330, 130, 1030
    group_h, bar_h = 47, 12
    max_v = 100.0

    for tick in range(0, 101, 20):
        x = left + int(chart_w * tick / max_v)
        draw.line((x, top - 10, x, top + group_h * len(class_ids) + 8), fill=(224, 220, 212), width=1)
        draw.text((x - 10, top - 36), str(tick), fill=(118, 124, 130), font=small)
    draw.text((left + chart_w + 18, top - 36), "%", fill=(118, 124, 130), font=small)

    by_run = {
        run["label"]: {row["class_index"]: row for row in run["classes"]}
        for run in runs
    }
    for idx, cid in enumerate(class_ids):
        ref = selected[idx]
        y = top + idx * group_h
        draw.text((36, y + 10), f"{cid:03d} {ref['class_name']}", fill=(45, 52, 59), font=normal)
        for r_idx, run in enumerate(runs):
            row = by_run[run["label"]].get(cid)
            value = (row["iou"] * 100.0) if row else 0.0
            yy = y + 4 + r_idx * (bar_h + 2)
            bar_w = int(chart_w * value / max_v)
            draw.rounded_rectangle((left, yy, left + bar_w, yy + bar_h), radius=5, fill=run["color"])
            draw.text((left + bar_w + 8, yy - 3), f"{run['label']} {value:.1f}", fill=(45, 52, 59), font=small)

    img.save(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for spec in RUNS:
        run = dict(spec)
        run["history"] = parse_history(run["log"])
        run["classes"] = read_class_csv(run["best_csv"])
        run["best_eval_miou"] = sum(row["iou"] for row in run["classes"]) / max(1, len(run["classes"]))
        run["best_eval_macc"] = sum(row["accuracy"] for row in run["classes"]) / max(1, len(run["classes"]))
        run["best_eval_allacc"] = (
            sum(row["intersection"] for row in run["classes"])
            / max(1.0, sum(row["target"] for row in run["classes"]))
        )
        run["best_miou_percent"] = run["best_eval_miou"] * 100.0
        runs.append(run)

    best_rows = []
    for run in runs:
        best_rows.append({
            "run": run["label"],
            "name": run["name"],
            "best_epoch": run["best_epoch"],
            "checkpoint_miou_percent": f"{run['best_ckpt_miou'] * 100:.2f}",
            "eval_miou_percent": f"{run['best_eval_miou'] * 100:.2f}",
            "eval_macc_percent": f"{run['best_eval_macc'] * 100:.2f}",
            "eval_allacc_percent": f"{run['best_eval_allacc'] * 100:.2f}",
            "val_points": f"{sum(row['target'] for row in run['classes']):.0f}",
            "checkpoint": str(run["checkpoint"]),
        })
    write_csv(
        out_dir / "comparison_best_metrics_ru.csv",
        best_rows,
        [
            "run",
            "name",
            "best_epoch",
            "checkpoint_miou_percent",
            "eval_miou_percent",
            "eval_macc_percent",
            "eval_allacc_percent",
            "val_points",
            "checkpoint",
        ],
    )

    history_rows = []
    for run in runs:
        for item in run["history"]:
            history_rows.append({
                "run": run["label"],
                "name": run["name"],
                "epoch": item["epoch"],
                "steps": item["steps"],
                "lr": item["lr"],
                "mIoU_percent_log": f"{item['miou'] * 100:.2f}",
                "mAcc_percent_log": f"{item['macc'] * 100:.2f}",
                "allAcc_percent_log": f"{item['allacc'] * 100:.2f}",
            })
    write_csv(
        out_dir / "comparison_val_history_ru.csv",
        history_rows,
        ["run", "name", "epoch", "steps", "lr", "mIoU_percent_log", "mAcc_percent_log", "allAcc_percent_log"],
    )

    top_rows = []
    for run in runs:
        for row in sorted(run["classes"], key=lambda x: x["iou"], reverse=True)[:20]:
            top_rows.append({
                "run": run["label"],
                "class_index": row["class_index"],
                "class_name": row["class_name"],
                "iou_percent": f"{row['iou'] * 100:.2f}",
                "accuracy_percent": f"{row['accuracy'] * 100:.2f}",
                "target_points": f"{row['target']:.0f}",
            })
    write_csv(
        out_dir / "comparison_top20_classes_ru.csv",
        top_rows,
        ["run", "class_index", "class_name", "iou_percent", "accuracy_percent", "target_points"],
    )

    draw_best_bars(runs, out_dir / "comparison_best_miou.png")
    draw_history(runs, out_dir / "comparison_miou_history.png")
    draw_top_classes(runs, out_dir / "comparison_top_classes_iou.png")

    lines = [
        "Сравнение трех запусков PointMixer panoptic: 19%, 17%, 5%",
        "",
    ]
    for run in sorted(runs, key=lambda r: r["best_miou_percent"], reverse=True):
        lines.append(
            f"{run['label']} ({run['name']}): best epoch {run['best_epoch']}, "
            f"mIoU {run['best_miou_percent']:.2f}%, "
            f"mAcc {run['best_eval_macc'] * 100:.2f}%, "
            f"allAcc {run['best_eval_allacc'] * 100:.2f}%."
        )
    lines.extend([
        "",
        "Вывод:",
        "1. Лучший из этих трех запусков - 19%: он дает самый высокий mIoU и лучший баланс по основным классам.",
        "2. 17% близок по крупным классам, но в среднем слабее; вероятно, сказались другой prepared dataset/resume и настройки.",
        "3. 5% - неудачный/сломанный запуск: модель почти не учит классы, кроме первых грубых поверхностей.",
        "",
        "Важно:",
        "папки относятся к разным подготовкам датасета (pv004, pv003, dense005), поэтому сравнение полезно как инженерное,",
        "но для строгого научного сравнения лучше прогнать все три чекпоинта на одном и том же val-root.",
    ])
    (out_dir / "comparison_summary_ru.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
