from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def read_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "rank_by_iou": int(row["rank_by_iou"]),
                "class_index": int(row["class_index"]),
                "class_name": row["class_name"],
                "iou": float(row["iou"]),
                "accuracy": float(row["accuracy"]),
                "intersection": float(row["intersection"]),
                "union": float(row["union"]),
                "target": float(row["target"]),
            }
            rows.append(item)
    total_target = sum(row["target"] for row in rows) or 1.0
    for row in rows:
        row["iou_percent"] = row["iou"] * 100.0
        row["accuracy_percent"] = row["accuracy"] * 100.0
        row["target_percent"] = row["target"] / total_target * 100.0
    return rows


def write_percent_csv(rows: list[dict], out_path: Path) -> None:
    fields = [
        "rank_by_iou",
        "class_index",
        "class_name",
        "iou_percent",
        "accuracy_percent",
        "target_points",
        "target_percent",
        "intersection",
        "union",
    ]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for row in sorted(rows, key=lambda x: x["rank_by_iou"]):
            writer.writerow({
                "rank_by_iou": row["rank_by_iou"],
                "class_index": row["class_index"],
                "class_name": row["class_name"],
                "iou_percent": f"{row['iou_percent']:.2f}",
                "accuracy_percent": f"{row['accuracy_percent']:.2f}",
                "target_points": f"{row['target']:.0f}",
                "target_percent": f"{row['target_percent']:.4f}",
                "intersection": f"{row['intersection']:.0f}",
                "union": f"{row['union']:.0f}",
            })


def short_name(name: str, max_len: int = 28) -> str:
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def draw_bar_chart(
    rows: list[dict],
    out_path: Path,
    title: str,
    subtitle: str,
    value_key: str,
    color: tuple[int, int, int],
) -> None:
    width = 1500
    row_h = 34
    top = 118
    bottom = 48
    left = 360
    right = 150
    height = top + bottom + row_h * len(rows)

    img = Image.new("RGB", (width, height), (250, 248, 243))
    draw = ImageDraw.Draw(img)

    title_font = load_font(34, bold=True)
    sub_font = load_font(20)
    label_font = load_font(18)
    small_font = load_font(16)

    draw.text((36, 24), title, fill=(31, 38, 44), font=title_font)
    draw.text((38, 70), subtitle, fill=(92, 99, 105), font=sub_font)

    chart_w = width - left - right
    max_value = max([row[value_key] for row in rows] + [1.0])
    scale_max = max(5.0, min(100.0, ((int(max_value / 10) + 1) * 10)))

    # Grid lines.
    for tick in range(0, int(scale_max) + 1, 10):
        x = left + int(chart_w * tick / scale_max)
        draw.line((x, top - 8, x, height - bottom + 6), fill=(224, 220, 212), width=1)
        draw.text((x - 12, top - 34), f"{tick}", fill=(118, 124, 130), font=small_font)
    draw.text((left + chart_w + 18, top - 34), "%", fill=(118, 124, 130), font=small_font)

    for idx, row in enumerate(rows):
        y = top + idx * row_h
        name = f"{row['class_index']:03d}  {short_name(row['class_name'])}"
        draw.text((36, y + 5), name, fill=(45, 52, 59), font=label_font)

        value = row[value_key]
        bar_w = int(chart_w * value / scale_max)
        draw.rounded_rectangle(
            (left, y + 5, left + bar_w, y + row_h - 7),
            radius=7,
            fill=color,
        )
        value_text = f"{value:.1f}%"
        draw.text((left + bar_w + 10, y + 5), value_text, fill=(31, 38, 44), font=label_font)
        target_text = f"target: {row['target']:.0f}"
        draw.text((width - right + 8, y + 6), target_text, fill=(105, 111, 117), font=small_font)

    img.save(out_path)


def write_summary(rows: list[dict], out_path: Path, run_label: str) -> None:
    valid_rows = [row for row in rows if row["target"] > 0]
    miou = sum(row["iou"] for row in rows) / max(1, len(rows)) * 100.0
    macc = sum(row["accuracy"] for row in rows) / max(1, len(rows)) * 100.0
    all_acc = (
        sum(row["intersection"] for row in rows)
        / max(1.0, sum(row["target"] for row in rows))
        * 100.0
    )
    best = sorted(valid_rows, key=lambda x: x["iou_percent"], reverse=True)[:10]
    worst = sorted(valid_rows, key=lambda x: x["iou_percent"])[:10]

    lines = [
        f"ScanNet++ validation, PointMixer {run_label}",
        "",
        f"mIoU по 100 классам: {miou:.2f}%",
        f"mAcc по 100 классам: {macc:.2f}%",
        f"allAcc по всем точкам: {all_acc:.2f}%",
        f"Всего val-точек с разметкой: {sum(row['target'] for row in rows):.0f}",
        "",
        "Лучше всего модель понимает:",
    ]
    for row in best:
        lines.append(
            f"- {row['class_index']:03d} {row['class_name']}: "
            f"IoU {row['iou_percent']:.2f}%, Acc {row['accuracy_percent']:.2f}%, "
            f"точек {row['target']:.0f}"
        )
    lines.extend(["", "Хуже всего среди присутствующих классов:"])
    for row in worst:
        lines.append(
            f"- {row['class_index']:03d} {row['class_name']}: "
            f"IoU {row['iou_percent']:.2f}%, Acc {row['accuracy_percent']:.2f}%, "
            f"точек {row['target']:.0f}"
        )
    lines.extend([
        "",
        "Короткий вывод:",
        "модель уверенно распознает крупные и частые классы сцены: floor, ceiling, wall, table.",
        f"Для редких и мелких категорий качество резко ниже, поэтому средний mIoU остается около {miou:.2f}%,",
        "хотя общая accuracy по точкам высокая за счет крупных поверхностей.",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="scannetpp_val_epoch020")
    parser.add_argument(
        "--run-label",
        default="checkpoint epoch=020--mIoU_val=0.1720",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_metrics(csv_path)
    present = [row for row in rows if row["target"] > 0]
    top30 = sorted(present, key=lambda x: x["iou_percent"], reverse=True)[:30]
    bottom30 = sorted(present, key=lambda x: x["iou_percent"])[:30]
    top30_acc = sorted(present, key=lambda x: x["accuracy_percent"], reverse=True)[:30]
    top30_target = sorted(present, key=lambda x: x["target_percent"], reverse=True)[:30]

    write_percent_csv(rows, out_dir / f"{args.prefix}_class_metrics_percent_ru.csv")
    draw_bar_chart(
        top30,
        out_dir / f"{args.prefix}_top30_iou_percent.png",
        "ScanNet++ val: топ-30 классов по IoU",
        f"PointMixer {args.run_label}, метрика IoU по классам",
        "iou_percent",
        (45, 128, 115),
    )
    draw_bar_chart(
        bottom30,
        out_dir / f"{args.prefix}_bottom30_iou_percent.png",
        "ScanNet++ val: 30 самых слабых классов по IoU",
        f"PointMixer {args.run_label}: классы, которые модель почти не отделяет",
        "iou_percent",
        (190, 90, 76),
    )
    draw_bar_chart(
        top30_acc,
        out_dir / f"{args.prefix}_top30_accuracy_percent.png",
        "ScanNet++ val: топ-30 классов по accuracy",
        f"PointMixer {args.run_label}, доля правильно найденных точек внутри класса",
        "accuracy_percent",
        (70, 112, 168),
    )
    draw_bar_chart(
        top30_target,
        out_dir / f"{args.prefix}_top30_class_frequency_percent.png",
        "ScanNet++ val: самые частые классы в разметке",
        "Доля точек класса среди всех val-точек; крупные классы сильнее влияют на allAcc",
        "target_percent",
        (185, 122, 54),
    )
    write_summary(rows, out_dir / f"{args.prefix}_summary_ru.txt", args.run_label)


if __name__ == "__main__":
    main()
