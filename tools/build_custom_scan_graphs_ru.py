import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_OUT = Path(r"E:\pointmixer_outputs\eval_1file500000000_custom_finetune_ep005_true_accuracy_nearest005")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--scan-title", default="собственный скан института")
    parser.add_argument("--prefix", default="custom_scan")
    return parser.parse_args()


def parse_summary(path: Path) -> dict:
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def load_font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_bar_chart(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    subtitle: str,
    footnote: str,
    out_path: Path,
    sort_col: str,
    color: tuple[int, int, int],
    max_100: bool = True,
) -> None:
    data = data.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce").fillna(0.0)
    data = data.sort_values(sort_col, ascending=False).head(20).iloc[::-1]

    width, height = 1580, 940
    margin_left, margin_right, margin_top, margin_bottom = 325, 170, 120, 90
    row_h = (height - margin_top - margin_bottom) / max(len(data), 1)
    plot_w = width - margin_left - margin_right
    max_val = 100.0 if max_100 else max(1.0, float(data[value_col].max()) * 1.18)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_title = load_font(34, True)
    font_sub = load_font(20)
    font_axis = load_font(18)
    font_label = load_font(20)
    font_small = load_font(16)

    draw.text((42, 28), title, fill=(22, 31, 45), font=font_title)
    draw.text((44, 75), subtitle, fill=(87, 96, 112), font=font_sub)

    if max_100:
        for tick in range(0, 101, 20):
            x = margin_left + plot_w * tick / 100.0
            draw.line((x, margin_top - 10, x, height - margin_bottom + 5), fill=(229, 234, 240), width=1)
            draw.text((x - 13, height - margin_bottom + 17), f"{tick}%", fill=(120, 130, 142), font=font_axis)

    for j, row in enumerate(data.itertuples(index=False)):
        y = margin_top + j * row_h + 6
        label = f"{int(getattr(row, 'class_index')):02d} {getattr(row, 'class_name')}"
        if len(label) > 30:
            label = label[:29] + "..."
        draw.text((38, y + 8), label, fill=(35, 44, 58), font=font_label)

        value = float(getattr(row, value_col))
        bar_w = int(plot_w * value / max_val)
        x0 = margin_left
        y0 = int(y + 7)
        y1 = int(y + row_h - 8)
        draw.rounded_rectangle((x0, y0, x0 + max(bar_w, 2), y1), radius=6, fill=color)
        draw.text((x0 + max(bar_w, 2) + 10, y0 + 2), f"{value:.2f}%", fill=(29, 38, 54), font=font_label)

        extra = (
            f"GT {int(getattr(row, 'gt_count')):,} | "
            f"Pred {int(getattr(row, 'pred_count')):,} | "
            f"TP {int(getattr(row, 'tp')):,}"
        )
        draw.text((x0 + max(bar_w, 2) + 125, y0 + 6), extra, fill=(99, 108, 122), font=font_small)

    draw.text((42, height - 48), footnote, fill=(99, 108, 122), font=font_small)
    image.save(out_path)


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    summary_src = out / "summary.txt"
    metrics_path = out / "custom_scan_class_iou_and_percent.csv"
    if not metrics_path.exists():
        raw_metrics = pd.read_csv(out / "class_metrics.csv")
        raw_metrics["IoU_percent"] = raw_metrics["iou"].fillna(0) * 100
        raw_metrics["Acc_percent"] = raw_metrics["acc"].fillna(0) * 100
        raw_metrics["GT_share_percent"] = raw_metrics["gt_count"] / max(raw_metrics["gt_count"].sum(), 1) * 100
        raw_metrics["Pred_share_valid_percent"] = raw_metrics["pred_count"] / max(raw_metrics["pred_count"].sum(), 1) * 100
        raw_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    values = parse_summary(summary_src)
    df = pd.read_csv(metrics_path)

    pred_points = int(values.get("pred_points", "0").replace(",", ""))
    matched_any = int(values.get("matched_gt_any_label", "0").replace(",", ""))
    valid_points = int(values.get("valid_gt_mapped_points", "0").replace(",", ""))
    present_gt_classes = int(values.get("present_gt_classes", "0"))
    miou = float(values.get("mIoU_gt_present", "0")) * 100
    macc = float(values.get("mAcc_gt_present", "0")) * 100
    allacc = float(values.get("allAcc", "0")) * 100

    present = df[df["gt_count"] > 0].copy()

    draw_bar_chart(
        present,
        "IoU_percent",
        f"IoU по классам: {args.scan_title}",
        f"mIoU={miou:.2f}% | mAcc={macc:.2f}% | allAcc={allacc:.2f}% | GT-классов={present_gt_classes}",
        "IoU = TP / (GT + Pred - TP). График построен по классам, присутствующим в ручной GT-разметке.",
        out / f"{args.prefix}_iou_by_class.png",
        "gt_count",
        (44, 112, 203),
        True,
    )

    draw_bar_chart(
        df[df["pred_count"] > 0].copy(),
        "Pred_share_valid_percent",
        f"Доля предсказанных классов: {args.scan_title}",
        f"Распределение ответов модели на {valid_points:,} валидно сопоставленных точках",
        "Pred% показывает, какую долю оценочных точек модель отнесла к каждому классу.",
        out / f"{args.prefix}_predicted_class_percent.png",
        "pred_count",
        (36, 151, 116),
        False,
    )

    top_pred = df.sort_values("pred_count", ascending=False).head(8)
    best_iou = present.sort_values("IoU_percent", ascending=False).head(6)
    worst_iou = present.sort_values("IoU_percent", ascending=True).head(6)

    lines = [
        f"Краткое описание результатов: {args.scan_title}",
        "",
        (
            f"Для собственного скана было предсказано {pred_points:,} точек. "
            f"С ручной GT-разметкой удалось сопоставить {matched_any:,} точек, "
            f"из них {valid_points:,} точек вошли в расчёт метрик после remap к top-100 классам ScanNet++."
        ),
        (
            f"По {present_gt_classes} классам, присутствующим в GT-разметке, "
            f"получено mIoU = {miou:.2f}%, mAcc = {macc:.2f}%, allAcc = {allacc:.2f}%."
        ),
        "",
        (
            "mIoU считается как среднее IoU по классам, присутствующим в GT. "
            "Для каждого класса IoU = TP / (GT + Pred - TP), где TP — правильно найденные точки класса, "
            "GT — количество точек класса в разметке, Pred — количество точек, которые модель отнесла к этому классу."
        ),
        "",
        "Чаще всего модель предсказывала:",
    ]

    for row in top_pred.itertuples(index=False):
        lines.append(
            f"- {int(row.class_index):02d} {row.class_name}: "
            f"{int(row.pred_count):,} точек, {float(row.Pred_share_valid_percent):.2f}% оценочных точек"
        )

    lines.append("")
    lines.append("Лучшие классы по IoU:")
    for row in best_iou.itertuples(index=False):
        lines.append(
            f"- {int(row.class_index):02d} {row.class_name}: "
            f"IoU {float(row.IoU_percent):.2f}%, GT {int(row.gt_count):,}, "
            f"Pred {int(row.pred_count):,}, TP {int(row.tp):,}"
        )

    lines.append("")
    lines.append("Проблемные классы по IoU:")
    for row in worst_iou.itertuples(index=False):
        lines.append(
            f"- {int(row.class_index):02d} {row.class_name}: "
            f"IoU {float(row.IoU_percent):.2f}%, GT {int(row.gt_count):,}, "
            f"Pred {int(row.pred_count):,}, TP {int(row.tp):,}"
        )

    lines.extend(
        [
            "",
            (
                "Вывод: модель лучше всего распознала крупные и геометрически выраженные области, "
                "прежде всего wall, computer tower и ceiling. При этом часть классов собственного скана "
                "почти не совпала с ручной разметкой, что видно по нулевому IoU у floor, table, tv, "
                "ceiling lamp и chair. Это объясняет низкий mIoU: модель часто угадывает крупные поверхности, "
                "но плохо переносится на специфичный собственный скан и путает редкие/локальные объекты."
            ),
        ]
    )

    (out / f"{args.prefix}_short_description_ru.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
