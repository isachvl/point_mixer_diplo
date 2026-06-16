from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LOG_PATH = Path(
    r"E:\сохры\метрики хорошие паноптика и лоссы"
    r"\2026-05-05_18-50-33__scannetpp__pointmixer_panoptic_3060"
    r"\log_train_2026-05-05 1852.log"
)
CLASS_CSV = Path(
    r"E:\pointmixer_outputs\scannetpp_val_epoch010_19percent_class_eval"
    r"\class_metrics\val_class_metrics_epoch_000.csv"
)
OUT_DIR = Path(
    r"E:\pointmixer_outputs\scannetpp_val_epoch010_19percent_class_eval"
    r"\class_metrics"
)


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


def parse_history() -> list[dict]:
    items = []
    for line in LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        match = VAL_RE.search(line)
        if not match:
            continue
        items.append({
            "epoch": int(match.group("epoch")),
            "steps": int(match.group("steps")),
            "lr": float(match.group("lr")),
            "miou": float(match.group("miou")),
            "macc": float(match.group("macc")),
            "allacc": float(match.group("allacc")),
        })
    return items


def exact_miou_from_csv() -> float:
    with CLASS_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return sum(float(row["iou"]) for row in rows) / max(1, len(rows))


def write_history_csv(history: list[dict], exact_best: float) -> None:
    out = OUT_DIR / "scannetpp_val_19percent_miou_history_ru.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter=";",
            fieldnames=[
                "epoch",
                "steps",
                "lr",
                "mIoU_percent_log",
                "mAcc_percent_log",
                "allAcc_percent_log",
                "note",
            ],
        )
        writer.writeheader()
        for item in history:
            writer.writerow({
                "epoch": item["epoch"],
                "steps": item["steps"],
                "lr": item["lr"],
                "mIoU_percent_log": f"{item['miou'] * 100:.2f}",
                "mAcc_percent_log": f"{item['macc'] * 100:.2f}",
                "allAcc_percent_log": f"{item['allacc'] * 100:.2f}",
                "note": "best checkpoint; exact re-eval mIoU %.2f%%" % (exact_best * 100.0)
                if item["epoch"] == 10 else "",
            })


def draw_miou_history(history: list[dict], exact_best: float) -> None:
    width, height = 1450, 820
    img = Image.new("RGB", (width, height), (250, 248, 243))
    draw = ImageDraw.Draw(img)
    title = load_font(36, True)
    normal = load_font(20)
    small = load_font(16)
    label = load_font(18, True)

    draw.text((38, 28), "mIoU PointMixer panoptic, запуск 19%", fill=(31, 38, 44), font=title)
    draw.text(
        (40, 78),
        "ScanNet++ validation по эпохам; лучший чекпоинт epoch=010",
        fill=(92, 99, 105),
        font=normal,
    )

    left, top, right, bottom = 95, 150, 80, 105
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_epoch = max(item["epoch"] for item in history)
    max_y = 0.20

    for tick in range(0, 21, 5):
        y = top + chart_h - int(chart_h * (tick / 100.0) / max_y)
        draw.line((left, y, width - right, y), fill=(224, 220, 212), width=1)
        draw.text((35, y - 10), f"{tick}%", fill=(105, 111, 117), font=small)

    for epoch in range(0, max_epoch + 1, 2):
        x = left + int(chart_w * epoch / max(1, max_epoch))
        draw.line((x, top, x, top + chart_h), fill=(232, 228, 220), width=1)
        draw.text((x - 8, top + chart_h + 14), str(epoch), fill=(105, 111, 117), font=small)

    draw.line((left, top, left, top + chart_h), fill=(90, 96, 102), width=2)
    draw.line((left, top + chart_h, width - right, top + chart_h), fill=(90, 96, 102), width=2)
    draw.text((left + chart_w // 2 - 35, height - 42), "Эпоха", fill=(80, 86, 92), font=normal)
    draw.text((22, top - 34), "mIoU", fill=(80, 86, 92), font=normal)

    points = []
    for item in history:
        x = left + int(chart_w * item["epoch"] / max(1, max_epoch))
        y = top + chart_h - int(chart_h * item["miou"] / max_y)
        points.append((x, y, item))

    line_points = [(x, y) for x, y, _ in points]
    draw.line(line_points, fill=(45, 128, 115), width=5)

    best_log = max(history, key=lambda item: item["miou"])
    for x, y, item in points:
        is_best = item["epoch"] == best_log["epoch"]
        r = 8 if is_best else 5
        fill = (31, 38, 44) if is_best else (45, 128, 115)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)
        if is_best:
            draw.text((x + 14, y - 18), f"best log: {item['miou'] * 100:.0f}%", fill=(31, 38, 44), font=label)

    card_x, card_y = 930, 160
    draw.rounded_rectangle((card_x, card_y, card_x + 405, card_y + 170), radius=18, fill=(238, 234, 224))
    draw.text((card_x + 24, card_y + 22), "Лучший результат", fill=(31, 38, 44), font=label)
    draw.text((card_x + 24, card_y + 62), f"mIoU = {exact_best * 100:.2f}%", fill=(45, 128, 115), font=load_font(34, True))
    draw.text((card_x + 24, card_y + 112), "epoch=010, чекпоинт 0.1855", fill=(92, 99, 105), font=normal)

    img.save(OUT_DIR / "scannetpp_val_19percent_miou_history.png")


def draw_miou_card(exact_best: float) -> None:
    width, height = 1100, 430
    img = Image.new("RGB", (width, height), (250, 248, 243))
    draw = ImageDraw.Draw(img)
    title = load_font(34, True)
    normal = load_font(20)
    huge = load_font(72, True)

    draw.text((40, 30), "Итоговая mIoU для 19% запуска", fill=(31, 38, 44), font=title)
    draw.text((42, 78), "ScanNet++ validation, PointMixer panoptic, best epoch=010", fill=(92, 99, 105), font=normal)

    left, top, chart_w, bar_h = 80, 190, 880, 70
    draw.rounded_rectangle((left, top, left + chart_w, top + bar_h), radius=18, fill=(224, 220, 212))
    fill_w = int(chart_w * (exact_best * 100.0) / 25.0)
    draw.rounded_rectangle((left, top, left + fill_w, top + bar_h), radius=18, fill=(45, 128, 115))
    draw.text((left + fill_w + 22, top - 9), f"{exact_best * 100:.2f}%", fill=(31, 38, 44), font=huge)
    draw.text((80, 310), "mIoU = mean Intersection over Union по всем 100 классам", fill=(92, 99, 105), font=normal)
    img.save(OUT_DIR / "scannetpp_val_19percent_miou_result.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history = parse_history()
    exact_best = exact_miou_from_csv()
    write_history_csv(history, exact_best)
    draw_miou_history(history, exact_best)
    draw_miou_card(exact_best)


if __name__ == "__main__":
    main()
