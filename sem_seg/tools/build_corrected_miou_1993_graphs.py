from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"E:\pointmixer_outputs")
OUT_DIR = ROOT / "scannetpp_compare_19_17_5_panoptic"

RUNS = [
    {
        "name": "pv004 focal+lovasz 19.93%",
        "short": "19.93%",
        "color": (28, 118, 194),
        "train_dir": ROOT
        / "PointMixerScanNetPP_panoptic_rgb_pv004_focal_lovasz_ep10_continue_to_miou020"
        / "2026-05-06_17-54-56__scannetpp__pointmixer_panoptic_3060",
    },
    {
        "name": "pv003 focal+lovasz 17.59%",
        "short": "17.59%",
        "color": (245, 147, 49),
        "train_dir": ROOT
        / "PointMixerScanNetPP_panoptic_rgb_pv003_focal_lovasz_ep20_resume"
        / "2026-05-10_17-47-13__scannetpp__pointmixer_panoptic_3060",
    },
    {
        "name": "dense005 focal+lovasz 0.70%",
        "short": "0.70%",
        "color": (148, 82, 173),
        "train_dir": ROOT
        / "PointMixerScanNetPP_panoptic_rgb_dense005_focal_lovasz_ep20"
        / "2026-05-08_19-31-36__scannetpp__pointmixer_panoptic_3060",
    },
]


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


FONT_TITLE = load_font(34, True)
FONT_SUBTITLE = load_font(20)
FONT_LABEL = load_font(21, True)
FONT_AXIS = load_font(18)
FONT_SMALL = load_font(15)


def t(text: str) -> str:
    return text


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def parse_checkpoints(train_dir: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    for path in train_dir.glob("epoch=*--mIoU_val=*.ckpt"):
        match = re.search(r"epoch=(\d+)--mIoU_val=([0-9.]+)--", path.name)
        if match:
            values[int(match.group(1))] = float(match.group(2))
    return values


def parse_log(train_dir: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    log_paths = sorted(train_dir.glob("log_train_*.log"))
    for log_path in log_paths:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"val\s*:\s*epoch\[(\d+)\].*?mIoU_val\[([0-9.]+)\]", text):
            values[int(match.group(1))] = float(match.group(2))
    return values


def collect_run(run: dict) -> dict:
    train_dir = run["train_dir"]
    log_values = parse_log(train_dir)
    ckpt_values = parse_checkpoints(train_dir)
    history = dict(log_values)
    history.update(ckpt_values)
    best_epoch = None
    best_miou = -1.0
    for epoch, miou in sorted(history.items()):
        if miou > best_miou:
            best_epoch = epoch
            best_miou = miou
    return {
        **run,
        "history": history,
        "ckpt_values": ckpt_values,
        "log_values": log_values,
        "best_epoch": best_epoch,
        "best_miou": best_miou,
    }


def draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((60, 34), title, font=FONT_TITLE, fill=(28, 31, 38))
    draw.text((60, 78), subtitle, font=FONT_SUBTITLE, fill=(88, 96, 110))


def draw_best_miou(runs: list[dict], out_path: Path) -> None:
    width, height = 1500, 900
    img = Image.new("RGB", (width, height), (248, 249, 252))
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "\u041e\u0431\u0449\u0438\u0439 \u0433\u0440\u0430\u0444\u0438\u043a best mIoU",
        "\u0421\u0440\u0430\u0432\u043d\u0435\u043d\u0438\u0435 \u043b\u0443\u0447\u0448\u0438\u0445 validation mIoU \u043f\u043e \u0437\u0430\u043f\u0443\u0441\u043a\u0430\u043c",
    )

    left, right = 420, 1380
    top, row_h = 185, 150
    max_val = max(run["best_miou"] for run in runs) * 100
    axis_max = max(22.0, ((int(max_val / 5) + 1) * 5))

    for tick in range(0, int(axis_max) + 1, 5):
        x = left + int((right - left) * tick / axis_max)
        draw.line((x, top - 30, x, top + row_h * len(runs) - 20), fill=(222, 226, 234), width=1)
        label = f"{tick}%"
        tw, _ = text_size(draw, label, FONT_SMALL)
        draw.text((x - tw / 2, top + row_h * len(runs) - 5), label, font=FONT_SMALL, fill=(95, 103, 118))

    for i, run in enumerate(runs):
        y = top + i * row_h
        miou_pct = run["best_miou"] * 100
        bar_w = int((right - left) * miou_pct / axis_max)
        draw.text((60, y + 19), run["name"], font=FONT_LABEL, fill=(34, 39, 48))
        draw.text(
            (60, y + 50),
            f"epoch {run['best_epoch']} | mIoU={miou_pct:.2f}%",
            font=FONT_AXIS,
            fill=(92, 101, 116),
        )
        draw.rounded_rectangle((left, y + 18, right, y + 75), radius=14, fill=(234, 238, 245))
        draw.rounded_rectangle((left, y + 18, left + bar_w, y + 75), radius=14, fill=run["color"])
        value = f"{miou_pct:.2f}%"
        draw.text((left + bar_w + 18, y + 31), value, font=FONT_LABEL, fill=(28, 31, 38))

    note = (
        "\u0418\u0441\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u043e: \u0434\u043b\u044f pv004 \u0431\u0435\u0440\u0435\u0442\u0441\u044f "
        "epoch=008--mIoU_val=0.1993--.ckpt, \u0442\u043e \u0435\u0441\u0442\u044c 19.93%."
    )
    draw.text((60, height - 70), note, font=FONT_SMALL, fill=(105, 112, 126))
    img.save(out_path)


def draw_history(runs: list[dict], out_path: Path, only_first: bool = False) -> None:
    plot_runs = runs[:1] if only_first else runs
    width, height = 1500, 900
    img = Image.new("RGB", (width, height), (248, 249, 252))
    draw = ImageDraw.Draw(img)

    title = "\u0418\u0441\u0442\u043e\u0440\u0438\u044f mIoU \u0434\u043b\u044f 19.93%" if only_first else "\u0418\u0441\u0442\u043e\u0440\u0438\u044f mIoU \u043f\u043e \u0437\u0430\u043f\u0443\u0441\u043a\u0430\u043c"
    subtitle = "mIoU \u043f\u043e epoch; \u0442\u043e\u0447\u043d\u044b\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f checkpoint \u0437\u0430\u043c\u0435\u043d\u044f\u044e\u0442 \u043e\u043a\u0440\u0443\u0433\u043b\u0435\u043d\u043d\u044b\u0435 \u043b\u043e\u0433\u0438"
    draw_header(draw, title, subtitle)

    left, right = 105, 1415
    top, bottom = 170, 760
    all_epochs = [epoch for run in plot_runs for epoch in run["history"]]
    all_vals = [value * 100 for run in plot_runs for value in run["history"].values()]
    min_epoch, max_epoch = min(all_epochs), max(all_epochs)
    min_val = min(0.0, min(all_vals) - 2)
    max_val = max(22.0, max(all_vals) + 2)

    draw.rectangle((left, top, right, bottom), outline=(215, 220, 229), width=2, fill=(255, 255, 255))

    for tick in range(0, int(max_val) + 1, 5):
        y = bottom - int((bottom - top) * (tick - min_val) / (max_val - min_val))
        draw.line((left, y, right, y), fill=(230, 234, 241), width=1)
        draw.text((35, y - 10), f"{tick}%", font=FONT_SMALL, fill=(98, 106, 121))

    epoch_ticks = range(min_epoch, max_epoch + 1, max(1, (max_epoch - min_epoch) // 10 or 1))
    for epoch in epoch_ticks:
        x = left + int((right - left) * (epoch - min_epoch) / max(1, max_epoch - min_epoch))
        draw.line((x, bottom, x, bottom + 8), fill=(98, 106, 121), width=1)
        label = str(epoch)
        tw, _ = text_size(draw, label, FONT_SMALL)
        draw.text((x - tw / 2, bottom + 14), label, font=FONT_SMALL, fill=(98, 106, 121))

    for run in plot_runs:
        points = []
        for epoch, miou in sorted(run["history"].items()):
            x = left + int((right - left) * (epoch - min_epoch) / max(1, max_epoch - min_epoch))
            y = bottom - int((bottom - top) * ((miou * 100) - min_val) / (max_val - min_val))
            points.append((x, y, epoch, miou * 100))
        if len(points) >= 2:
            draw.line([(x, y) for x, y, _, _ in points], fill=run["color"], width=5)
        for x, y, epoch, miou_pct in points:
            r = 7 if epoch == run["best_epoch"] else 5
            draw.ellipse((x - r, y - r, x + r, y + r), fill=run["color"], outline=(255, 255, 255), width=2)
            if epoch == run["best_epoch"]:
                label = f"{miou_pct:.2f}%"
                draw.rounded_rectangle((x - 54, y - 46, x + 54, y - 18), radius=8, fill=(28, 31, 38))
                tw, th = text_size(draw, label, FONT_SMALL)
                draw.text((x - tw / 2, y - 42), label, font=FONT_SMALL, fill=(255, 255, 255))

    legend_y = 805
    legend_x = 105
    for run in plot_runs:
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 30, legend_y + 18), radius=5, fill=run["color"])
        label = f"{run['name']} | best {run['best_miou'] * 100:.2f}%"
        draw.text((legend_x + 42, legend_y - 2), label, font=FONT_AXIS, fill=(43, 48, 58))
        legend_x += 470 if not only_first else 0
        legend_y += 0 if not only_first else 34

    draw.text((left, bottom + 52), "epoch", font=FONT_AXIS, fill=(79, 87, 101))
    draw.text((40, top - 35), "mIoU (%)", font=FONT_AXIS, fill=(79, 87, 101))
    img.save(out_path)


def write_csv(runs: list[dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "best_epoch", "best_miou", "best_miou_percent", "train_dir"])
        for run in runs:
            writer.writerow(
                [
                    run["name"],
                    run["best_epoch"],
                    f"{run['best_miou']:.4f}",
                    f"{run['best_miou'] * 100:.2f}",
                    str(run["train_dir"]),
                ]
            )

    history_path = out_path.with_name("comparison_miou_history_corrected_1993.csv")
    with history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "epoch", "miou", "miou_percent", "source"])
        for run in runs:
            for epoch, miou in sorted(run["history"].items()):
                source = "checkpoint" if epoch in run["ckpt_values"] else "log_rounded"
                writer.writerow([run["name"], epoch, f"{miou:.4f}", f"{miou * 100:.2f}", source])


def write_summary(runs: list[dict], out_path: Path) -> None:
    lines = [
        "\u0418\u0441\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u043d\u044b\u0435 \u0433\u0440\u0430\u0444\u0438\u043a\u0438 mIoU",
        "",
        "\u0413\u043b\u0430\u0432\u043d\u044b\u0439 \u0437\u0430\u043f\u0443\u0441\u043a:",
        "pv004 focal+lovasz 19.93%",
        str(RUNS[0]["train_dir"] / "epoch=008--mIoU_val=0.1993--.ckpt"),
        "",
        "\u041b\u0443\u0447\u0448\u0438\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u044f:",
    ]
    for run in runs:
        lines.append(
            f"- {run['name']}: epoch {run['best_epoch']}, mIoU {run['best_miou'] * 100:.2f}%"
        )
    lines.extend(
        [
            "",
            "\u0424\u0430\u0439\u043b\u044b:",
            "comparison_best_miou_corrected_1993.png",
            "comparison_miou_history_corrected_1993.png",
            "miou_1993_history.png",
            "comparison_best_metrics_corrected_1993.csv",
            "comparison_miou_history_corrected_1993.csv",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = [collect_run(run) for run in RUNS]
    best_path = OUT_DIR / "comparison_best_miou_corrected_1993.png"
    history_path = OUT_DIR / "comparison_miou_history_corrected_1993.png"
    draw_best_miou(runs, best_path)
    draw_history(runs, history_path, only_first=False)
    draw_history(runs, OUT_DIR / "miou_1993_history.png", only_first=True)
    write_csv(runs, OUT_DIR / "comparison_best_metrics_corrected_1993.csv")
    write_summary(runs, OUT_DIR / "comparison_summary_corrected_1993.txt")
    # Keep the old expected filenames updated as well, so the previous report
    # does not accidentally show the obsolete 18.66% label.
    draw_best_miou(runs, OUT_DIR / "comparison_best_miou.png")
    draw_history(runs, OUT_DIR / "comparison_miou_history.png", only_first=False)
    print(OUT_DIR)


if __name__ == "__main__":
    main()
