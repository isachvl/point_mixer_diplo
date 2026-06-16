from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(r"E:\pointmixer_outputs\scannetpp_compare_19_17_5_panoptic")

RUNS = [
    {
        "label": "19.93% run",
        "legend": "19.93% mIoU_val",
        "color": (122, 64, 238),
        "dir": Path(
            r"E:\pointmixer_outputs\PointMixerScanNetPP_panoptic_rgb_pv004_focal_lovasz_ep10_continue_to_miou020\2026-05-06_17-54-56__scannetpp__pointmixer_panoptic_3060"
        ),
    },
    {
        "label": "17.59% run",
        "legend": "17.59% mIoU_val",
        "color": (43, 126, 214),
        "dir": Path(
            r"E:\pointmixer_outputs\PointMixerScanNetPP_panoptic_rgb_pv003_focal_lovasz_ep20_resume\2026-05-10_17-47-13__scannetpp__pointmixer_panoptic_3060"
        ),
    },
    {
        "label": "5% run",
        "legend": "5% run mIoU_val",
        "color": (85, 150, 75),
        "dir": Path(
            r"E:\pointmixer_outputs\PointMixerScanNetPP_panoptic_rgb_dense005_focal_lovasz_ep20\2026-05-08_19-31-36__scannetpp__pointmixer_panoptic_3060"
        ),
    },
]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


FONT_TITLE = font(18, True)
FONT_AXIS = font(15)
FONT_TICK = font(13)
FONT_LEGEND = font(14)


def text_wh(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def parse_checkpoints(run_dir: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    for path in run_dir.glob("epoch=*--mIoU_val=*.ckpt"):
        match = re.search(r"epoch=(\d+)--mIoU_val=([0-9.]+)--", path.name)
        if match:
            result[int(match.group(1))] = float(match.group(2))
    return result


def parse_log(run_dir: Path) -> list[tuple[int, int, float]]:
    result: list[tuple[int, int, float]] = []
    for path in sorted(run_dir.glob("log_train_*.log")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        pattern = r"val\s*:\s*epoch\[(\d+)\],\s*steps\[(\d+)\].*?mIoU_val\[([0-9.]+)\]"
        for match in re.finditer(pattern, text):
            epoch = int(match.group(1))
            step = int(match.group(2))
            miou = float(match.group(3))
            result.append((epoch, step, miou))
    return result


def load_run(run: dict) -> dict:
    log_values = parse_log(run["dir"])
    ckpt_values = parse_checkpoints(run["dir"])
    points: list[dict] = []
    max_step_by_epoch: dict[int, int] = {}
    for epoch, step, _ in log_values:
        max_step_by_epoch[epoch] = max(max_step_by_epoch.get(epoch, -1), step)
    for epoch, step, miou in sorted(log_values, key=lambda item: item[1]):
        # If we have an exact checkpoint value, use it on the final validation
        # point of this epoch. Early sanity/initial validation points stay as-is.
        if epoch in ckpt_values and step == max_step_by_epoch.get(epoch):
            exact = ckpt_values[epoch]
            source = "checkpoint"
        else:
            exact = miou
            source = "log"
        points.append({"epoch": epoch, "step": step, "miou": exact, "source": source})
    for epoch, miou in sorted(ckpt_values.items()):
        if epoch not in max_step_by_epoch:
            # Fallback if a checkpoint has no log line: keep epoch order and
            # approximate the step from nearby epochs.
            step = epoch
            points.append({"epoch": epoch, "step": step, "miou": miou, "source": "checkpoint"})
    points = sorted(points, key=lambda item: item["step"])
    best = max(points, key=lambda item: item["miou"]) if points else None
    return {**run, "points": points, "best": best, "ckpt_values": ckpt_values}


def nice_ticks(vmin: float, vmax: float, count: int = 8) -> list[float]:
    if vmax <= vmin:
        return [vmin]
    raw = (vmax - vmin) / max(1, count)
    candidates = [0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1]
    step = candidates[-1]
    for candidate in candidates:
        if raw <= candidate:
            step = candidate
            break
    start = int(vmin / step) * step
    ticks: list[float] = []
    value = start
    while value <= vmax + step * 0.5:
        if value >= vmin - step * 0.5:
            ticks.append(round(value, 4))
        value += step
    return ticks


def plot_line(
    runs: list[dict],
    title: str,
    out_path: Path,
    y_min: float | None = None,
    y_max: float | None = None,
    single_legend: bool = False,
) -> None:
    width, height = 1340, 565
    left, right = 78, 1305
    top, bottom = 38, 505

    all_points = [point for run in runs for point in run["points"]]
    x_min = min(point["step"] for point in all_points)
    x_max = max(point["step"] for point in all_points)
    values = [point["miou"] for point in all_points]
    if y_min is None:
        y_min = min(0.0, min(values) - 0.01)
    if y_max is None:
        y_max = max(values) + 0.012

    def x_map(step: int) -> float:
        return left + (step - x_min) * (right - left) / max(1, x_max - x_min)

    def y_map(miou: float) -> float:
        return bottom - (miou - y_min) * (bottom - top) / max(1e-9, y_max - y_min)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Border and grid, close to a default matplotlib look.
    draw.rectangle((left, top, right, bottom), outline=(55, 55, 55), width=2)
    y_ticks = nice_ticks(y_min, y_max, count=7)
    for tick in y_ticks:
        y = y_map(tick)
        draw.line((left, y, right, y), fill=(222, 222, 222), width=1)
        label = f"{tick:.3f}"
        tw, th = text_wh(draw, label, FONT_TICK)
        draw.text((left - tw - 10, y - th / 2), label, fill=(28, 28, 28), font=FONT_TICK)

    x_tick_count = 6
    for i in range(x_tick_count):
        value = int(round(x_min + (x_max - x_min) * i / (x_tick_count - 1)))
        x = x_map(value)
        draw.line((x, top, x, bottom), fill=(232, 232, 232), width=1)
        label = str(value)
        tw, th = text_wh(draw, label, FONT_TICK)
        draw.text((x - tw / 2, bottom + 12), label, fill=(28, 28, 28), font=FONT_TICK)

    title_w, _ = text_wh(draw, title, FONT_TITLE)
    draw.text(((width - title_w) / 2, 10), title, fill=(28, 28, 28), font=FONT_TITLE)

    x_label = "Training step"
    tw, th = text_wh(draw, x_label, FONT_AXIS)
    draw.text(((width - tw) / 2, height - 28), x_label, fill=(28, 28, 28), font=FONT_AXIS)

    y_label = "mIoU"
    # Vertical label.
    label_img = Image.new("RGBA", (70, 24), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((0, 0), y_label, fill=(28, 28, 28), font=FONT_AXIS)
    label_img = label_img.rotate(90, expand=True)
    img.paste(label_img, (8, int((top + bottom) / 2 - label_img.height / 2)), label_img)

    # Lines and markers.
    for run in runs:
        coords = [(x_map(point["step"]), y_map(point["miou"])) for point in run["points"]]
        if len(coords) >= 2:
            draw.line(coords, fill=run["color"], width=4)
        for x, y in coords:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=run["color"], outline=run["color"])

    # Red best markers after lines so they stay visible.
    for run in runs:
        best = run["best"]
        if not best:
            continue
        x, y = x_map(best["step"]), y_map(best["miou"])
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(239, 64, 64), outline=(239, 64, 64))

    # Legend box.
    legend_items: list[tuple[str, tuple[int, int, int], str]] = []
    if single_legend:
        legend_items.append(("mIoU_val", runs[0]["color"], "line"))
        legend_items.append(("best in log", (239, 64, 64), "dot"))
    else:
        for run in runs:
            best = run["best"]
            best_text = f"{run['legend']} (best {best['miou']:.4f})" if best else run["legend"]
            legend_items.append((best_text, run["color"], "line"))
        legend_items.append(("best in log/checkpoint", (239, 64, 64), "dot"))

    legend_x, legend_y = left + 12, top + 10
    item_h = 25
    legend_w = max(text_wh(draw, item[0], FONT_LEGEND)[0] for item in legend_items) + 58
    legend_h = len(legend_items) * item_h + 14
    draw.rounded_rectangle(
        (legend_x, legend_y, legend_x + legend_w, legend_y + legend_h),
        radius=2,
        fill=(255, 255, 255),
        outline=(205, 205, 205),
    )
    for idx, (label, color, style) in enumerate(legend_items):
        y = legend_y + 12 + idx * item_h
        if style == "line":
            draw.line((legend_x + 12, y + 7, legend_x + 42, y + 7), fill=color, width=4)
            draw.ellipse((legend_x + 24, y + 2, legend_x + 34, y + 12), fill=color, outline=color)
        else:
            draw.ellipse((legend_x + 22, y, legend_x + 36, y + 14), fill=color, outline=color)
        draw.text((legend_x + 52, y - 3), label, fill=(28, 28, 28), font=FONT_LEGEND)

    img.save(out_path)


def write_points_csv(runs: list[dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "epoch", "step", "miou", "miou_percent", "source"])
        for run in runs:
            for point in run["points"]:
                writer.writerow(
                    [
                        run["label"],
                        point["epoch"],
                        point["step"],
                        f"{point['miou']:.4f}",
                        f"{point['miou'] * 100:.2f}",
                        point["source"],
                    ]
                )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = [load_run(run) for run in RUNS]
    plot_line(
        [runs[0]],
        "PointMixer mIoU over training",
        OUT_DIR / "miou_1993_matplotlib_style.png",
        y_min=-0.005,
        y_max=0.215,
        single_legend=True,
    )
    plot_line(
        runs,
        "PointMixer mIoU over training: 19 vs 17 vs 5",
        OUT_DIR / "comparison_miou_19_17_5_matplotlib_style.png",
        y_min=-0.005,
        y_max=0.215,
        single_legend=False,
    )
    write_points_csv(runs, OUT_DIR / "miou_19_17_5_matplotlib_style_points.csv")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
