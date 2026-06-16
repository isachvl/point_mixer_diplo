from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for name in names:
        p = Path(name)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


NAVY = "#111827"
MUTED = "#64748b"
GRID = "#e5e7eb"
GREEN = "#2f9e44"
BLUE = "#2563eb"
CYAN = "#06b6d4"
ORANGE = "#f97316"
PURPLE = "#7c3aed"
BG = "#f8fafc"


def rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_header(draw, title, subtitle=None, w=1600):
    draw.text((70, 46), title, fill=NAVY, font=font(54, True))
    if subtitle:
        draw.text((72, 115), subtitle, fill=MUTED, font=font(25))
    draw.line((70, 165, w - 70, 165), fill=GRID, width=2)


def save_model_comparison():
    w, h = 1600, 950
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    draw_header(
        d,
        "Размер моделей: PointMixer против близких аналогов",
        "Сравнение по числу параметров; чем короче полоса, тем легче модель.",
        w,
    )

    data = [
        ("PointGroup", 7.700000, "#94a3b8"),
        ("SoftGroup", 30.858000, ORANGE),
        ("Mask3D", 39.617000, PURPLE),
        ("OneFormer3D", 17.515048, CYAN),
        ("Моя PointMixer 19.93", 6.315149, GREEN),
        ("Student 17.83", 4.865435, BLUE),
    ]
    max_v = max(v for _, v, _ in data)
    x0, x1 = 410, 1370
    y = 235
    step = 96
    for i in range(5):
        gx = x0 + (x1 - x0) * i / 4
        d.line((gx, 205, gx, 810), fill=GRID, width=1)
        d.text((gx - 20, 820), f"{max_v * i / 4:.0f}M", fill=MUTED, font=font(20))

    for label, value, color in data:
        d.text((70, y + 8), label, fill=NAVY, font=font(28, True if "Моя" in label or "Student" in label else False))
        bar_w = int((x1 - x0) * value / max_v)
        rounded_rect(d, (x0, y, x0 + bar_w, y + 48), 18, color)
        d.text((x0 + bar_w + 18, y + 6), f"{value:.3f}M", fill=NAVY, font=font(27, True))
        y += step

    rounded_rect(d, (70, 845, 1530, 910), 22, "#ecfdf5", outline="#bbf7d0", width=2)
    d.text(
        (95, 862),
        "Итог: основная PointMixer меньше SoftGroup на 79.535%, Mask3D на 84.059%, OneFormer3D на 63.944%.",
        fill="#166534",
        font=font(27, True),
    )
    im.save(OUT / "defense_model_size_comparison.png", quality=95)


def save_panoptic_metrics():
    w, h = 1600, 900
    im = Image.new("RGB", (w, h), "#0f172a")
    d = ImageDraw.Draw(im)
    d.text((70, 48), "Что показала panoptic-проверка", fill="white", font=font(54, True))
    d.text(
        (72, 117),
        "mIoU отвечает за классы точек, а PQ/SQ/RQ/AP — за разделение объектов на экземпляры.",
        fill="#cbd5e1",
        font=font(25),
    )

    cards = [
        ("mIoU", "19.93%", "лучший val при обучении", GREEN),
        ("full-scene mIoU", "17.32%", "строгий прогон по сценам", BLUE),
        ("PQ", "11.20%", "общее panoptic-качество", ORANGE),
        ("SQ", "74.42%", "форма найденных объектов", "#22c55e"),
        ("RQ", "15.05%", "полнота и точность поиска", "#ef4444"),
        ("AP50", "7.43%", "качество instance-обнаружения", PURPLE),
    ]
    x, y = 70, 210
    card_w, card_h = 465, 185
    for idx, (name, value, desc, color) in enumerate(cards):
        cx = x + (idx % 3) * (card_w + 35)
        cy = y + (idx // 3) * (card_h + 35)
        rounded_rect(d, (cx, cy, cx + card_w, cy + card_h), 28, "#1e293b", outline="#334155", width=2)
        d.text((cx + 34, cy + 26), name, fill="#cbd5e1", font=font(27, True))
        d.text((cx + 34, cy + 66), value, fill=color, font=font(56, True))
        d.text((cx + 34, cy + 137), desc, fill="#e2e8f0", font=font(22))

    rounded_rect(d, (70, 680, 1530, 820), 28, "#172554", outline="#2563eb", width=2)
    d.text((105, 705), "Главный вывод", fill="#bfdbfe", font=font(28, True))
    d.text(
        (105, 748),
        "Если модель уже нашла объект, его геометрия часто совпадает хорошо: SQ = 74.42%.",
        fill="white",
        font=font(27, True),
    )
    d.text(
        (105, 786),
        "Слабое место — не форма маски, а пропуски и лишние инстансы: это видно по RQ и AP50.",
        fill="#cbd5e1",
        font=font(24),
    )
    im.save(OUT / "defense_panoptic_metrics.png", quality=95)


def save_distillation():
    w, h = 1600, 880
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    draw_header(d, "Дистилляция: что стало меньше и что потеряли", "Сравнение основной модели 19.93% и student-модели 17.83%.", w)

    main = {"label": "Основная модель", "miou": 19.93, "params": 6.315149, "mb": 25.261196, "color": GREEN}
    stu = {"label": "Student", "miou": 17.83, "params": 4.865435, "mb": 19.462340, "color": BLUE}
    diff_params = main["params"] - stu["params"]
    diff_pct = diff_params / main["params"] * 100
    loss = main["miou"] - stu["miou"]

    def model_box(x, y, m):
        rounded_rect(d, (x, y, x + 500, y + 310), 34, "white", outline="#dbeafe", width=3)
        d.text((x + 34, y + 34), m["label"], fill=NAVY, font=font(38, True))
        d.text((x + 34, y + 105), f"{m['params']:.6f}M", fill=m["color"], font=font(58, True))
        d.text((x + 36, y + 174), "параметров", fill=MUTED, font=font(24))
        d.text((x + 34, y + 220), f"mIoU {m['miou']:.2f}%  ·  fp32 {m['mb']:.3f} MB", fill=NAVY, font=font(26, True))

    model_box(95, 235, main)
    model_box(1005, 235, stu)

    d.line((650, 390, 950, 390), fill=NAVY, width=7)
    d.polygon([(950, 390), (912, 368), (912, 412)], fill=NAVY)
    d.text((665, 318), "обучение через teacher", fill=MUTED, font=font(24, True))
    d.text((675, 430), "мягкие вероятности + разметка", fill=MUTED, font=font(22))

    rounded_rect(d, (180, 620, 1420, 790), 32, "#fff7ed", outline="#fed7aa", width=2)
    d.text((220, 646), "Итог сжатия", fill="#9a3412", font=font(31, True))
    d.text((220, 694), f"−{diff_params:.6f}M параметров  ·  −{diff_pct:.3f}% размера модели", fill=NAVY, font=font(34, True))
    d.text((220, 742), f"Потеря качества: −{loss:.2f} процентного пункта mIoU", fill=MUTED, font=font(27))
    im.save(OUT / "defense_distillation_comparison.png", quality=95)


def crop_fit(path, size):
    src = Image.open(path).convert("RGB")
    return ImageOps.fit(src, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def save_story_strip():
    w, h = 1600, 980
    im = Image.new("RGB", (w, h), "#020617")
    d = ImageDraw.Draw(im)
    d.text((70, 44), "От облака точек к разметке помещения", fill="white", font=font(54, True))
    d.text((72, 115), "Три шага проекта: данные, semantic-классы, panoptic-экземпляры.", fill="#cbd5e1", font=font(25))
    imgs = [
        ("ScanNet++ сцена", ROOT / "docs/images/02_scannetpp_clean_scene.png"),
        ("Semantic", ROOT / "docs/images/03_scannetpp_semantic_example.png"),
        ("Panoptic", ROOT / "docs/images/04_scannetpp_panoptic_example.png"),
    ]
    x = 70
    for title, p in imgs:
        rounded_rect(d, (x, 200, x + 460, 800), 28, "#0f172a", outline="#334155", width=2)
        img = crop_fit(p, (420, 420))
        im.paste(img, (x + 20, 240))
        d.text((x + 24, 685), title, fill="white", font=font(31, True))
        x += 520
    d.text((90, 840), "Идея простая:", fill="#e2e8f0", font=font(26, True))
    d.text(
        (295, 840),
        "сначала модель определяет класс каждой точки, затем группирует точки одного объекта.",
        fill="#e2e8f0",
        font=font(26, True),
    )
    d.text(
        (90, 890),
        "Так получается не просто цветная карта комнаты, а разметка объектов внутри неё.",
        fill="#94a3b8",
        font=font(24),
    )
    im.save(OUT / "defense_scene_story.png", quality=95)


if __name__ == "__main__":
    save_model_comparison()
    save_panoptic_metrics()
    save_distillation()
    save_story_strip()
    print("Wrote defense assets to", OUT)
