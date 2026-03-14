import shutil
from pathlib import Path

import pillow_heif
from PIL import Image, ImageDraw, ImageFont, ImageOps

pillow_heif.register_heif_opener()

PADDING = 40
BASE_FONT_SIZE = 52        # larger, Pacifico is a display font
BASE_SUBTITLE_SIZE = 28    # city/country line
BASE_WIDTH = 1000
LINE_SPACING_RATIO = 0.4

# Pacifico bundled in the project
_HERE = Path(__file__).parent
GREAT_VIBES_PATH = _HERE / "fonts" / "GreatVibes-Regular.ttf"


def annotate_image(
    image_path: Path,
    output_path: Path,
    location_data: dict,
    placement_data: dict,
) -> None:
    confidence = location_data.get("confidence", "low")
    popular_name = location_data.get("popular_name") or location_data.get("landmark")

    if confidence == "low" and not popular_name:
        _copy_file(image_path, output_path)
        return

    try:
        with Image.open(image_path) as test:
            test.verify()
    except Exception:
        _copy_file(image_path, output_path)
        return

    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode == "P":
            img = img.convert("RGBA")
        elif img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        width, height = img.size
        font_main = _load_pacifico(int(BASE_FONT_SIZE * width / BASE_WIDTH), min_size=24)
        font_sub  = _load_pacifico(int(BASE_SUBTITLE_SIZE * width / BASE_WIDTH), min_size=14)
        lines = _build_text_lines(location_data)

        if not lines:
            _copy_file(image_path, output_path)
            return

        placement = _resolve_placement(placement_data, confidence)

        base = img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        _draw_text(draw, base, lines, placement, font_main, font_sub, (width, height))

        composited = Image.alpha_composite(base, overlay)
        is_png = output_path.suffix.lower() == ".png"
        final = composited if is_png else composited.convert("RGB")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final.save(str(output_path), format="PNG" if is_png else "JPEG", quality=95)

    _copy_exif(image_path, output_path)


# ── Font loading ──────────────────────────────────────────────────────────────

def _load_pacifico(size: int, min_size: int = 12) -> ImageFont.FreeTypeFont:
    size = max(size, min_size)
    if GREAT_VIBES_PATH.exists():
        try:
            return ImageFont.truetype(str(GREAT_VIBES_PATH), size)
        except (IOError, OSError):
            pass
    # Fallbacks
    for fp in [
        "C:/Windows/Fonts/georgia.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(fp, size)
        except (IOError, OSError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ── Text building ─────────────────────────────────────────────────────────────

def _build_text_lines(location_data: dict) -> list[str]:
    popular_name = location_data.get("popular_name") or location_data.get("landmark")
    return [popular_name] if popular_name else []


def _resolve_placement(placement_data: dict, confidence: str) -> str:
    if confidence == "low":
        return "bottom-center"
    rec = placement_data.get("recommendation", "bottom-center")
    valid = {"top-left", "top-right", "bottom-left", "bottom-right", "top-center", "bottom-center"}
    return rec if rec in valid else "bottom-center"


# ── Contrast detection ────────────────────────────────────────────────────────

def _sample_brightness(img_rgba: Image.Image, x: int, y: int, w: int, h: int) -> float:
    """Return average perceived brightness (0–255) of the given region."""
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_rgba.width, x + w)
    y1 = min(img_rgba.height, y + h)
    if x1 <= x0 or y1 <= y0:
        return 128.0
    region = img_rgba.crop((x0, y0, x1, y1)).convert("RGB")
    pixels = list(region.getdata())
    if not pixels:
        return 128.0
    return sum(0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels) / len(pixels)


def _text_color(brightness: float) -> tuple[int, int, int, int]:
    """White on dark, near-black on light."""
    return (255, 255, 255, 255) if brightness < 148 else (20, 20, 20, 255)


def _shadow_color(brightness: float) -> tuple[int, int, int, int]:
    """Shadow contrasts with the text colour."""
    return (0, 0, 0, 180) if brightness < 148 else (255, 255, 255, 180)


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_text(
    draw: ImageDraw.ImageDraw,
    base_img: Image.Image,
    lines: list[str],
    placement: str,
    font_main: ImageFont.FreeTypeFont,
    font_sub: ImageFont.FreeTypeFont,
    image_size: tuple[int, int],
) -> None:
    width, height = image_size
    fonts = [font_main] + [font_sub] * (len(lines) - 1)

    # Measure each line
    bboxes   = [draw.textbbox((0, 0), line, font=fonts[i]) for i, line in enumerate(lines)]
    lwidths  = [bb[2] - bb[0] for bb in bboxes]
    lheights = [bb[3] - bb[1] for bb in bboxes]
    spacing  = max(8, int(font_main.size * LINE_SPACING_RATIO * 0.5))
    total_h  = sum(lheights) + spacing * (len(lines) - 1)
    max_w    = max(lwidths)

    # Vertical position
    if "top" in placement:
        start_y = PADDING
    else:
        start_y = height - total_h - PADDING

    # Horizontal anchor for the block
    if "left" in placement:
        anchor_x = PADDING
        def line_x(lw): return anchor_x
    elif "right" in placement:
        anchor_x = width - max_w - PADDING
        def line_x(lw): return width - lw - PADDING
    else:  # center
        def line_x(lw): return (width - lw) // 2

    # Sample background brightness across the whole text block
    block_x = line_x(max_w)
    brightness = _sample_brightness(base_img, block_x, start_y, max_w, total_h)
    text_fill   = _text_color(brightness)
    shadow_fill = _shadow_color(brightness)

    # Shadow offset scales with font size
    shadow_offset = max(2, font_main.size // 20)

    text_y = start_y
    for i, line in enumerate(lines):
        lw = lwidths[i]
        lh = lheights[i]
        tx = line_x(lw)

        # Draw shadow
        for dx, dy in [
            (-shadow_offset, -shadow_offset), (0, -shadow_offset), (shadow_offset, -shadow_offset),
            (-shadow_offset, 0),                                     (shadow_offset, 0),
            (-shadow_offset,  shadow_offset), (0,  shadow_offset), (shadow_offset,  shadow_offset),
        ]:
            draw.text((tx + dx, text_y + dy), line, font=fonts[i], fill=shadow_fill)

        # Draw text
        draw.text((tx, text_y), line, font=fonts[i], fill=text_fill)
        text_y += lh + spacing


# ── Helpers ───────────────────────────────────────────────────────────────────

def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".heic" and dst.suffix.lower() in (".jpg", ".jpeg"):
        try:
            heif_file = pillow_heif.open_heif(str(src))
            img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
            img.convert("RGB").save(str(dst), format="JPEG", quality=95)
            return
        except Exception:
            pass
        try:
            # Fallback: file may be JPEG with wrong extension
            img = Image.open(src)
            img.load()
            img.convert("RGB").save(str(dst), format="JPEG", quality=95)
            return
        except Exception:
            pass
    shutil.copy2(str(src), str(dst))


def _copy_exif(source_path: Path, output_path: Path) -> None:
    try:
        if source_path.suffix.lower() == ".heic":
            try:
                heif_file = pillow_heif.open_heif(str(source_path))
                exif_bytes = heif_file.info.get("exif")
            except Exception:
                exif_bytes = None
        else:
            with Image.open(source_path) as src:
                exif_bytes = src.getexif().tobytes()
        if exif_bytes:
            with Image.open(output_path) as out:
                out.save(str(output_path), format="JPEG", quality=95, exif=exif_bytes)
    except Exception:
        pass
