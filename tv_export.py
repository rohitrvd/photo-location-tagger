import sys
from pathlib import Path

import pillow_heif
from PIL import Image, ImageFilter, ImageOps

pillow_heif.register_heif_opener()

TV_SIZE = (1920, 1080)
BLUR_RADIUS = 40
BACKGROUND_DIM = 0.8       # 1.0 = no dimming; slightly darker blur keeps focus on the photo
JPEG_QUALITY = 92
EXTENSIONS = {".jpg", ".jpeg", ".heic"}


def make_tv_image(src: Image.Image, size: tuple[int, int] = TV_SIZE) -> Image.Image:
    """Sharp photo fitted in the centre, over a blurred, enlarged copy that fills the frame."""
    tw, th = size
    img = ImageOps.exif_transpose(src).convert("RGB")
    w, h = img.size

    # Background: scale to cover the frame, centre-crop, blur
    cover = max(tw / w, th / h)
    bg = img.resize((round(w * cover), round(h * cover)), Image.LANCZOS)
    left, top = (bg.width - tw) // 2, (bg.height - th) // 2
    bg = bg.crop((left, top, left + tw, top + th)).filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
    bg = Image.eval(bg, lambda v: int(v * BACKGROUND_DIM))

    # Foreground: scale to fit inside the frame
    fit = min(tw / w, th / h)
    fg = img.resize((round(w * fit), round(h * fit)), Image.LANCZOS)
    bg.paste(fg, ((tw - fg.width) // 2, (th - fg.height) // 2))
    return bg


def export_folder(src_dir: Path, out_dir: Path) -> tuple[int, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    done, failed = 0, []
    for path in sorted(src_dir.iterdir()):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            with Image.open(path) as src:
                make_tv_image(src).save(out_dir / (path.stem + ".jpg"), quality=JPEG_QUALITY)
            done += 1
        except Exception as e:
            failed.append(f"{path.name}: {e}")
    return done, failed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python tv_export.py <photos_folder> [output_folder]")
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src / "tv"
    count, errors = export_folder(src, out)
    print(f"Exported {count} images to {out}")
    for err in errors:
        print("FAILED:", err)
