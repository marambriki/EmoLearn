"""Extrait et agrandit les logos ESCT / ISAMM / UMA depuis la page de garde modèle."""
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps

SRC = Path(
    r"C:\Users\Lenovo\.cursor\projects\c-Users-Lenovo-EmoLearn\assets"
    r"\c__Users_Lenovo_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-cd130022-e6fe-477f-bc56-b4653bc6c005.png"
)
OUT = Path(__file__).resolve().parent
OV = OUT.parent / "overleaf" / "logos"

# Zones calibrées sur la capture 672x815
CROPS = {
    "logo_esct.png": (68, 52, 128, 128),
    "logo_isamm.png": (122, 52, 195, 128),
    "logo_uma.png": (495, 45, 655, 135),
}


def trim_whitespace(img: Image.Image, pad: int = 8) -> Image.Image:
    gray = img.convert("L")
    bw = gray.point(lambda p: 0 if p < 245 else 255)
    inv = ImageOps.invert(bw)
    bbox = inv.getbbox()
    if not bbox:
        return img
    l, t, r, b = bbox
    l = max(0, l - pad)
    t = max(0, t - pad)
    r = min(img.width, r + pad)
    b = min(img.height, b + pad)
    return img.crop((l, t, r, b))


def upscale(img: Image.Image, scale: int = 8) -> Image.Image:
    w, h = img.size
    big = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
    return big.filter(ImageFilter.SHARPEN)


def main() -> None:
    OV.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        raise SystemExit(f"Source introuvable: {SRC}")

    img = Image.open(SRC).convert("RGB")
    print("SRC", SRC, img.size)
    for name, box in CROPS.items():
        crop = trim_whitespace(img.crop(box), pad=4)
        hq = upscale(crop, scale=8)
        side = max(hq.size) + 40
        canvas = Image.new("RGB", (side, side), "white")
        canvas.paste(hq, ((side - hq.width) // 2, (side - hq.height) // 2))
        canvas.save(OUT / name, optimize=True)
        (OV / name).write_bytes((OUT / name).read_bytes())
        print(f"OK {name} {canvas.size}")


if __name__ == "__main__":
    main()
