"""Fond noir remove.bg -> blanc ; texte blanc -> sombre (lisible sur page blanche)."""
from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
FILES = {
    "ESCT-removebg-preview.png": "logo_esct.png",
    "ISAM-removebg-preview.png": "logo_isamm.png",
    "uma-removebg-preview.png": "logo_uma.png",
}


def is_bg(r, g, b, a, thr=45):
    return a < 15 or (r < thr and g < thr and b < thr)


def flood_background(arr, thr=45):
    h, w = arr.shape[:2]
    bg = np.zeros((h, w), dtype=bool)
    q = deque()
    for y, x in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        q.append((y, x))
    # aussi le bord
    for x in range(0, w, 4):
        q.append((0, x))
        q.append((h - 1, x))
    for y in range(0, h, 4):
        q.append((y, 0))
        q.append((y, w - 1))

    while q:
        y, x = q.popleft()
        if y < 0 or y >= h or x < 0 or x >= w or bg[y, x]:
            continue
        r, g, b, a = arr[y, x]
        if not is_bg(int(r), int(g), int(b), int(a), thr):
            continue
        bg[y, x] = True
        q.extend([(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)])
    return bg


def convert(src: Path, dst: Path) -> None:
    im = Image.open(src).convert("RGBA")
    arr = np.array(im).copy()
    bg = flood_background(arr, thr=48)

    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    a = arr[:, :, 3]

    # texte / traits clairs sur fond noir -> sombre
    light = (~bg) & (r > 200) & (g > 200) & (b > 200) & (a > 40)
    arr[light, 0] = 30
    arr[light, 1] = 30
    arr[light, 2] = 30
    arr[light, 3] = 255

    # fond -> blanc
    arr[bg, 0] = 255
    arr[bg, 1] = 255
    arr[bg, 2] = 255
    arr[bg, 3] = 255

    out = Image.fromarray(arr, mode="RGBA")
    canvas = Image.new("RGB", out.size, (255, 255, 255))
    canvas.paste(out, mask=out.split()[3])
    canvas.save(dst, optimize=True)
    print(f"OK {src.name} -> {dst.name} {canvas.size}")


def main() -> None:
    ov = OUT.parent / "overleaf" / "logos"
    ov.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in FILES.items():
        src = OUT / src_name
        if not src.exists():
            print("missing", src)
            continue
        dst = OUT / dst_name
        convert(src, dst)
        (ov / dst_name).write_bytes(dst.read_bytes())


if __name__ == "__main__":
    main()
