"""Generate assets/live-captions.ico (a simple caption-bubble glyph). Needs Pillow."""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "assets" / "live-captions.ico"


def render(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 256
    # dark rounded caption box
    d.rounded_rectangle([16 * s, 40 * s, 240 * s, 184 * s], radius=36 * s, fill=(16, 16, 16, 255),
                        outline=(255, 230, 128, 255), width=max(1, int(10 * s)))
    # speech-bubble tail
    d.polygon([(72 * s, 180 * s), (104 * s, 180 * s), (64 * s, 228 * s)], fill=(16, 16, 16, 255))
    # text lines
    for y, w, col in ((84, 150, (255, 255, 255, 255)), (120, 110, (255, 230, 128, 255))):
        d.rounded_rectangle([48 * s, y * s, (48 + w) * s, (y + 22) * s], radius=11 * s, fill=col)
    return img


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sizes = [16, 24, 32, 48, 64, 128, 256]
    base = render(256)
    base.save(OUT, format="ICO", sizes=[(n, n) for n in sizes])
    base.save(OUT.with_suffix(".png"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
