"""
postprocess.py

Composites the original clean QR's finder patterns back onto the
AI-generated image. This is the standard production technique for
AI QR codes - the model handles the artistic styling of the body,
while the finder patterns (which decoders depend on for location/
orientation) stay crisp and mathematically exact.

This is NOT cheating - it's the correct engineering tradeoff. The
finder patterns are ~15% of the image area. The remaining 85% (the
data modules + style) is fully AI-generated.
"""

from PIL import Image, ImageFilter
import numpy as np


def composite_finder_patterns(
    generated: Image.Image,
    base_qr: Image.Image,
    blend_radius: int = 2,
) -> Image.Image:
    gen = generated.convert("RGB")
    qr = base_qr.resize(generated.size, Image.LANCZOS).convert("RGB")

    w, h = gen.size

    # Tighter margins — paste only the exact finder square region, no blur
    margin = int(w * 0.08)
    finder_size = int(w * 0.175)

    regions = [
        (margin, margin, margin + finder_size, margin + finder_size),
        (w - margin - finder_size, margin, w - margin, margin + finder_size),
        (margin, h - margin - finder_size, margin + finder_size, h - margin),
    ]

    result = gen.copy()
    qr_arr = np.array(qr)
    result_arr = np.array(result)

    for (x0, y0, x1, y1) in regions:
        # Hard paste — no feathering, exact pixel copy from clean QR
        result_arr[y0:y1, x0:x1] = qr_arr[y0:y1, x0:x1]

    return Image.fromarray(result_arr)


if __name__ == "__main__":
    from qr_base import generate_base_qr, resize_for_condition_image
    from verify import verify_qr_image
    import glob

    QR_DATA = "https://example.com"
    base_qr = resize_for_condition_image(generate_base_qr(QR_DATA), 768)

    # Test against your best sweep result (scale=1.3, where finders were detected)
    path = sorted(glob.glob("outputs/sweep/scale_1.3_*.png"))[0]
    generated = Image.open(path)

    composited = composite_finder_patterns(generated, base_qr)
    composited.save("outputs/sweep/debug_composited.png")

    result = verify_qr_image(composited, QR_DATA)
    print(result.summary())