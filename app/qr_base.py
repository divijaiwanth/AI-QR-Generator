"""
qr_base.py

Generates the "base" QR code image that ControlNet will use as a structural
guide. This is NOT the final styled output - it's the plain black/white QR
that tells ControlNet "this is the pattern you must preserve."

WHY ERROR_CORRECT_H MATTERS:
QR codes have 4 error-correction levels (L=7%, M=15%, Q=25%, H=30%).
H means up to 30% of the code can be visually "damaged" or obscured and it
will STILL scan correctly. Since the diffusion model is going to stylize
and distort the QR pattern, we need maximum tolerance for that distortion.
This single setting is one of the biggest levers for hitting high scan
fidelity - more important than almost any generation parameter.
"""

import qrcode
from qrcode.constants import ERROR_CORRECT_H
from PIL import Image


def generate_base_qr(
    data: str,
    box_size: int = 16,
    border: int = 4,
) -> Image.Image:
    """
    Generate a high-error-correction QR code as a PIL Image.

    Args:
        data: The URL or text the QR code should encode.
        box_size: Pixel size of each QR "module" (box). Larger = higher res
                  base image, which gives the diffusion model more detail
                  to work with before downscaling/upscaling.
        border: Number of quiet-zone modules around the QR code. QR spec
                requires at least 4 - going lower hurts scan reliability
                because scanners use the border to detect the code's edges.

    Returns:
        PIL Image in RGB mode (RGB, not L/1-bit, because the diffusion
        pipeline expects 3-channel input).
    """
    qr = qrcode.QRCode(
        version=None,  # None = auto-select smallest version that fits data
        error_correction=ERROR_CORRECT_H,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    return img.convert("RGB")


def resize_for_condition_image(input_image: Image.Image, resolution: int = 768) -> Image.Image:
    """
    Resize an image so its shorter side equals `resolution`, with both
    dimensions rounded to the nearest multiple of 64.

    WHY MULTIPLE OF 64:
    Stable Diffusion's VAE downsamples by a factor of 8, and the UNet
    operates on latents that need clean downsampling through several
    resolution halvings. Non-multiple-of-64 dimensions cause shape
    mismatches or silent quality degradation. This is a standard SD
    gotcha, not specific to QR generation.

    WHY 768:
    The ControlNet QR model card recommends 768 as the sweet spot -
    enough resolution for fine QR module detail to survive generation,
    without the proportionally higher VRAM/time cost of 1024.
    """
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H, W = H * k, W * k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    return input_image.resize((W, H), resample=Image.LANCZOS)


if __name__ == "__main__":
    # Quick manual check - run this file directly to sanity-check QR generation
    # before touching the (much slower, much heavier) diffusion pipeline.
    img = generate_base_qr("https://example.com/test")
    img = resize_for_condition_image(img, 768)
    img.save("/home/claude/ai-qr-generator/outputs_test_base_qr.png")
    print(f"Generated base QR: {img.size}, mode={img.mode}")
