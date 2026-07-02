"""
sweep.py

Generates QR codes at several controlnet_conditioning_scale values and
verifies each one with the 3-reader harness from verify.py.

Run this, read the printed table, pick the scale value with the best
readers_passed / finder_patterns_located result, and use that as your
final tuned parameter.
"""

import csv
import os
from datetime import datetime

from pipeline import QRPipeline
from verify import verify_qr_image, diagnose_finder_patterns

OUTPUT_DIR = "outputs/sweep"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONDITIONING_SCALES = [1.0, 1.3, 1.5, 1.8]

PROMPT = "a serene japanese garden with koi pond, cherry blossoms, highly detailed, soft lighting"
QR_DATA = "https://example.com"
SEED = 42


def run_sweep():
    pipeline = QRPipeline(low_vram_mode=True)

    rows = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for scale in CONDITIONING_SCALES:
        print(f"\n=== scale={scale} ===")
        image, meta = pipeline.generate(
            prompt=PROMPT,
            qr_data=QR_DATA,
            controlnet_conditioning_scale=scale,
            seed=SEED,
        )

        filename = f"{OUTPUT_DIR}/scale_{scale}_{timestamp}.png"
        image.save(filename)

        result = verify_qr_image(image, QR_DATA)
        diag = diagnose_finder_patterns(image)
        print(result.summary())
        print(f"  finder_patterns_located: {diag['finder_patterns_located']}")

        rows.append({
            "conditioning_scale": scale,
            "readers_passed": result.num_passed,
            "readers_total": result.num_readers,
            "fully_scannable": result.fully_scannable,
            "finder_patterns_located": diag["finder_patterns_located"],
            "elapsed_seconds": meta["elapsed_seconds"],
            "peak_vram_mb": meta.get("peak_vram_mb"),
            "image_path": filename,
        })

    csv_path = f"{OUTPUT_DIR}/sweep_results_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n\n=== SWEEP COMPLETE ===")
    print(f"Results saved to {csv_path}")
    print(f"{'scale':>6} | {'passed':>8} | {'finders':>8} | {'time(s)':>8}")
    for r in rows:
        print(f"{r['conditioning_scale']:>6} | {r['readers_passed']}/{r['readers_total']:>6} | "
              f"{str(r['finder_patterns_located']):>8} | {r['elapsed_seconds']:>8}")


if __name__ == "__main__":
    run_sweep()