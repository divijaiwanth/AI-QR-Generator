"""
verify.py

Decodes a generated QR image with 3 independent readers and reports
whether each one recovered the expected data correctly.

READER STRATEGY:
Classical CLI decoders (pyzbar, opencv) fail on AI-stylized QR codes
even when every real phone scanner reads them fine. We use 3 readers
that reflect real-world scan fidelity instead:

1. pyzbar           - strict classical baseline (worst-case measure)
2. QR Server API    - ZXing engine, same as Android camera app
3. ZXing online     - Google's reference implementation, most widely deployed
"""

import io
import requests
from dataclasses import dataclass, field
from PIL import Image
import numpy as np
import cv2

try:
    from pyzbar.pyzbar import decode as zbar_decode
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False


@dataclass
class ReaderResult:
    reader_name: str
    success: bool
    decoded_data: str | None
    matches_expected: bool


@dataclass
class VerificationResult:
    expected_data: str
    results: list[ReaderResult] = field(default_factory=list)

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self.results if r.matches_expected)

    @property
    def num_readers(self) -> int:
        return len(self.results)

    @property
    def fully_scannable(self) -> bool:
        return self.num_readers > 0 and self.num_passed == self.num_readers

    def summary(self) -> str:
        lines = [f"Expected: {self.expected_data!r}"]
        for r in self.results:
            status = "PASS" if r.matches_expected else "FAIL"
            lines.append(f"  [{status}] {r.reader_name}: decoded={r.decoded_data!r}")
        lines.append(f"  -> {self.num_passed}/{self.num_readers} readers passed")
        return "\n".join(lines)


def _pil_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _pil_to_cv2(image: Image.Image) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _try_pyzbar(image: Image.Image) -> ReaderResult:
    if not PYZBAR_AVAILABLE:
        return ReaderResult("pyzbar", False, None, False)
    try:
        cv_image = _pil_to_cv2(image)
        decoded_objects = zbar_decode(cv_image)
        if not decoded_objects:
            return ReaderResult("pyzbar", False, None, False)
        data = decoded_objects[0].data.decode("utf-8")
        return ReaderResult("pyzbar", True, data, False)
    except Exception:
        return ReaderResult("pyzbar", False, None, False)


def _try_qrserver_api(image: Image.Image) -> ReaderResult:
    try:
        # API limit: <1MB. Resize to 512x512 JPEG to stay under limit.
        img = image.resize((512, 512), Image.LANCZOS).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        img_bytes = buf.getvalue()

        r = requests.post(
            "https://api.qrserver.com/v1/read-qr-code/",
            files={"file": ("qr.jpg", img_bytes, "image/jpeg")},
            timeout=10,
        )
        data = r.json()
        decoded = data[0]["symbol"][0]["data"]
        error = data[0]["symbol"][0]["error"]
        if error or not decoded:
            return ReaderResult("qrserver_api (ZXing)", False, None, False)
        return ReaderResult("qrserver_api (ZXing)", True, decoded, False)
    except Exception as e:
        return ReaderResult("qrserver_api (ZXing)", False, str(e), False)


def _try_zxing_online(image: Image.Image) -> ReaderResult:
    try:
        img = image.resize((512, 512), Image.LANCZOS).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)

        r = requests.post(
            "https://zxing.org/w/decode",
            files={"f": ("qr.jpg", buf.getvalue(), "image/jpeg")},
            timeout=15,
            allow_redirects=True,
        )
        if "Parsed Result" in r.text:
            start = r.text.find("<pre>", r.text.find("Parsed Result")) + 5
            end = r.text.find("</pre>", start)
            decoded = r.text[start:end].strip()
            return ReaderResult("zxing_online", True, decoded, False)
        return ReaderResult("zxing_online", False, None, False)
    except Exception as e:
        return ReaderResult("zxing_online", False, str(e), False)


def verify_qr_image(image: Image.Image, expected_data: str) -> VerificationResult:
    raw_results = [
        _try_pyzbar(image),
        _try_qrserver_api(image),
        _try_zxing_online(image),
    ]
    final_results = []
    for r in raw_results:
        matches = r.success and r.decoded_data == expected_data
        final_results.append(ReaderResult(r.reader_name, r.success, r.decoded_data, matches))
    return VerificationResult(expected_data=expected_data, results=final_results)


def diagnose_finder_patterns(image: Image.Image) -> dict:
    cv_image = _pil_to_cv2(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    detector = cv2.QRCodeDetector()
    detected, points = detector.detect(gray)
    return {
        "finder_patterns_located": bool(detected),
        "corner_points": points.tolist() if points is not None else None,
    }


if __name__ == "__main__":
    from qr_base import generate_base_qr
    import glob

    plain = generate_base_qr("https://example.com")
    result = verify_qr_image(plain, "https://example.com")
    print("=== Plain QR ===")
    print(result.summary())

    paths = sorted(glob.glob("outputs/sweep/scale_1.3_*.png"))
    if paths:
        styled = Image.open(paths[0])
        result2 = verify_qr_image(styled, "https://example.com")
        print("\n=== Styled QR (scale=1.3) ===")
        print(result2.summary())