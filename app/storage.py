"""
storage.py
Handles image upload to Supabase Storage and job metadata to Postgres.
"""

import io
import os
import uuid
from PIL import Image
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
BUCKET = "qr-codes"

def get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def upload_image(image: Image.Image, job_id: str) -> str:
    """Upload PIL image to Supabase Storage, return public URL."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    client = get_client()
    path = f"{job_id}.png"

    client.storage.from_(BUCKET).upload(
        path=path,
        file=buf.getvalue(),
        file_options={"content-type": "image/png"},
    )

    res = client.storage.from_(BUCKET).get_public_url(path)
    return res


def save_job(job_id: str, url: str, prompt: str, image_url: str,
             elapsed: float, vram: float | None):
    """Insert a completed job record into qr_jobs table."""
    client = get_client()
    client.table("qr_jobs").insert({
        "id": job_id,
        "url": url,
        "prompt": prompt,
        "image_url": image_url,
        "elapsed_seconds": elapsed,
        "peak_vram_mb": vram,
    }).execute()