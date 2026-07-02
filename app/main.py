"""
main.py
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.pipeline import QRPipeline
from app.storage import upload_image, save_job

executor = ThreadPoolExecutor(max_workers=1)
pipeline: QRPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("[startup] Loading QR pipeline...")
    pipeline = QRPipeline(low_vram_mode=True)
    print("[startup] Pipeline ready.")
    yield


app = FastAPI(title="AI QR Generator", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    url: str
    prompt: str
    negative_prompt: str = "ugly, disfigured, low quality, blurry, nsfw"
    controlnet_conditioning_scale: float = 1.3
    num_inference_steps: int = 30
    seed: int | None = None


class GenerateResponse(BaseModel):
    job_id: str
    image_url: str
    elapsed_seconds: float
    peak_vram_mb: float | None


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    job_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()

    def _run():
        return pipeline.generate(
            prompt=req.prompt,
            qr_data=req.url,
            negative_prompt=req.negative_prompt,
            controlnet_conditioning_scale=req.controlnet_conditioning_scale,
            num_inference_steps=req.num_inference_steps,
            seed=req.seed,
        )

    try:
        image, meta = await loop.run_in_executor(executor, _run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Upload to Supabase
    image_url = upload_image(image, job_id)
    save_job(
        job_id=job_id,
        url=req.url,
        prompt=req.prompt,
        image_url=image_url,
        elapsed=meta["elapsed_seconds"],
        vram=meta.get("peak_vram_mb"),
    )

    return GenerateResponse(
        job_id=job_id,
        image_url=image_url,
        elapsed_seconds=meta["elapsed_seconds"],
        peak_vram_mb=meta.get("peak_vram_mb"),
    )