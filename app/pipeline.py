"""
pipeline.py

The core AI pipeline: loads SD1.5 + the QR ControlNet, and generates a
styled, scannable QR code from a prompt + target URL.

Uses monster-labs' QR ControlNet, which is purpose-trained for scan
reliability (vs. general-purpose QR ControlNets that need heavy manual
tuning of strength/conditioning_scale to become reliably scannable).

THIS FILE IS DESIGNED TO RUN ON AN 8GB VRAM GPU. Every choice below that
mentions "memory" exists because of that constraint.
"""

import time
import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
)

from app.qr_base import generate_base_qr, resize_for_condition_image

CONTROLNET_MODEL_ID = "monster-labs/control_v1p_sd15_qrcode_monster"
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"


class QRPipeline:
    """
    Wraps the ControlNet + SD1.5 pipeline as a singleton-style object.
    Load once per process, reuse for every generation request.
    """

    def __init__(self, device: str = "cuda", low_vram_mode: bool = True):
        self.device = device
        self.low_vram_mode = low_vram_mode
        self.pipe = self._load_pipeline()

    def _load_pipeline(self) -> StableDiffusionControlNetPipeline:
        print(f"[QRPipeline] Loading ControlNet from {CONTROLNET_MODEL_ID} ...")
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL_ID,
            torch_dtype=torch.float16,  # MEMORY TRICK #1: fp16 instead of fp32.
        )

        print(f"[QRPipeline] Loading base SD1.5 pipeline from {BASE_MODEL_ID} ...")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        )

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        if self.low_vram_mode:
            # MEMORY TRICK #2: CPU offloading - keeps model in CPU RAM,
            # moves submodules to GPU only when needed.
            pipe.enable_model_cpu_offload()

            # MEMORY TRICK #3: attention slicing - lowers peak memory
            # during attention computation.
            pipe.enable_attention_slicing()

            # MEMORY TRICK #4: VAE slicing/tiling - lowers peak memory
            # during the final latent->pixel decode step.
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        else:
            pipe.to(self.device)

        try:
            # MEMORY TRICK #5: xFormers memory-efficient attention kernel.
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[QRPipeline] xformers not available, continuing without it: {e}")

        return pipe

    def generate(
        self,
        prompt: str,
        qr_data: str,
        negative_prompt: str = "ugly, disfigured, low quality, blurry, nsfw",
        controlnet_conditioning_scale: float = 2.5,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        seed: int | None = None,
    ) -> tuple[Image.Image, dict]:
        """
        Generate a styled QR code via plain txt2img ControlNet.

        controlnet_conditioning_scale (typically 1.0-2.0):
            How strongly the QR pattern is enforced. Higher = more
            scannable, lower = more stylized but riskier to scan. This is
            the main lever for hitting reliable scan fidelity.

        Returns:
            (generated PIL image, metadata dict with timing/VRAM info)
        """
        if seed is not None:
            generator = torch.Generator(
                device=self.device if not self.low_vram_mode else "cpu"
            ).manual_seed(seed)
        else:
            generator = None

        base_qr = generate_base_qr(qr_data)
        condition_image = resize_for_condition_image(base_qr, 768)

        start = time.perf_counter()
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=condition_image,
            width=768,
            height=768,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        elapsed = time.perf_counter() - start

        metadata = {
            "elapsed_seconds": round(elapsed, 3),
            "num_inference_steps": num_inference_steps,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }

        if self.device == "cuda" and torch.cuda.is_available():
            metadata["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            torch.cuda.reset_peak_memory_stats()

        return result.images[0], metadata


if __name__ == "__main__":
    pipeline = QRPipeline(low_vram_mode=True)

    image, meta = pipeline.generate(
        prompt="a serene japanese garden with koi pond, cherry blossoms, highly detailed, soft lighting",
        qr_data="https://example.com",
        seed=42,
    )

    image.save("outputs_test_styled_qr.png")
    print("Generation metadata:", meta)