# AI-Styled QR Code Generator - Where Art Meets Functionality

A production-grade pipeline that transforms plain QR codes into stunning, brand-aligned visual assets using Stable Diffusion and ControlNet. Instead of generic black-and-white squares, clients get scannable QR codes that look like actual artwork — koi ponds, cityscapes, abstract patterns — generated in under 30 seconds via a single API call, eliminating hours of manual design work per asset.

Built end-to-end with a focus on real engineering constraints: memory-efficient GPU inference on consumer hardware (8GB VRAM), async concurrent request handling so the server stays responsive under load, and persistent cloud storage so every generated asset is immediately shareable via a public URL.

## System Architecture

```
React Frontend (Vite)
        │
        │ POST /generate { url, prompt }
        ▼
FastAPI Backend (uvicorn)
        │
        ├── ThreadPoolExecutor (GPU-bound work off the async event loop)
        │         │
        │         ▼
        │   QR Base Generator (qrcode, ERROR_CORRECT_H)
        │         │
        │         ▼
        │   ControlNet Pipeline (SD 1.5 + monster-labs QR ControlNet)
        │   fp16 · CPU offloading · attention slicing · VAE tiling
        │         │
        │         ▼
        │   Generated PIL Image
        │
        ├── Supabase Storage  ──► Public image URL
        └── Supabase Postgres ──► Job metadata (url, prompt, elapsed, vram)
                │
                ▼
        JSON response { image_url, elapsed_seconds, peak_vram_mb }
```

## ✨ Key Features

* **AI-Styled QR Generation** — Stable Diffusion 1.5 + monster-labs ControlNet conditions on the QR code's structural pattern while applying any text prompt as visual style, producing scannable branded assets no manual designer could match at this speed.
* **Memory-Efficient Inference** — Runs on 8GB VRAM consumer GPUs via fp16 precision, CPU model offloading, attention slicing, and VAE tiling — peak usage ~4GB, leaving headroom for the OS and other processes.
* **Async Concurrent API** — FastAPI with a single-worker ThreadPoolExecutor correctly separates GPU-bound inference from the async event loop, so the server remains responsive to health checks and queued requests during generation.
* **Cloud Asset Persistence** — Every generated QR is uploaded to Supabase Storage (public CDN URL) with job metadata written to Postgres — prompt, URL encoded, generation time, VRAM usage — making every run auditable and every asset immediately shareable.
* **High Error-Correction QR Codes** — Base QR codes generated with `ERROR_CORRECT_H` (30% damage tolerance), giving the diffusion model maximum headroom to stylize without breaking scannability.
* **Parameter Sweep Tooling** — Built-in sweep script tests multiple `controlnet_conditioning_scale` values systematically with a 3-reader verification harness, producing a CSV of scan fidelity vs. style tradeoff data.

## Project Structure

```text
ai-qr-generator/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, /generate + /health endpoints
│   ├── pipeline.py       # ControlNet + SD1.5 pipeline, memory optimizations
│   ├── qr_base.py        # Base QR generation (qrcode lib, ERROR_CORRECT_H)
│   ├── storage.py        # Supabase Storage upload + Postgres job insert
│   ├── verify.py         # 3-reader scan fidelity harness (pyzbar, ZXing APIs)
│   ├── sweep.py          # Parameter sweep script for conditioning_scale tuning
│   └── postprocess.py    # Finder-pattern compositing (post-processing utility)
├── qr-frontend/
│   ├── src/
│   │   ├── App.jsx       # Main React component
│   │   ├── main.jsx      # React entry point
│   │   └── style.css
│   ├── index.html
│   └── package.json
├── docs/
│   └── PHASE_1_2_SETUP.md
├── outputs/
│   └── sweep/            # Generated images + CSV from sweep runs
├── .env                  # SUPABASE_URL, SUPABASE_SERVICE_KEY (never commit)
├── .env.example
├── .gitignore
└── requirements.txt
```

## Quick Start

### 1. Prerequisites

- Python 3.10+ with CUDA-capable GPU (8GB+ VRAM recommended)
- Node.js 18+ for the frontend
- A free [Supabase](https://supabase.com) project

### 2. Installation

```bash
git clone https://github.com/yourusername/ai-qr-generator.git
cd ai-qr-generator
pip install -r requirements.txt
```

### 3. Prepare Environment

Create a `.env` file in the project root:

```bash
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_key_here
```

In your Supabase dashboard:
- **Storage** → create a public bucket named `qr-codes`
- **SQL Editor** → run:

```sql
create table qr_jobs (
  id uuid primary key default gen_random_uuid(),
  url text not null,
  prompt text not null,
  image_url text,
  elapsed_seconds float,
  peak_vram_mb float,
  created_at timestamp default now()
);

create policy "service role can upload" on storage.objects
  for insert to service_role
  with check (bucket_id = 'qr-codes');

create policy "public can read" on storage.objects
  for select to public
  using (bucket_id = 'qr-codes');
```

### 4. Run the Backend

```bash
# From project root — first run downloads ~5GB of model weights
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for `[startup] Pipeline ready.` then open `http://localhost:8000/docs` for the interactive API.

### 5. Run the Frontend

```bash
cd qr-frontend
npm install
npm run dev
```

Open `http://localhost:5173`, enter a URL and style prompt, hit Generate.

### 6. (Optional) Parameter Sweep

```bash
python -m app.sweep
```

Generates QR codes at multiple conditioning scales, verifies each with the scan harness, and saves results to `outputs/sweep/sweep_results_*.csv`.

## Technology Stack

| Component | Technology |
| --- | --- |
| **Frontend** | React 18, Vite |
| **Backend** | Python, FastAPI, Uvicorn |
| **ML Pipeline** | Stable Diffusion 1.5, ControlNet (monster-labs QR model), HuggingFace Diffusers |
| **GPU Optimization** | PyTorch fp16, CPU offloading, attention slicing, VAE tiling, xFormers |
| **QR Generation** | qrcode (ERROR_CORRECT_H), pyzbar |
| **Database** | Supabase Postgres |
| **Storage** | Supabase Storage (S3-compatible CDN) |
| **Scan Verification** | pyzbar (ZBar), QR Server API (ZXing), ZXing Online |

## How It Works

1. **Base QR Generation**
   The target URL is encoded into a high-error-correction QR code (`ERROR_CORRECT_H` = 30% damage tolerance) using the `qrcode` library, resized to 768×768 with dimensions rounded to multiples of 64 (required by the VAE's downsampling factor). This becomes the ControlNet conditioning image.

2. **Styled Image Generation**
   The ControlNet reads the QR image and produces spatial "hint" tensors at each UNet layer — encoding where dark and light modules must appear. The frozen SD 1.5 UNet simultaneously processes the text prompt (encoded by CLIP) and denoises a random latent over 30 DDIM steps, with the ControlNet hints injected at each step. `controlnet_conditioning_scale` controls the balance: higher preserves more QR structure, lower gives the style more freedom. All inference runs in fp16 with CPU offloading and attention slicing to stay within 8GB VRAM.

3. **Storage + Response**
   The generated PIL image is uploaded to Supabase Storage, returning a permanent public CDN URL. Job metadata (prompt, target URL, elapsed time, peak VRAM) is inserted into Postgres. The API returns a JSON response with the image URL and metrics — the React frontend renders the image immediately and shows generation stats.

## Why AI QR Generator?

This project showcases strong end-to-end ML engineering skills — from understanding diffusion model internals to shipping a working API backed by cloud infrastructure.

It combines:
* Deep learning inference (ControlNet, Stable Diffusion, latent diffusion theory)
* Production GPU memory management (fp16, offloading, attention slicing)
* Async Python backend engineering (FastAPI, ThreadPoolExecutor, ASGI)
* Cloud BaaS integration (Supabase Postgres + Storage)
* Systematic ML evaluation (parameter sweeps, multi-reader scan verification)

into one complete end-to-end pipeline — from a URL string to a publicly hosted, scannable piece of AI art in under 30 seconds.

## ❤️ Made With Passion

Made with ❤️ by Divi Jaiwanth