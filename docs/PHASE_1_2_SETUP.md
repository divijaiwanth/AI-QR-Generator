# Setup & Run Instructions — Phase 1 & 2

## What's in this phase
- `app/qr_base.py` — generates plain QR codes with high error correction (run this first, no GPU needed)
- `app/pipeline.py` — the ControlNet + SD1.5 pipeline (needs your GPU)

## 1. Set up your environment

```bash
cd ai-qr-generator
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

If `xformers` fails to install or import later, that's OK — the code
falls back gracefully. It's an optimization, not a requirement.

## 2. Check your GPU is visible to PyTorch

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

This should print `True` and your GPU name. If it prints `False`, stop here —
you likely have a CUDA/driver mismatch. Tell me what it prints and we'll debug
before going further (don't waste time downloading 5GB of model weights if
the GPU isn't even visible).

## 3. Test QR generation alone (fast, no GPU needed)

```bash
python3 -m app.qr_base
```

You should get `outputs_test_base_qr.png` — open it and confirm it looks like
a normal QR code with the three corner squares clearly visible.

## 4. Run the full pipeline (this is the big one)

```bash
python3 -m app.pipeline
```

**First run will download ~5GB of model weights** (SD1.5 + ControlNet) from
Hugging Face — this only happens once, cached afterward in
`~/.cache/huggingface`.

Expect this run to take a few minutes the first time (download) and then
30 seconds to a few minutes for the actual generation, depending on your GPU.

**If you get a CUDA out-of-memory error**, tell me the exact error message —
there are more aggressive options (sequential CPU offload, smaller resolution,
fewer inference steps) we can layer on, and the specific error tells us which
stage is spiking.

## 5. What to check in the output

Open `outputs_test_styled_qr.png`. At `controlnet_conditioning_scale=1.3`,
you should see a recognizable QR grid pattern blended with garden/koi pond
imagery. It might look more like "QR code with a faint image overlay" than
"beautiful art" at this stage — that's expected and correct. We'll tune the
style-vs-scannability balance once we have the scan verification harness
built (Phase 3), because right now you have no way to *measure* if it still
scans — you're just eyeballing it.

Also check the printed metadata dict — note the `elapsed_seconds` and
`peak_vram_mb` values. Send those to me along with the image. They're our
first real data points for your "memory-efficient inference" and "30 second
turnaround" metrics.

---

**Report back with:** (1) did it run without OOM, (2) the metadata dict
output, (3) the generated image. Then we move to Phase 3: building the
3-reader scan verification harness, which is what turns "looks scannable"
into an actual measured percentage.
