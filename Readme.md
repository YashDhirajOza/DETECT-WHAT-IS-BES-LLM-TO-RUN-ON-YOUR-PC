# LLM Hardware Capability Checker

A small script that scans your CPU, RAM, GPU (NVIDIA), and disk, then
estimates the largest local LLM you can comfortably run.

## Setup

```bash
pip install -r requirements.txt
```

`pynvml` is optional — it gives more accurate free-VRAM readings than
shelling out to `nvidia-smi`. If it's missing, the script falls back to
`nvidia-smi` automatically, and to a RAM-based estimate if neither is
available (e.g. AMD/Intel GPUs, or no GPU at all).

## Usage

```bash
python app.py                          # default report, Q4_K_M quant, 4K/16K/32K contexts
python app.py --quant q8_0             # size for a different quantization
python app.py --context 8,64,128       # custom context-length scenarios (in K tokens)
python app.py --json                   # machine-readable output
python app.py --save report.txt        # save the report to a file
python app.py --catalog other.json     # use a different/updated model catalog
python app.py --no-color               # disable ANSI colors (useful when piping/saving)
```

## Files

- `app.py` — all hardware scanning + sizing math. You shouldn't need to
  edit this when new models come out.
- `models_catalog.json` — the list of "at this many params, you're in
  the ballpark of model X" examples, plus quantization byte-costs. This
  is the part that goes stale as new models release — edit or replace
  this file on its own.

## How the estimate works

1. Detects usable memory: free GPU VRAM if an NVIDIA GPU is present,
   ~75% of available RAM on Apple Silicon (unified memory), otherwise
   ~50% of available system RAM as a rough CPU-inference budget.
2. Subtracts ~1GB for framework/OS overhead and an estimated KV-cache
   cost that scales with context length (~0.1GB per 1K tokens, FP16).
3. Divides what's left by the chosen quantization's bytes-per-parameter
   to get a max parameter count, then looks up the closest example
   model from `models_catalog.json`.

These are heuristics, not a guarantee — actual usable memory varies by
runtime (llama.cpp, Ollama, vLLM, etc.), OS overhead, and other apps
using the GPU at the time.