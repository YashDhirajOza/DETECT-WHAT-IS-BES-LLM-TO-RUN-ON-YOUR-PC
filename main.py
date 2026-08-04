#!/usr/bin/env python3
"""
LLM Hardware Capability Checker
--------------------------------
Scans your CPU, RAM, GPU (NVIDIA) and disk space, then estimates the
largest local LLM (at a given quantization) you can comfortably run.

Model "tiers" -- which parameter count maps to which example model --
live in an external models_catalog.json file instead of being hardcoded
here. New models drop constantly; this file is meant to be edited or
swapped out on its own, without touching the scanning/math logic below.

Usage:
    python app.py                          # pretty text report, Q4_K_M, default contexts
    python app.py --quant q8_0             # size for a different quantization level
    python app.py --context 8,64,128       # custom context-length scenarios (in K tokens)
    python app.py --json                   # machine-readable output
    python app.py --save report.txt        # save the report to a file
    python app.py --catalog my_models.json # point at a different/updated catalog
    python app.py --no-color               # disable ANSI colors
"""

import argparse
import csv
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime

try:
    import psutil
except ImportError:
    print("ERROR: 'psutil' is required. Install it with: pip install psutil")
    sys.exit(1)

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

GB = 1024 ** 3
DEFAULT_CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_catalog.json")

FALLBACK_CATALOG = {
    "last_updated": "unknown (using built-in fallback -- no catalog file found)",
    "quantizations": {
        "q4_k_m": {"bytes_per_param": 0.6, "label": "Q4_K_M (Balanced, Recommended)"},
        "q5_k_m": {"bytes_per_param": 0.75, "label": "Q5_K_M (Higher Quality)"},
        "q8_0": {"bytes_per_param": 1.0, "label": "Q8_0 (Near-Lossless)"},
        "fp16": {"bytes_per_param": 2.0, "label": "FP16 (Full Precision)"},
    },
    "model_tiers": [
        {"min_params_billion": 70, "example": "Llama 3.3 70B / Qwen 2.5 72B"},
        {"min_params_billion": 32, "example": "Qwen 2.5 32B / QwQ 32B"},
        {"min_params_billion": 12, "example": "Gemma 3 12B / Mistral Nemo 12B"},
        {"min_params_billion": 8, "example": "Llama 3.1 8B / Qwen 2.5 7B"},
        {"min_params_billion": 3, "example": "Llama 3.2 3B / Phi-3.5 Mini"},
        {"min_params_billion": 0, "example": "Sub-2B edge models (Qwen 2.5 0.5B/1.5B)"},
    ],
}


# ---------- tiny ANSI color helper (no extra dependency) ----------
class C:
    enabled = True

    @staticmethod
    def w(code, text):
        return f"\033[{code}m{text}\033[0m" if C.enabled else text


def bold(t): return C.w("1", t)
def dim(t): return C.w("2", t)
def green(t): return C.w("32", t)
def yellow(t): return C.w("33", t)
def cyan(t): return C.w("36", t)
def red(t): return C.w("31", t)


# ---------- hardware scanning ----------

def get_cpu_info():
    return {
        "architecture": platform.machine(),
        "processor": platform.processor() or "[Unable to detect]",
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
    }


def get_ram_info():
    svmem = psutil.virtual_memory()
    return {"total_gb": svmem.total / GB, "available_gb": svmem.available / GB}


def get_disk_info(path="."):
    try:
        usage = shutil.disk_usage(path)
        return {"free_gb": usage.free / GB, "total_gb": usage.total / GB}
    except OSError:
        return {"free_gb": None, "total_gb": None}


def get_nvidia_gpu_info():
    """Returns (list_of_gpu_dicts, max_free_vram_gb)."""
    os_name = platform.system()
    gpus = []

    if os_name == "Darwin":
        return gpus, 0.0

    max_free_vram = 0.0

    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_gb = mem_info.total / GB
                free_gb = mem_info.free / GB
                gpus.append({"name": name, "total_gb": total_gb, "free_gb": free_gb})
                max_free_vram = max(max_free_vram, free_gb)
            pynvml.nvmlShutdown()
            return gpus, max_free_vram
        except Exception:
            pass  # fall through to nvidia-smi

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        reader = csv.reader(io.StringIO(result.stdout.strip()))
        for row in reader:
            if len(row) == 3:
                name = row[0].strip()
                try:
                    total_gb = int(row[1].strip()) / 1024
                    free_gb = int(row[2].strip()) / 1024
                    gpus.append({"name": name, "total_gb": total_gb, "free_gb": free_gb})
                    max_free_vram = max(max_free_vram, free_gb)
                except ValueError:
                    gpus.append({"name": name, "total_gb": None, "free_gb": None})
    except FileNotFoundError:
        pass  # no nvidia-smi -> non-NVIDIA system or drivers not installed
    except Exception:
        pass

    return gpus, max_free_vram


# ---------- catalog (swappable model data) ----------

def load_catalog(path):
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_source"] = path
            return data
        except (json.JSONDecodeError, OSError) as e:
            print(yellow(f"Warning: couldn't read catalog '{path}' ({e}). Using built-in fallback."))
    fallback = dict(FALLBACK_CATALOG)
    fallback["_source"] = "built-in fallback"
    return fallback


def pick_example(model_tiers, max_params_billion):
    for tier in sorted(model_tiers, key=lambda t: -t["min_params_billion"]):
        if max_params_billion >= tier["min_params_billion"]:
            return tier["example"]
    return model_tiers[-1]["example"] if model_tiers else "N/A"


# ---------- recommendation math ----------

def compute_recommendations(free_vram_gb, available_ram_gb, os_name, catalog, quant_key, context_ks):
    quant = catalog["quantizations"].get(quant_key)
    if quant is None:
        quant_key = "q4_k_m"
        quant = catalog["quantizations"][quant_key]

    if os_name == "Darwin":
        usable_memory = available_ram_gb * 0.75
        memory_source = "Unified Memory (Apple Silicon estimate, ~75% of available RAM)"
    elif free_vram_gb > 0:
        usable_memory = free_vram_gb
        memory_source = "GPU VRAM (free)"
    else:
        usable_memory = available_ram_gb * 0.5
        memory_source = "System RAM (no GPU detected, CPU inference estimate)"

    scenarios = []
    for size_k in context_ks:
        kv_overhead = size_k * 0.1  # ~0.1GB per 1K tokens, FP16 KV cache heuristic
        available_for_weights = usable_memory - 1.0 - kv_overhead  # ~1.0GB framework overhead
        if available_for_weights <= 0:
            scenarios.append({"context_k": size_k, "max_params_billion": 0.0,
                               "example": None, "status": "insufficient_memory"})
            continue
        max_params_billion = available_for_weights / quant["bytes_per_param"]
        example = pick_example(catalog["model_tiers"], max_params_billion)
        scenarios.append({"context_k": size_k, "max_params_billion": max_params_billion,
                           "example": example, "status": "ok"})

    return {
        "usable_memory_gb": usable_memory,
        "memory_source": memory_source,
        "quant_key": quant_key,
        "quant_label": quant["label"],
        "scenarios": scenarios,
    }


# ---------- report assembly + rendering ----------

def build_report(catalog_path, quant_key, context_ks):
    os_name = platform.system()
    cpu = get_cpu_info()
    ram = get_ram_info()
    disk = get_disk_info()
    gpus, free_vram = get_nvidia_gpu_info()
    catalog = load_catalog(catalog_path)
    rec = compute_recommendations(free_vram, ram["available_gb"], os_name, catalog, quant_key, context_ks)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "os": os_name,
        "cpu": cpu,
        "ram": ram,
        "disk": disk,
        "gpus": gpus,
        "catalog_source": catalog.get("_source"),
        "catalog_last_updated": catalog.get("last_updated"),
        "recommendation": rec,
    }


def render_text(report):
    lines = []
    lines.append(bold(cyan("=" * 59)))
    lines.append(bold(cyan("           LOCAL LLM CAPABILITY REPORT")))
    lines.append(bold(cyan("=" * 59)))
    lines.append(dim(f"Generated: {report['generated_at']}"))
    lines.append(dim(f"Model catalog: {report['catalog_source']} (last updated: {report['catalog_last_updated']})"))

    lines.append("\n" + bold("--- CPU ---"))
    c = report["cpu"]
    lines.append(f"Architecture:    {c['architecture']}")
    lines.append(f"Processor:       {c['processor']}")
    lines.append(f"Physical Cores:  {c['physical_cores'] or '[Unable to detect]'}")
    lines.append(f"Total Threads:   {c['logical_cores'] or '[Unable to detect]'}")

    lines.append("\n" + bold("--- System RAM ---"))
    r = report["ram"]
    lines.append(f"Total RAM:       {r['total_gb']:.2f} GB")
    lines.append(f"Available RAM:   {r['available_gb']:.2f} GB")

    lines.append("\n" + bold("--- Disk ---"))
    d = report["disk"]
    if d["free_gb"] is not None:
        lines.append(f"Free Disk Space: {d['free_gb']:.2f} GB  {dim('(model files typically run 2-40GB each)')}")
    else:
        lines.append("Free Disk Space: [Unable to detect]")

    lines.append("\n" + bold("--- GPU (NVIDIA) ---"))
    if report["gpus"]:
        for g in report["gpus"]:
            if g["total_gb"] is not None:
                lines.append(f"  - {g['name']}: {g['total_gb']:.2f} GB total | {g['free_gb']:.2f} GB free")
            else:
                lines.append(f"  - {g['name']} (failed to parse VRAM)")
    else:
        if report["os"] == "Darwin":
            lines.append("Apple Silicon detected -- using unified memory estimate instead.")
        else:
            lines.append("No NVIDIA GPU detected (or drivers/nvidia-smi unavailable).")
            lines.append(dim("AMD ROCm / Intel Arc users: check VRAM manually; falling back to RAM-based estimate."))

    rec = report["recommendation"]
    lines.append("\n" + bold(green(f"--- Recommendation ({rec['quant_label']}) ---")))
    lines.append(f"Memory source: {rec['memory_source']}")
    lines.append(f"Usable memory: {rec['usable_memory_gb']:.2f} GB\n")

    for s in rec["scenarios"]:
        label = f"{s['context_k']}K context"
        if s["status"] == "insufficient_memory":
            lines.append(f"  {label:<16}: {red('Insufficient memory (OOM before weights load)')}")
        else:
            lines.append(f"  {label:<16}: ~{s['max_params_billion']:.1f}B params  "
                          f"{dim('(e.g., ' + s['example'] + ')')}")

    lines.append("\n" + dim("*Assumes a single GPU, FP16 KV cache, no tensor-parallel splitting."))
    lines.append(dim("*Quantize the KV cache (Q4/Q8) to fit longer contexts than shown above."))
    lines.append(dim("*Model examples come from models_catalog.json -- update that file as new models drop."))
    return "\n".join(lines)


def parse_context_arg(raw):
    try:
        values = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if not values:
            raise ValueError
        return values
    except ValueError:
        print(red(f"Invalid --context value '{raw}', falling back to default 4,16,32"))
        return [4, 16, 32]


def main():
    parser = argparse.ArgumentParser(description="Scan hardware and estimate the largest local LLM you can run.")
    parser.add_argument("--quant", default="q4_k_m", choices=["q4_k_m", "q5_k_m", "q8_0", "fp16"],
                         help="Quantization level to size for (default: q4_k_m)")
    parser.add_argument("--context", default="4,16,32",
                         help="Comma-separated context sizes in K tokens (default: 4,16,32)")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG_PATH,
                         help="Path to models_catalog.json (default: bundled catalog next to this script)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON instead of text")
    parser.add_argument("--save", metavar="PATH", help="Save the report to a file")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors in text output")
    args = parser.parse_args()

    C.enabled = not args.no_color and sys.stdout.isatty()

    context_ks = parse_context_arg(args.context)
    if not args.json:
        print("Scanning hardware...\n")

    report = build_report(args.catalog, args.quant, context_ks)
    output = json.dumps(report, indent=2) if args.json else render_text(report)
    print(output)

    if args.save:
        try:
            save_content = re.sub(r"\033\[[0-9;]*m", "", output) if not args.json else output
            with open(args.save, "w", encoding="utf-8") as f:
                f.write(save_content)
            if not args.json:
                print(f"\n{dim('Report saved to ' + args.save)}")
        except OSError as e:
            print(red(f"Failed to save report: {e}"))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # e.g. piping into `head` and it closes the stream early -- not a real error
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(130)