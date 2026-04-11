# test_load2.py
import sys
print("1. Importing...", flush=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"   torch={torch.__version__}, cpu only", flush=True)

name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("2. Tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(name)
print("   OK", flush=True)

# Test A: le plus basique possible, aucune option
print("3A. Loading model (no options)...", flush=True)
try:
    model = AutoModelForCausalLM.from_pretrained(name)
    print(f"   OK! dtype={next(model.parameters()).dtype}", flush=True)
    p = sum(p.numel() for p in model.parameters())
    print(f"   {p:,} params", flush=True)
    del model
    import gc; gc.collect()
    print("   Deleted, trying next test", flush=True)
except Exception as e:
    print(f"   FAILED: {e}", flush=True)

# Test B: float32 explicit
print("3B. Loading model (torch_dtype=float32)...", flush=True)
try:
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    print(f"   OK! dtype={next(model.parameters()).dtype}", flush=True)
    del model; gc.collect()
except Exception as e:
    print(f"   FAILED: {e}", flush=True)

# Test C: float16 without low_cpu_mem
print("3C. Loading model (torch_dtype=float16, no low_cpu)...", flush=True)
try:
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
    print(f"   OK! dtype={next(model.parameters()).dtype}", flush=True)
    del model; gc.collect()
except Exception as e:
    print(f"   FAILED: {e}", flush=True)

print("DONE", flush=True)
input("Press Enter...")