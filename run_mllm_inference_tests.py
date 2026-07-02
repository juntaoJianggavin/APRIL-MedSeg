#!/usr/bin/env python
"""Run MLLM inference pipeline tests for all text-guided MLLM configs."""

import os
import sys
import time
import yaml
import subprocess
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = os.path.join(BASE_DIR, 'configs', 'training_paradigms', 'text_guided')
PYTHON = sys.executable

MLLM_CONFIGS = [
    'synapse_grounding_dino_medsam.yaml',
    'synapse_grounding_dino_sam2.yaml',
    'synapse_grounding_dino_sammed2d.yaml',
    'synapse_grounding_dino_litemedsam.yaml',
    'synapse_internvl_medsam.yaml',
    'synapse_internvl_sam2.yaml',
    'synapse_qwen2vl_medsam.yaml',
    'synapse_qwen2vl_sam2.yaml',
    'synapse_qwen3vl_medsam.yaml',
    'synapse_qwen3vl_sam2.yaml',
    'synapse_qwen3vl_sammed2d.yaml',
]


def write_test_script(yaml_path):
    """Write an inline test script for a single config."""
    script = '''import sys, os, yaml, time, traceback
import numpy as np

BASE_DIR = %r
YAML_PATH = %r
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

import h5py
h5_path = os.path.join(BASE_DIR, "data", "Synapse", "test_vol_h5", "case0001.npy.h5")
with h5py.File(h5_path, "r") as f:
    image = f["image"][:]
mid = image.shape[0] // 2
slice_2d = image[mid]
slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
slice_rgb = np.stack([slice_norm] * 3, axis=-1)
slice_rgb = (slice_rgb * 255).astype(np.uint8)
print("Test image shape: " + str(slice_rgb.shape), flush=True)

with open(YAML_PATH) as f:
    cfg = yaml.safe_load(f)

# Disable flash_attn if not available
try:
    import flash_attn
except ImportError:
    mllm = cfg.get("mllm", {})
    g = mllm.get("grounder", {})
    if "use_flash_attn" in g:
        g["use_flash_attn"] = False
        print("Disabled use_flash_attn", flush=True)

try:
    print("Building pipeline...", flush=True)
    t0 = time.time()
    from medseg.inference.mllm import build_pipeline_from_config
    pipe = build_pipeline_from_config(cfg)
    print("Pipeline built in " + str(round(time.time()-t0, 1)) + "s", flush=True)

    print("Running inference...", flush=True)
    t0 = time.time()
    out = pipe(slice_rgb)
    elapsed = time.time() - t0
    print("Inference done in " + str(round(elapsed, 1)) + "s", flush=True)
    print("Label map shape: " + str(out.label_map.shape), flush=True)
    print("Unique labels: " + str(np.unique(out.label_map)), flush=True)
    print("Per-class masks: " + str(list(out.per_class_masks.keys())), flush=True)
    print("SUCCESS", flush=True)
except Exception as e:
    traceback.print_exc()
    print("FAILED: " + str(e), flush=True)
    sys.exit(1)
''' % (BASE_DIR, yaml_path)
    return script


def run_single_test(yaml_path, timeout=3600):
    script = write_test_script(yaml_path)
    script_file = os.path.join(BASE_DIR, 'output_test', '_mllm_inline_test.py')
    os.makedirs(os.path.dirname(script_file), exist_ok=True)
    with open(script_file, 'w') as f:
        f.write(script)

    env = os.environ.copy()
    env.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

    try:
        result = subprocess.run(
            [PYTHON, script_file],
            capture_output=True, text=True, timeout=timeout,
            cwd=BASE_DIR, env=env
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, 'TIMEOUT after ' + str(timeout) + 's'
    finally:
        try:
            os.unlink(script_file)
        except:
            pass


def extract_error(output):
    for line in reversed(output.strip().split('\n')):
        line = line.strip()
        if line and ('Error' in line or 'error' in line or 'Exception' in line or 'FAILED' in line):
            return line[:200]
    lines = output.strip().split('\n')
    return lines[-1][:200] if lines else 'unknown'


def main():
    results = {'pass': [], 'fail': [], 'skip': []}

    for yf in MLLM_CONFIGS:
        name = yf.replace('.yaml', '')
        yaml_path = os.path.join(TEXT_DIR, yf)

        print("\n" + "=" * 60)
        print("  [MLLM] " + name)
        print("=" * 60)

        t0 = time.time()
        success, output = run_single_test(yaml_path)
        elapsed = time.time() - t0

        status = 'PASS' if success else 'FAIL'
        err = ''
        if not success:
            err = extract_error(output)

        results[status.lower()].append({'name': name, 'error': err, 'time': elapsed})

        out_lines = output.strip().split('\n')
        for line in out_lines[-5:]:
            print("  " + line)

        print("  " + status + " (" + str(round(elapsed, 1)) + "s)" + (" -- " + err if err else ''))
        sys.stdout.flush()

    print("\n" + "=" * 60)
    print("  MLLM INFERENCE FINAL REPORT")
    print("=" * 60)
    print("  PASS: " + str(len(results['pass'])))
    print("  FAIL: " + str(len(results['fail'])))

    if results['pass']:
        print("\n  -- PASSED --")
        for item in results['pass']:
            print("  " + item['name'] + " (" + str(round(item['time'], 1)) + "s)")

    if results['fail']:
        print("\n  -- FAILED --")
        for item in results['fail']:
            print("  " + item['name'] + ": " + item['error'])


if __name__ == '__main__':
    main()
