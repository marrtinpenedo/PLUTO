#!/usr/bin/env python
# -*- coding: utf-8 -*-


import subprocess
import sys
import os
import re
import json
import time

EXPERIMENTS = [
    ("01", "experiment_01_advanced.py"),
    ("02", "experiment_02_super_ensemble.py"),
    ("03", "experiment_03_pytorch_mlp.py"),
    ("04", "experiment_04_cluster_and_conquer.py"),
    ("05", "experiment_05_ultimate_vae_fe.py"),
    ("06", "experiment_06_xgbod_sota.py"),
    ("07", "experiment_07_insane_niche.py"),
    ("08", "experiment_08_nature_sota.py"),
    ("09", "experiment_09_supmin_tabm.py"),
    ("10", "experiment_10_autogluon_cleanlab.py"),
    ("11", "experiment_11_ultimate_hybrid.py"),
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

results = {}


def parse_classification_report(output):
    """Extract key metrics from the classification report in stdout."""
    metrics = {}

    # Find the classification report section
    lines = output.split('\n')

    # Look for the line with "weighted avg"
    for i, line in enumerate(lines):
        if 'weighted avg' in line:
            parts = line.split()
            # weighted avg  precision  recall  f1-score  support
            try:
                metrics['f1_weighted'] = float(parts[-2])
                metrics['recall_weighted'] = float(parts[-3])
                metrics['precision_weighted'] = float(parts[-4])
            except (ValueError, IndexError):
                pass

        if 'macro avg' in line:
            parts = line.split()
            try:
                metrics['f1_macro'] = float(parts[-2])
            except (ValueError, IndexError):
                pass

        # Extract per-class metrics
        if 'OK' in line and ('Min' in line or 'Minoritario' in line):
            parts = line.split()
            try:
                # Find precision, recall, f1 values (they are floats)
                floats = [float(x) for x in parts if re.match(r'^\d+\.\d+$', x)]
                if len(floats) >= 3:
                    metrics['ok_precision'] = floats[0]
                    metrics['ok_recall'] = floats[1]
                    metrics['ok_f1'] = floats[2]
            except (ValueError, IndexError):
                pass

        if 'NOK' in line and ('May' in line or 'Mayoritario' in line):
            parts = line.split()
            try:
                floats = [float(x) for x in parts if re.match(r'^\d+\.\d+$', x)]
                if len(floats) >= 3:
                    metrics['nok_precision'] = floats[0]
                    metrics['nok_recall'] = floats[1]
                    metrics['nok_f1'] = floats[2]
            except (ValueError, IndexError):
                pass

    # Extract time
    for line in lines:
        if 'Tiempo Total' in line or 'Tiempo total' in line:
            m = re.search(r'(\d+\.\d+)s', line)
            if m:
                metrics['time_seconds'] = float(m.group(1))

    return metrics


def run_experiment(exp_id, script_name):
    script_path = os.path.join(SCRIPT_DIR, script_name)
    print(f"\n{'='*70}")
    print(f"  RUNNING EXPERIMENT {exp_id}: {script_name}")
    print(f"{'='*70}")

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=1200,  # 20 min timeout
            cwd=SCRIPT_DIR
        )
        elapsed = time.time() - start

        output = result.stdout + result.stderr
        print(output[-3000:] if len(output) > 3000 else output)

        if result.returncode != 0:
            print(f"  [FAILED] Exit code: {result.returncode}")
            return {'status': 'FAILED', 'error': result.stderr[-500:], 'time_seconds': elapsed}

        metrics = parse_classification_report(output)
        metrics['status'] = 'OK'
        return metrics

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Experiment {exp_id} exceeded 20 minutes")
        return {'status': 'TIMEOUT'}
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {'status': 'ERROR', 'error': str(e)}


if __name__ == '__main__':
    print("=" * 70)
    print("  PLUTO EXPERIMENT RUNNER — ALL 11 EXPERIMENTS (LEAK-FREE)")
    print("=" * 70)

    all_results = {}
    for exp_id, script in EXPERIMENTS:
        metrics = run_experiment(exp_id, script)
        all_results[f"Exp_{exp_id}"] = metrics
        print(f"\n  >> Exp {exp_id} result: {metrics}")

    # Save results
    output_path = os.path.join(SCRIPT_DIR, '..', '..', 'experiment_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Exp':<8} {'Status':<10} {'F1-W':<8} {'F1-M':<8} {'OK-F1':<8} {'NOK-F1':<8} {'OK-Rec':<8} {'NOK-Rec':<8} {'Time':<8}")
    print("-" * 80)
    for name, m in sorted(all_results.items()):
        if m.get('status') == 'OK':
            print(f"{name:<8} {'OK':<10} "
                  f"{m.get('f1_weighted', 0):<8.4f} "
                  f"{m.get('f1_macro', 0):<8.4f} "
                  f"{m.get('ok_f1', 0):<8.4f} "
                  f"{m.get('nok_f1', 0):<8.4f} "
                  f"{m.get('ok_recall', 0):<8.4f} "
                  f"{m.get('nok_recall', 0):<8.4f} "
                  f"{m.get('time_seconds', 0):<8.1f}")
        else:
            print(f"{name:<8} {m.get('status', '?'):<10}")
