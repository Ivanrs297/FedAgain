"""
Quick smoke test: 1 dataset, 2 rounds, 5 clients.
Should finish in under 1 minute if the full pipeline works.
"""

import subprocess
import sys
import time


def update_config(updates: dict):
    config_path = 'config.py'
    with open(config_path, 'r') as f:
        lines = f.readlines()
    with open(config_path, 'w') as f:
        for line in lines:
            updated = False
            for key, value in updates.items():
                if line.strip().startswith(f"{key} ="):
                    if isinstance(value, str) and not value.startswith(('"', "'")):
                        f.write(f'{key} = "{value}"\n')
                    else:
                        f.write(f"{key} = {value}\n")
                    updated = True
                    break
            if not updated:
                f.write(line)


# Save original config
with open('config.py', 'r') as f:
    original_config = f.read()

print("=== SMOKE TEST: FedMedian + DAUDON_MIX + iid ===")
print("Config: 5 clients, 2 rounds, 1 epoch")
print()

update_config({
    "DATASET": "DAUDON_MIX",
    "STRATEGY": "FedMedian",
    "DATA_DISTRIBUTION": "iid",
    "NUM_CLIENTS": 5,
    "NUM_ROUNDS": 2,
    "LOCAL_EPOCHS": 1,
    "CORRUPT_CLIENTS": True,
    "CORRUPT_CLIENT_RATIO": 0.3,
    "CORRUPTION_SEVERITY": 3,
})

t0 = time.time()
proc = subprocess.run(
    [sys.executable, "-u", "main.py"],
    text=True,
)
elapsed = time.time() - t0

# Restore original config
with open('config.py', 'w') as f:
    f.write(original_config)

print()
if proc.returncode == 0:
    print(f"✓ PIPELINE OK — finished in {elapsed:.1f}s")
else:
    print(f"✗ PIPELINE FAILED after {elapsed:.1f}s")

sys.exit(proc.returncode)
