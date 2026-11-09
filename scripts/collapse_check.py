"""Exit 0 if a run has collapsed to the uniform answer, 1 otherwise.

Collapse = the last 3 evaluations all have test and dev accuracy in [0.44, 0.56] (chance
on SST-2) and the mean of the last 100 logged train losses is within 0.03 of ln 2.
Healthy runs never match both: early LDSD/greedy evals are far below chance (0.24-0.27,
loss ~2), and once train loss is near ln 2 a healthy run is already above 0.8.
Dev accuracy is checked too: a slow run (ZoAR LoRA, test ~0.55, dev ~0.60) is not collapsed.
Usage: python3 scripts/collapse_check.py runs/<name>/stdout.log
"""
import math
import re
import sys

ACC_RE = re.compile(r"'test_acc': ([0-9.]+), 'val_acc': ([0-9.]+)")
LOSS_RE = re.compile(r"'loss': ([0-9.]+)")


def collapsed(text: str) -> bool:
    accs = [(float(m.group(1)), float(m.group(2))) for m in ACC_RE.finditer(text)]
    losses = [float(m.group(1)) for m in LOSS_RE.finditer(text)]
    if len(accs) < 3 or len(losses) < 100:
        return False
    chance = all(0.44 <= a <= 0.56 for pair in accs[-3:] for a in pair)
    mean_loss = sum(losses[-100:]) / 100
    return chance and abs(mean_loss - math.log(2)) < 0.03


if __name__ == "__main__":
    try:
        with open(sys.argv[1], errors="replace") as f:
            sys.exit(0 if collapsed(f.read()) else 1)
    except OSError:
        sys.exit(1)
