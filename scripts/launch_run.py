"""Run run.py under the lab recorder (LabRun) and keep its log and metrics.

Usage (from the repository root):
    python scripts/launch_run.py --name <run-name> --objective "<why>" -- <run.py args>

The recorder writes lab-runs/<name>.json plus the attached stdout log and the
final metrics json. Set LAB_RUN_OFFLINE=1 to keep everything on disk.
"""
import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_run import LabRun  # noqa: E402

logger = logging.getLogger(__name__)
WORK = "zo-ldsd"
EVAL_RE = re.compile(r"Evaluating at step (\d+)")
ACC_RE = re.compile(r"'test_acc': ([0-9.]+), 'val_acc': ([0-9.]+)")
LOSS_RE = re.compile(r"\{'loss': ([0-9.eE+-]+|nan|inf)")
LOSS_EVERY = 100


def _settings(run_args: List[str]) -> dict:
    settings, key = {}, None
    for token in run_args:
        if token.startswith("--"):
            key, _, value = token[2:].partition("=")
            settings[key] = value if value else True
    return settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--out_dir", default="runs")
    args, run_args = parser.parse_known_args()
    run_args = [a for a in run_args if a != "--"]

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout.log"
    # run.py overrides output_dir and result_file with result/<tag>/, so pin the tag.
    result_path = Path("result") / args.name / "results.json"
    command = [sys.executable, "run.py", *run_args, f"--tag={args.name}",
               f"--output_dir=result/{args.name}"]

    with LabRun(work=WORK, name=args.name, settings=_settings(run_args),
                objective=args.objective, command=" ".join(command)) as run:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            eval_step, loss_step = 0, 0
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if (m := EVAL_RE.search(line)):
                    eval_step = int(m.group(1))
                elif (m := ACC_RE.search(line)):
                    run.metric("test/accuracy", float(m.group(1)), step=eval_step)
                    run.metric("val/accuracy", float(m.group(2)), step=eval_step)
                elif (m := LOSS_RE.search(line)):
                    loss_step += 1
                    if loss_step % LOSS_EVERY == 0:
                        run.metric("train/loss", float(m.group(1)), step=loss_step)
            code = proc.wait()
        run.attach(log_path, "stdout.log")
        if code != 0:
            run.finish("failed", reason=f"run.py exited with code {code}")
            return code
        metrics = json.loads(result_path.read_text())
        run.attach(result_path, "result.json")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                run.metric(key, value, final=True)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
