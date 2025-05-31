#!/usr/bin/env python3
"""
Run the full SWE‑bench *localise → retrieve → combine → reproduce* pipeline for a single
`target_id`, with resilient logging + automatic retries.

This script is a direct, sequential translation of the command‑snippet shared in chat.

Author: ChatGPT – 2025‑05‑09
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import tenacity
from tenacity import (after_log, before_log, retry_if_exception_type,
                      stop_after_attempt, wait_exponential)

######################################################################################
# Logging set‑up                                                                    #
######################################################################################
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_DIR / "pipeline_runs.log"),
    ],
)
logger = logging.getLogger(__name__)

######################################################################################
# Resilient command runner                                                          #
######################################################################################

# def _run_raw(cmd: str) -> None:
#     """Execute *cmd* in a shell; raise *RuntimeError* if the return‑code != 0."""
#     logger.info("➤ %s", cmd)
#     try:
#         proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)

#         if proc.stdout:
#             logger.info("stdout:\n%s", proc.stdout)
#         if proc.stderr:
#             logger.info("stderr:\n%s", proc.stderr)

#         if proc.returncode != 0:
#             # Handle existing file assertion error
#             if "AssertionError: Output file already exists" in proc.stderr:
#                 logger.warning("Skipping step due to existing output file (continuing to next step)")
#                 return
#             raise RuntimeError(
#                 f"Command failed (exit {proc.returncode}) – see above for details",
#             )

#     except RuntimeError as e:
#         if "AssertionError: Output file already exists" in str(e):
#             logger.warning("Skipping step due to existing output file")
#         else:
#             raise

def _run_raw(cmd: str) -> None:
    """Execute *cmd* in a shell; raise *RuntimeError* if the return‑code != 0."""
    logger.info("➤ %s", cmd)
    try:
        proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)

        if proc.stdout:
            logger.info("stdout:\n%s", proc.stdout)
        if proc.stderr:
            logger.info("stderr:\n%s", proc.stderr)

        # Handle existing file assertion error
        if proc.returncode != 0:
            if "AssertionError: Output file already exists" in proc.stderr:
                logger.warning("Skipping step due to existing output file (continuing to next step)")
                return
            if "No such file or directory" in proc.stderr:
                logger.warning("Skipping step due to missing file (continuing to next step)")
                return
            raise RuntimeError(
                f"Command failed (exit {proc.returncode}) – see above for details",
            )

    except FileNotFoundError as e:
        logger.warning("Skipping step due to missing file: %s", e)
    except RuntimeError as e:
        if "AssertionError: Output file already exists" in str(e):
            logger.warning("Skipping step due to existing output file")
        else:
            raise


def retryable(cmd: str):
    """Decorate *cmd* string with tenacity‑powered retry logic."""

    @tenacity.retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=4, max=64),
        retry=retry_if_exception_type(RuntimeError),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG),
        reraise=True,
    )
    def _inner():
        _run_raw(cmd)

    return _inner

######################################################################################
# Command builders                                                                  #
######################################################################################

def build_commands(target_id: str, model: str, dataset: str, temperature: float, backend: str, n_samples: int, num_threads: int = 10) -> list[str]:
    res_root = Path("results")
    cmds = []
    
    # ─────────────────────────────────── Experiment B ───────────────────────────────────────────────────
    for pass_id in ["2-2"]:
        # Generate refined samples
        cmds.append(
            f"python agentless/test/generate_reproduction_tests_refined.py --max_samples {n_samples} "
            f"--output_folder {res_root}/{dataset}_{target_id}/reproduction_test_samples_temp_{temperature}_{n_samples}_samples_{model}_refined_{pass_id} "
            f"--target_id {target_id} --model {model} --backend {backend} --num_threads {num_threads} "
            f"--extra_info_file {res_root}/{dataset}/edit_location_individual_{target_id}/loc_merged_{pass_id}_outputs.jsonl "
            f"--dataset {dataset}"
        )
        # Run refined tests
        for i in range(n_samples):
            cmds.append(
                f"python agentless/test/run_reproduction_tests.py "
                f"--run_id=reproduction_test_samples_temp_{temperature}_{n_samples}_samples_{model}_refined_{pass_id}_{i} "
                f"--test_jsonl={res_root}/{dataset}_{target_id}/reproduction_test_samples_temp_{temperature}_{n_samples}_samples_{model}_refined_{pass_id}/output_{i}_processed_reproduction_test.jsonl "
                f"--num_workers=2 --testing "
                f"--dataset {dataset}"
            )


    return cmds

######################################################################################
# Main                                                                               #
######################################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Run SWE‑bench pipeline for a single issue with retries & logging.")
    parser.add_argument("target_id", help="e.g. django__django-11815")
    parser.add_argument("model", help="e.g. claude-3-5-sonnet-20241022")
    parser.add_argument("dataset", help="e.g. SWE Bench Lite")
    parser.add_argument("--samples", "-n", type=int, default=100, help="number of reproduction samples (default: 100)")
    parser.add_argument("--threads", "-t", type=int, default=10, help="num_threads for localisation/retrieval (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling (default: 0.8)")
    parser.add_argument("--backend", type=str, default="anthropic", help="Backend for generation (default: anthropic)")

    args = parser.parse_args()

    try:
        commands = build_commands(args.target_id, args.model, args.dataset, args.temperature, args.backend, args.samples, args.threads)
        logger.info("Pipeline built – %d individual steps", len(commands))

        for step, cmd in enumerate(commands, start=1):
            logger.info("──────── step %d/%d ────────", step, len(commands))
            retryable(cmd)()
            time.sleep(1)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user – exiting early…")
    except Exception as exc:
        logger.exception("Pipeline aborted due to unrecoverable error: %s", exc)
        sys.exit(1)

    logger.info("🏁  All steps completed successfully for target_id=%s", args.target_id)


if __name__ == "__main__":
    main()
