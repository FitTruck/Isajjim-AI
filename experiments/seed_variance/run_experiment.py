"""
Seed Variance Experiment Runner

실험 전체를 조율하는 메인 스크립트.
Phase 1 (Original, 250 runs) 과 Phase 2 (Optimized, 50 runs) 를
GPU별 subprocess로 병렬 실행합니다.

Usage:
    python run_experiment.py [--phase 1|2|all] [--gpu-ids 0,1]
"""

import argparse
import json
import subprocess
import sys
import time
import logging
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PIX3D_DIR, SAMPLE_LIST, RESULTS_DIR, GPU_IDS,
    SEEDS, OPTIMIZED_SEED, RAW_CSV, OPTIMIZED_CSV,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Runner] %(message)s")
logger = logging.getLogger(__name__)

WORKER_SCRIPT = Path(__file__).resolve().parent / "experiment_worker.py"

# SAM-3D requires sam3d-objects conda environment
SAM3D_PYTHON = Path("/home/rladlems1031/miniconda3/envs/sam3d-objects/bin/python")


def load_samples() -> list:
    """Load selected sample list."""
    if not SAMPLE_LIST.exists():
        logger.error(f"Sample list not found: {SAMPLE_LIST}")
        logger.error("Run download_pix3d.py first to create sample list.")
        sys.exit(1)

    with open(SAMPLE_LIST, 'r') as f:
        return json.load(f)


def create_phase1_tasks(samples: list) -> list:
    """Phase 1: 50 images × 5 seeds = 250 tasks."""
    tasks = []
    for sample in samples:
        img_path = str(PIX3D_DIR / sample["img"])
        mask_path = str(PIX3D_DIR / sample["mask"])
        sample_id = Path(sample["img"]).stem

        for seed in SEEDS:
            tasks.append({
                "category": sample["category"],
                "sample_id": sample_id,
                "img_path": img_path,
                "mask_path": mask_path,
                "seed": seed,
            })

    return tasks


def create_phase2_tasks(samples: list) -> list:
    """Phase 2: 50 images × 1 seed = 50 tasks."""
    tasks = []
    for sample in samples:
        img_path = str(PIX3D_DIR / sample["img"])
        mask_path = str(PIX3D_DIR / sample["mask"])
        sample_id = Path(sample["img"]).stem

        tasks.append({
            "category": sample["category"],
            "sample_id": sample_id,
            "img_path": img_path,
            "mask_path": mask_path,
            "seed": OPTIMIZED_SEED,
        })

    return tasks


def split_tasks(tasks: list, n_gpus: int) -> list[list]:
    """Split tasks evenly across GPUs."""
    chunks = [[] for _ in range(n_gpus)]
    for i, task in enumerate(tasks):
        chunks[i % n_gpus].append(task)
    return chunks


def run_workers(phase: str, tasks: list, gpu_ids: list[int]):
    """Launch worker subprocesses on each GPU."""
    n_gpus = len(gpu_ids)
    chunks = split_tasks(tasks, n_gpus)

    logger.info(f"Phase: {phase}")
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"GPUs: {gpu_ids}")
    for i, gpu_id in enumerate(gpu_ids):
        logger.info(f"  GPU {gpu_id}: {len(chunks[i])} tasks")

    # Save task files
    task_files = []
    output_files = []
    for i, gpu_id in enumerate(gpu_ids):
        task_file = RESULTS_DIR / f"tasks_{phase}_gpu{gpu_id}.json"
        output_file = RESULTS_DIR / f"results_{phase}_gpu{gpu_id}.csv"
        task_file.parent.mkdir(parents=True, exist_ok=True)

        with open(task_file, 'w') as f:
            json.dump(chunks[i], f)

        task_files.append(task_file)
        output_files.append(output_file)

    # Launch subprocesses
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        if not chunks[i]:
            continue

        cmd = [
            str(SAM3D_PYTHON), str(WORKER_SCRIPT),
            str(gpu_id),
            phase,
            str(task_files[i]),
            str(output_files[i]),
        ]

        # Redirect stderr to log file, stdout to devnull (workers log to stderr)
        log_file = RESULTS_DIR / f"worker_{phase}_gpu{gpu_id}.log"
        log_fh = open(log_file, 'w')

        logger.info(f"Launching worker on GPU {gpu_id} (log: {log_file})")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=log_fh,
        )
        processes.append((gpu_id, proc, output_files[i], log_fh))

    # Wait for all processes
    start_time = time.time()
    for gpu_id, proc, output_file, log_fh in processes:
        logger.info(f"Waiting for GPU {gpu_id} worker...")
        proc.wait()
        log_fh.close()

        if proc.returncode != 0:
            logger.error(f"GPU {gpu_id} worker failed (exit code {proc.returncode})")
            # Print last 20 lines of log
            log_file = RESULTS_DIR / f"worker_{phase}_gpu{gpu_id}.log"
            if log_file.exists():
                with open(log_file) as f:
                    lines = f.readlines()
                    for line in lines[-20:]:
                        logger.error(f"  [GPU{gpu_id}] {line.rstrip()}")
        else:
            logger.info(f"GPU {gpu_id} worker completed successfully")

    elapsed = time.time() - start_time
    logger.info(f"Phase {phase} completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Merge results
    merge_results(output_files, RAW_CSV if phase == "original" else OPTIMIZED_CSV)


def merge_results(output_files: list[Path], merged_path: Path):
    """Merge per-GPU CSV files into one."""
    import csv

    all_rows = []
    fieldnames = None

    for f in output_files:
        if not f.exists():
            logger.warning(f"Output file not found: {f}")
            continue
        with open(f, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                all_rows.append(row)

    if not all_rows:
        logger.error("No results to merge!")
        return

    with open(merged_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(f"Merged {len(all_rows)} results → {merged_path}")


def main():
    parser = argparse.ArgumentParser(description="Seed Variance Experiment Runner")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all",
                        help="Which phase to run (1=original, 2=optimized, all=both)")
    parser.add_argument("--gpu-ids", type=str, default=",".join(map(str, GPU_IDS)),
                        help="Comma-separated GPU IDs")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_samples()
    logger.info(f"Loaded {len(samples)} samples")

    if args.phase in ("1", "all"):
        logger.info("=" * 60)
        logger.info("PHASE 1: Original SAM-3D Seed Variance (250 runs)")
        logger.info("=" * 60)
        tasks = create_phase1_tasks(samples)
        run_workers("original", tasks, gpu_ids)

    if args.phase in ("2", "all"):
        logger.info("=" * 60)
        logger.info("PHASE 2: Optimized SAM-3D (50 runs)")
        logger.info("=" * 60)
        tasks = create_phase2_tasks(samples)
        run_workers("optimized", tasks, gpu_ids)

    if args.phase == "all":
        logger.info("=" * 60)
        logger.info("Both phases complete. Run analyze_results.py for analysis.")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
