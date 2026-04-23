import argparse
import json
import os
import time
import subprocess
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple


def load_jobs(jsonl_path: Path) -> List[Dict]:
    jobs = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
    return jobs


def deps_satisfied(job: Dict) -> bool:
    for dep in job.get("deps", []):
        if not Path(dep).exists():
            return False
    return True


def is_done(job: Dict) -> bool:
    return Path(job["done_marker"]).exists()


def touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("done\n")


def launch(job: Dict, gpu_id: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cwd = job.get("cwd", None)
    cmd = job["cmd"]

    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a", encoding="utf-8")

    log_f.write(f"\n=== LAUNCH gpu={gpu_id} ===\n")
    log_f.write("CWD: " + (cwd if cwd else os.getcwd()) + "\n")
    log_f.write("CMD: " + " ".join(cmd) + "\n")
    log_f.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._log_handle = log_f
    return proc


def run_post_cmd_if_any(job: Dict) -> int:
    env = os.environ.copy()
    post_cmd = job.get("post_cmd", None)
    if not post_cmd:
        return 0
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cwd = job.get("cwd", None)

    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write("\n=== POST_CMD ===\n")
        log_f.write("CWD: " + (cwd if cwd else os.getcwd()) + "\n")
        log_f.write("CMD: " + " ".join(post_cmd) + "\n")
        log_f.flush()
        res = subprocess.run(
            post_cmd,
            cwd=cwd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_f.write(f"\n=== POST_EXIT code={res.returncode} ===\n")
        log_f.flush()
        return res.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs_file", type=str, required=True, help="Path to jobs.jsonl")
    ap.add_argument("--gpus", type=str, required=True, help="Comma-separated GPU ids, e.g. '0,1,2,3'")
    ap.add_argument("--poll_sec", type=float, default=2.0)
    ap.add_argument("--max_retries", type=int, default=1, help="Retries per job on failure")
    ap.add_argument(
        "--deadlock_policy",
        type=str,
        default="report_and_exit",
        choices=["report_and_exit", "report_and_sleep"],
        help=(
            "If pending jobs are blocked forever by missing deps (and nothing is running), "
            "either report and exit (default) or report and keep sleeping." 
        ),
    )
    args = ap.parse_args()

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("No GPUs provided")

    jobs_file = Path(args.jobs_file).resolve()
    jobs = load_jobs(jobs_file)
    if not jobs:
        raise RuntimeError("No jobs loaded")

    # Retry bookkeeping
    retries_left = {i: args.max_retries for i in range(len(jobs))}

    # Execution state
    free_gpus = deque(gpus)
    running: List[Tuple[subprocess.Popen, int, int]] = []  # (proc, gpu_id, job_index)

    # Pre-skip jobs already done (resume)
    pending = deque(i for i, j in enumerate(jobs) if not is_done(j))

    print(f"Loaded {len(jobs)} jobs; pending={len(pending)}; gpus={gpus}")

    def format_missing_deps(job: Dict) -> List[str]:
        miss = []
        for dep in job.get("deps", []):
            if not Path(dep).exists():
                miss.append(dep)
        return miss
    
    while pending or running:
        # Launch ready jobs while GPUs free
        launched_any = False
        if free_gpus:
            # Iterate over pending to find schedulable jobs
            for _ in range(len(pending)):
                if not free_gpus:
                    break
                idx = pending[0]
                job = jobs[idx]

                if is_done(job):
                    pending.popleft()
                    continue

                if not deps_satisfied(job):
                    # rotate to back and continue searching
                    pending.rotate(-1)
                    continue

                # launch
                gpu = free_gpus.popleft()
                proc = launch(job, gpu)
                running.append((proc, gpu, idx))
                pending.popleft()
                launched_any = True

        # Poll running processes
        time.sleep(args.poll_sec)
        still_running: List[Tuple[subprocess.Popen, int, int]] = []
        for proc, gpu, idx in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, gpu, idx))
                continue

            # close log handle
            try:
                proc._log_handle.write(f"\n=== EXIT code={ret} ===\n")
                proc._log_handle.flush()
                proc._log_handle.close()
            except Exception:
                pass

            job = jobs[idx]
            if ret == 0:
                post_ret = run_post_cmd_if_any(job)
                if post_ret != 0:
                    # treat as failure for retry logic
                    if retries_left[idx] > 0:
                        retries_left[idx] -= 1
                        print(f"[RETRY] gpu={gpu} job={idx} post_cmd exit={post_ret} retries_left={retries_left[idx]}")
                        pending.appendleft(idx)
                    else:
                        print(f"[FAIL] gpu={gpu} job={idx} post_cmd exit={post_ret} (no retries left)")
                    free_gpus.append(gpu)
                    continue

                touch(Path(job["done_marker"]))
                print(f"[DONE] gpu={gpu} job={idx} {job.get('job_type')} scene={job.get('scene')} K={job.get('K')} run={job.get('run_id')} it={job.get('iteration')}")
            else:
                if retries_left[idx] > 0:
                    retries_left[idx] -= 1
                    print(f"[RETRY] gpu={gpu} job={idx} exit={ret} retries_left={retries_left[idx]}")
                    pending.appendleft(idx)  # retry soon
                else:
                    print(f"[FAIL] gpu={gpu} job={idx} exit={ret} (no retries left)")
                    # Leave as failed; not touching done_marker.

            free_gpus.append(gpu)

        running = still_running

        # Deadlock detection: no running jobs, pending non-empty, and none are schedulable due to deps.
        if pending and not running:
            blocked = []
            for idx in list(pending):
                job = jobs[idx]
                if is_done(job):
                    continue
                if not deps_satisfied(job):
                    blocked.append(idx)

            if len(blocked) == len([i for i in pending if not is_done(jobs[i])]):
                # Nothing can ever be launched given current filesystem state.
                print("\n[DEADLOCK] Pending jobs are blocked by missing dependency markers; nothing is running.")
                print("Blocked jobs summary:")
                for idx in blocked[:50]:
                    job = jobs[idx]
                    miss = format_missing_deps(job)
                    print(
                        f"  - job={idx} type={job.get('job_type')} scene={job.get('scene')} K={job.get('K')} "
                        f"run={job.get('run_id')} it={job.get('iteration')} missing_deps={len(miss)}"
                    )
                    for d in miss[:5]:
                        print(f"      missing: {d}")
                    if len(miss) > 5:
                        print(f"      ... (+{len(miss)-5} more)")
                if len(blocked) > 50:
                    print(f"  ... (+{len(blocked)-50} more blocked jobs)")

                if args.deadlock_policy == "report_and_exit":
                    print("\nExiting because deadlock_policy=report_and_exit.")
                    break
                else:
                    print("\nContinuing to sleep because deadlock_policy=report_and_sleep.")
                    time.sleep(max(args.poll_sec, 10.0))


    print("All schedulable jobs complete. Check _done markers and logs for failures.")


if __name__ == "__main__":
    main()
