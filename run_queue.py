"""Run the benchmark jobs a few at a time, starting a job only when enough memory is free.

  python run_queue.py                       # all jobs below, 2 at a time
  python run_queue.py --only bac_s1 sac_s1  # a subset

Each training process commits ~4-5 GB on Windows (CUDA context + replay buffer), so launching
everything at once can starve other programs. Logs go to runs/logs/<name>.out/.err; a job whose
runs/<name>/done marker exists is skipped, and interrupted off-policy jobs resume from last.pt.
While it runs, the queue asks Windows not to idle-sleep.
"""
import argparse
import ctypes
import os
import subprocess
import sys
import time

JOBS = {  # in priority order
    "bac_s1": "offpolicy.py --algo bac --total-steps 2500000 --seed 1",
    "sac_s1": "offpolicy.py --algo sac --total-steps 2000000 --seed 1",
    "ppo_s1": "ppo.py --total-steps 10000000 --seed 1",
    "crossq_s1": "offpolicy.py --algo crossq --total-steps 1000000 --seed 1",
    "td3_s1": "offpolicy.py --algo td3 --total-steps 2000000 --seed 1",
    "rpo_sym_s1": "ppo.py --total-steps 10000000 --seed 1 --rpo-alpha 0.5 --sym-coef 0.02",
    # Posture experiment: continue CrossQ from 1M to 2.5M with and without the upright-spine bonus.
    "crossq_posture_s1": "offpolicy.py --algo crossq --total-steps 2500000 --seed 1 "
                         "--init-from runs/crossq_s1/last.pt --posture-coef 150",
    "crossq_long_s1": "offpolicy.py --algo crossq --total-steps 2500000 --seed 1 --init-from runs/crossq_s1/last.pt",
}


def free_commit_gb():
    """Free commit charge (RAM + page file) on Windows; inf elsewhere."""
    if os.name != "nt":
        return float("inf")

    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("total_phys", "avail_phys", "total_pagefile", "avail_pagefile",
                                              "total_virtual", "avail_virtual", "avail_ext_virtual")]

    s = MemStatus()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.avail_pagefile / 2**30


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-parallel", type=int, default=2)
    p.add_argument("--min-free-gb", type=float, default=6.0, help="free commit needed to start a job")
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()

    if os.name == "nt":
        # Ask Windows not to idle-sleep while the queue runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED). This is
        # a per-process request that ends with the process; closing the lid still sleeps.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    os.makedirs("runs/logs", exist_ok=True)
    pending = [k for k in JOBS if (args.only is None or k in args.only) and not os.path.exists(f"runs/{k}/done")]
    running = {}
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore"}
    while pending or running:
        for name, proc in list(running.items()):
            if proc.poll() is not None:
                print(f"{time.strftime('%H:%M:%S')} finished {name} (exit {proc.returncode})", flush=True)
                if proc.returncode == 0:
                    open(f"runs/{name}/done", "w").close()
                del running[name]
        if pending and len(running) < args.max_parallel and free_commit_gb() >= args.min_free_gb:
            name = pending.pop(0)
            cmd = [sys.executable] + JOBS[name].split() + ["--run-name", name]
            if cmd[1] == "offpolicy.py":
                cmd.append("--resume")  # picks up runs/<name>/last.pt after an interruption
            out, err = open(f"runs/logs/{name}.out", "a"), open(f"runs/logs/{name}.err", "a")
            running[name] = subprocess.Popen(cmd, stdout=out, stderr=err, env=env)
            print(f"{time.strftime('%H:%M:%S')} started {name}: {' '.join(cmd[1:])}", flush=True)
            time.sleep(90)  # let it allocate before judging free memory for the next one
        time.sleep(15)


if __name__ == "__main__":
    main()
