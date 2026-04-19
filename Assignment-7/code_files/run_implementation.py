#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_implementation.py
=====================

Benchmarks the parallel PIC interpolation pipeline on gics4.

Pipeline:
    1. Compile input_file_maker, serial baseline, and parallel binary.
    2. Generate each test config's input.bin.
    3. For every config, run the serial binary once (baseline) + the parallel
       binary at T = 1, 2, 4, 8, 16 threads, taking the best wall time of N.
    4. Validate correctness against Test_Mesh.out using Test_input.bin.
    5. Dump per-config CSV + JSON + summary under ./outputs/.

main.cpp has been updated to use omp_get_wtime() so all reported times are
accurate per-phase wall-clock times.
"""

import csv
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

# (tag, NX, NY, Points, Maxiter)
CONFIGS = [
    ("a",  250, 100,    900000, 10),
    ("b",  250, 100,   5000000, 10),
    ("c",  500, 200,   3600000, 10),
    ("d",  500, 200,  20000000, 10),
    ("e", 1000, 400,  14000000, 10),
]

THREAD_COUNTS = [1, 2, 4, 8, 16]
REPEATS = 3  # best of N runs

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

PHASE_RE = re.compile(r"Total\s+(\w+)\s+Time\s*=\s*([0-9.eE+\-]+)")
VOIDS_RE = re.compile(r"Total Number of Voids\s*=\s*(\d+)")
ALGO_RE  = re.compile(r"Total Algorithm Time\s*=\s*([0-9.eE+\-]+)")


def run(cmd, env=None, cwd=HERE, stdin_data=None):
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdin=subprocess.PIPE if stdin_data is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    out, _ = proc.communicate(stdin_data)
    return proc.returncode, out


def die(msg, out=None):
    sys.stderr.write("[fatal] " + msg + "\n")
    if out:
        sys.stderr.write(out + "\n")
    sys.exit(1)


def compile_all():
    print("[build] compiling serial and parallel binaries")
    cc = "g++"
    base_flags = ["-O2", "-fopenmp"]

    # Serial build: uses utils_serial.cpp (no OMP pragmas)
    # -fopenmp needed because main.cpp uses omp_get_wtime()
    rc, out = run([cc] + base_flags + [
        "main.cpp", "utils_serial.cpp", "init.cpp",
        "-lm", "-o", "main_serial.out",
    ])
    if rc:
        die("serial build failed", out)

    # Parallel build
    rc, out = run([cc] + base_flags + [
        "main.cpp", "utils.cpp", "init.cpp",
        "-lm", "-o", "main_parallel.out",
    ])
    if rc:
        die("parallel build failed", out)


def make_input(tag, nx, ny, points, maxiter, out_path):
    """Generate binary input directly in Python (much faster than C maker
    which writes maxiter*points doubles we'd just truncate)."""
    import struct as st
    import random as rng
    rng.seed(42)
    with open(out_path, "wb") as f:
        f.write(st.pack("iiii", nx, ny, points, maxiter))
        # Write in 64KB chunks to avoid building a huge list
        CHUNK = 4096
        for start in range(0, points, CHUNK):
            end = min(start + CHUNK, points)
            buf = b""
            for _ in range(end - start):
                buf += st.pack("dd", rng.random(), rng.random())
            f.write(buf)
    mb = os.path.getsize(out_path) / (1024.0 * 1024.0)
    print("  generated {0} ({1:.1f} MB)".format(os.path.basename(out_path), mb))


def parse_output(stdout):
    data = {}
    for m in PHASE_RE.finditer(stdout):
        data[m.group(1).lower()] = float(m.group(2))
    m = VOIDS_RE.search(stdout)
    data["voids"] = int(m.group(1)) if m else -1
    m = ALGO_RE.search(stdout)
    data["algo_time"] = float(m.group(1)) if m else 0.0
    return data


def run_binary(binary, input_path, threads=None):
    env = os.environ.copy()
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
        env["OMP_PROC_BIND"] = "true"
    t0 = time.time()
    rc, out = run(["./" + binary, input_path], env=env)
    wall = time.time() - t0
    if rc:
        die("{0} crashed (threads={1})".format(binary, threads), out)
    data = parse_output(out)
    data["wall"] = wall
    return data


def best_of(runs):
    """Best algo_time wins (omp_get_wtime based, accurate wall time)."""
    return min(runs, key=lambda d: d["algo_time"])


# --------------------------------------------------------------------------- #
# Correctness                                                                 #
# --------------------------------------------------------------------------- #

def validate_parallel():
    print("[check] validating parallel binary against Test_Mesh.out")
    for t in THREAD_COUNTS:
        run_binary("main_parallel.out", os.path.join(HERE, "Test_input.bin"),
                   threads=t)
        rc, out = run(["diff", "-q", "Mesh.out", "Test_Mesh.out"])
        if rc:
            die("parallel output mismatch at T={0}\n{1}".format(t, out))
    print("  all thread counts match Test_Mesh.out")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    compile_all()
    validate_parallel()

    summary = {}

    for (tag, nx, ny, points, maxiter) in CONFIGS:
        print("")
        print("[config {0}] NX={1} NY={2} Points={3} Maxiter={4}".format(
            tag, nx, ny, points, maxiter))

        input_path = os.path.join(HERE, "input_{0}.bin".format(tag))
        if not os.path.exists(input_path):
            print("  generating " + os.path.basename(input_path))
            make_input(tag, nx, ny, points, maxiter, input_path)

        cfg_results = {
            "config": {"tag": tag, "nx": nx, "ny": ny,
                       "points": points, "maxiter": maxiter},
            "serial": None,
            "parallel": {},
        }

        # Serial baseline (OMP_NUM_THREADS=1 with serial utils)
        print("  [serial] best of {0}".format(REPEATS))
        serial_runs = [run_binary("main_serial.out", input_path, threads=1)
                       for _ in range(REPEATS)]
        serial = best_of(serial_runs)
        cfg_results["serial"] = serial
        serial_algo = serial["algo_time"]
        print("    algo={0:.4f}s  (int={1:.4f} norm={2:.4f} "
              "mov={3:.4f} denorm={4:.4f})".format(
                  serial_algo,
                  serial.get("interpolation", 0.0),
                  serial.get("normalization", 0.0),
                  serial.get("mover", 0.0),
                  serial.get("denormalization", 0.0)))

        # Parallel runs at each thread count
        for t in THREAD_COUNTS:
            print("  [parallel T={0:2d}] best of {1}".format(t, REPEATS))
            runs = [run_binary("main_parallel.out", input_path, threads=t)
                    for _ in range(REPEATS)]
            best = best_of(runs)
            cfg_results["parallel"][str(t)] = best
            algo = best["algo_time"]
            speedup = serial_algo / algo if algo > 0 else 0.0
            eff = speedup / t * 100.0
            print("    algo={0:.4f}s  speedup={1:6.2f}x  eff={2:5.1f}%  "
                  "(int={3:.4f} mov={4:.4f})".format(
                      algo, speedup, eff,
                      best.get("interpolation", 0.0),
                      best.get("mover", 0.0)))

        summary[tag] = cfg_results
        dump_config_outputs(cfg_results)

        # Incremental summary dump
        with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        write_summary_table(summary)

        # Drop the config's input.bin once done to save disk space
        try:
            os.remove(input_path)
        except OSError:
            pass

    print("")
    print("[done] outputs written to " + OUT_DIR)


def dump_config_outputs(cfg_results):
    cfg = cfg_results["config"]
    tag = cfg["tag"]
    serial = cfg_results["serial"]
    serial_algo = serial["algo_time"]

    path = os.path.join(OUT_DIR, "config_{0}.csv".format(tag))
    f = open(path, "w")
    w = csv.writer(f)
    w.writerow([
        "threads", "algo_time", "speedup", "efficiency_pct",
        "interpolation", "normalization", "mover", "denormalization",
        "voids",
    ])
    w.writerow([
        "serial", serial_algo, 1.0, 100.0,
        serial.get("interpolation", 0.0),
        serial.get("normalization", 0.0),
        serial.get("mover", 0.0),
        serial.get("denormalization", 0.0),
        serial.get("voids", -1),
    ])
    for t_str in sorted(cfg_results["parallel"].keys(), key=int):
        t = int(t_str)
        best = cfg_results["parallel"][t_str]
        algo = best["algo_time"]
        speedup = serial_algo / algo if algo > 0 else 0.0
        eff = speedup / t * 100.0
        w.writerow([
            t, algo, speedup, eff,
            best.get("interpolation", 0.0),
            best.get("normalization", 0.0),
            best.get("mover", 0.0),
            best.get("denormalization", 0.0),
            best.get("voids", -1),
        ])
    f.close()


def write_summary_table(summary):
    path = os.path.join(OUT_DIR, "summary.txt")
    f = open(path, "w")
    f.write(
        "All times are wall-clock (omp_get_wtime) from main.cpp output.\n"
        "Speedup = serial_algo_time / parallel_algo_time\n"
        "Efficiency = speedup / num_threads * 100%%\n\n")
    for tag in sorted(summary.keys()):
        data = summary[tag]
        cfg = data["config"]
        serial = data["serial"]
        serial_algo = serial["algo_time"]
        f.write("=== Config {0}: NX={1} NY={2} Points={3} Maxiter={4} ===\n".format(
            tag, cfg["nx"], cfg["ny"], cfg["points"], cfg["maxiter"]))
        f.write("serial algo = {0:.4f}s  (voids={1})\n".format(
            serial_algo, serial.get("voids", -1)))
        f.write("  int={0:.4f}s  norm={1:.4f}s  mov={2:.4f}s  denorm={3:.4f}s\n\n".format(
            serial.get("interpolation", 0.0),
            serial.get("normalization", 0.0),
            serial.get("mover", 0.0),
            serial.get("denormalization", 0.0)))
        f.write("{0:>4} {1:>10} {2:>9} {3:>7}   {4:>10} {5:>10} {6:>10} {7:>10}\n".format(
            "T", "algo(s)", "speedup", "eff%",
            "interp", "norm", "mover", "denorm"))
        f.write("-" * 85 + "\n")
        for t_str in sorted(data["parallel"].keys(), key=int):
            t = int(t_str)
            best = data["parallel"][t_str]
            algo = best["algo_time"]
            speedup = serial_algo / algo if algo > 0 else 0.0
            eff = speedup / t * 100.0
            f.write("{0:>4d} {1:>10.4f} {2:>9.2f} {3:>7.1f}   "
                    "{4:>10.4f} {5:>10.4f} {6:>10.4f} {7:>10.4f}\n".format(
                        t, algo, speedup, eff,
                        best.get("interpolation", 0.0),
                        best.get("normalization", 0.0),
                        best.get("mover", 0.0),
                        best.get("denormalization", 0.0)))
        f.write("\n")
    f.close()


if __name__ == "__main__":
    main()
