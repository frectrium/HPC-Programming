#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lab_08 benchmark harness (Python 2.7 compatible).

Usage:  python run.py [--quick]  [--configs a,b,c,d,e]  [--cores 2,4,8,16,32,64]

Generates input data per configuration, runs:
  - the serial baseline (np=1, OMP=1) once per config (used for speedup ref)
  - the hybrid MPI+OpenMP binary at each requested total-core count
Saves all stdout logs + a parsed CSV into outputs/.
Validates each parallel run's Mesh.out against the serial baseline within tol.

NUMA-aware launch layout (1 MPI rank per Haswell socket = 8 cores/rank):
  total_cores  ranks  threads  hosts
  2            1      2        gics1
  4            1      4        gics1
  8            1      8        gics1            (1 full socket)
  16           2      8        gics1            (1 full node, 2 sockets)
  32           4      8        gics1,gics2
  64           8      8        gics1,gics2,gics3,gics4
"""

from __future__ import print_function
import os, sys, re, csv, shutil, subprocess, time, argparse

LAB_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(LAB_DIR, "outputs")
HOSTFILE    = os.path.join(LAB_DIR, "hostfile")
ALL_HOSTS   = ["gics1", "gics2", "gics3", "gics4"]

MPI_PREFIX  = "/usr/mpi/gcc/openmpi-1.8.8"
ENV = os.environ.copy()
ENV["PATH"] = MPI_PREFIX + "/bin:" + ENV.get("PATH", "")
ENV["LD_LIBRARY_PATH"] = MPI_PREFIX + "/lib64:" + ENV.get("LD_LIBRARY_PATH", "")

CONFIGS = {
    "a": dict(NX=250,  NY=100, points=900000,   maxiter=10),
    "b": dict(NX=250,  NY=100, points=5000000,  maxiter=10),
    "c": dict(NX=500,  NY=200, points=3600000,  maxiter=10),
    "d": dict(NX=500,  NY=200, points=20000000, maxiter=10),
    "e": dict(NX=1000, NY=400, points=14000000, maxiter=10),
}

# (total_cores) -> (ranks, threads_per_rank, num_hosts_used)
CORE_LAYOUTS = {
    2:  (1, 2, 1),
    4:  (1, 4, 1),
    8:  (1, 8, 1),
    16: (2, 8, 1),
    32: (4, 8, 2),
    64: (8, 8, 4),
}

TOL_ABS = 5e-6     # absolute tolerance for mesh comparison (output is %lf, 6 dec)
TOL_REL = 1e-6


def sh(cmd, cwd=None, env=None, capture=True, timeout=None):
    """Run a shell command, return (rc, stdout)."""
    if env is None: env = ENV
    if isinstance(cmd, list): cmd = " ".join(cmd)
    p = subprocess.Popen(["bash", "-lc", cmd], cwd=cwd or LAB_DIR, env=env,
                         stdout=subprocess.PIPE if capture else None,
                         stderr=subprocess.STDOUT if capture else None)
    out, _ = p.communicate()
    if out is None: out = ""
    return p.returncode, out


def write_hostfile(num_hosts):
    with open(HOSTFILE, "w") as f:
        for h in ALL_HOSTS[:num_hosts]:
            f.write("%s slots=16\n" % h)


def compile_all():
    """Compile input_maker and the hybrid MPI+OMP binary."""
    print("[compile] building binaries...")
    rc, out = sh("g++ -O3 -std=c++11 input_file_maker.cpp -o input_maker.out")
    if rc != 0:
        print(out); sys.exit("input_file_maker compile failed")
    rc, out = sh("mpic++ -O3 -std=c++11 -fopenmp -mavx2 -DNDEBUG "
                 "main.cpp utils.cpp init.cpp -lm -o main_parallel.out")
    if rc != 0:
        print(out); sys.exit("main_parallel compile failed")
    print("[compile] OK")


def gen_input(cfg):
    """Run input_maker.out non-interactively."""
    rc, out = sh("./input_maker.out %d %d %d %d" %
                 (cfg["NX"], cfg["NY"], cfg["points"], cfg["maxiter"]))
    if rc != 0:
        print(out); sys.exit("input generation failed")


def run_one(total_cores, log_path):
    """Run main_parallel.out at the given total core count, save log."""
    ranks, threads, num_hosts = CORE_LAYOUTS[total_cores]
    write_hostfile(num_hosts)

    # Layout reasoning for OpenMPI 1.8.8 on 2-socket Haswell nodes (8 cores/socket):
    #   - 1 rank: bind-to socket so OMP threads land on real cores (default
    #     binding pins the rank to a single core, which would serialize OMP).
    #   - 2 ranks on 1 node: ppr:1:socket -> 1 rank per socket (NUMA-local).
    #   - 4..8 ranks across multiple nodes: ppr:2:node -> 2 ranks per node,
    #     1 per socket; bind-to socket keeps each rank's OMP threads NUMA-local.
    if ranks == 1:
        map_by = "-bind-to socket"
    elif ranks == 2:
        map_by = "-map-by ppr:1:socket -bind-to socket"
    else:
        map_by = "-map-by ppr:2:node -bind-to socket"

    cmd = ("OMP_NUM_THREADS=%d OMP_PROC_BIND=close OMP_PLACES=cores "
           "mpirun --hostfile %s -np %d %s "
           "-x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x LD_LIBRARY_PATH "
           "./main_parallel.out input.bin 2>&1") % (threads, HOSTFILE, ranks, map_by)

    # Best-of-3: re-run and keep the run with smallest total algorithm time.
    best_rc, best_out, best_wall, best_total = -1, "", 0.0, float("inf")
    for _rep in range(3):
        t0 = time.time()
        rc, out = sh(cmd)
        wall = time.time() - t0
        if rc != 0:
            best_rc, best_out, best_wall = rc, out, wall
            continue
        m = re.search(r"Total Algorithm Time = ([\d.]+)", out)
        tot = float(m.group(1)) if m else float("inf")
        if tot < best_total:
            best_total, best_rc, best_out, best_wall = tot, rc, out, wall

    with open(log_path, "w") as f:
        f.write("# cmd: %s (best of 3)\n# wall=%.4fs rc=%d\n%s" %
                (cmd, best_wall, best_rc, best_out))
    return best_rc, best_out, best_wall


_RE_TIME = {
    "interp":  re.compile(r"Total Interpolation Time = ([\d.]+)"),
    "norm":    re.compile(r"Total Normalization Time = ([\d.]+)"),
    "mover":   re.compile(r"Total Mover Time = ([\d.]+)"),
    "denorm":  re.compile(r"Total Denormalization Time = ([\d.]+)"),
    "total":   re.compile(r"Total Algorithm Time = ([\d.]+)"),
    "scat":    re.compile(r"MPI Scatter time = ([\d.]+)"),
    "ar_mesh": re.compile(r"MPI Allreduce\(mesh\) time = ([\d.]+)"),
    "ar_mm":   re.compile(r"MPI Allreduce\(min/max\) time = ([\d.]+)"),
    "voids":   re.compile(r"Total Number of Voids = (-?\d+)"),
}

def parse_log(text):
    out = {}
    for k, r in _RE_TIME.items():
        m = r.search(text)
        if m: out[k] = float(m.group(1)) if k != "voids" else int(m.group(1))
    return out


def compare_mesh(ref_path, cur_path):
    """Compare two Mesh.out files within tolerance. Returns (max_abs, max_rel, n)."""
    with open(ref_path) as f: a = f.read().split()
    with open(cur_path) as f: b = f.read().split()
    if len(a) != len(b):
        return ("len-mismatch", len(a), len(b))
    max_abs, max_rel = 0.0, 0.0
    for x, y in zip(a, b):
        fx, fy = float(x), float(y)
        d = abs(fx - fy)
        if d > max_abs: max_abs = d
        denom = max(abs(fx), abs(fy), 1e-12)
        r = d / denom
        if r > max_rel: max_rel = r
    return (max_abs, max_rel, len(a))


def run_config(cfg_id, cores_list):
    cfg = CONFIGS[cfg_id]
    print("\n" + "=" * 72)
    print("CONFIG %s : NX=%d NY=%d points=%d maxiter=%d" %
          (cfg_id, cfg["NX"], cfg["NY"], cfg["points"], cfg["maxiter"]))
    print("=" * 72)

    out_dir = os.path.join(OUTPUTS_DIR, "config_" + cfg_id)
    if not os.path.isdir(out_dir): os.makedirs(out_dir)

    print("[gen] generating input.bin (%d points)..." % cfg["points"])
    gen_input(cfg)
    sz = os.path.getsize(os.path.join(LAB_DIR, "input.bin"))
    print("[gen] input.bin = %.1f MB" % (sz / 1024.0 / 1024.0))

    # Serial baseline (np=1, OMP=1)
    print("[run] serial baseline (np=1, OMP=1)...")
    rc, _, wall = run_one(1, os.path.join(out_dir, "serial.log")) \
        if False else (None, None, None)  # we use a dedicated 1-core path
    # Actually, "serial" is simply 1 rank * 1 thread. Add an entry to layouts.
    # Run serial:
    write_hostfile(1)
    cmd = ("OMP_NUM_THREADS=1 mpirun --hostfile %s -np 1 "
           "-x OMP_NUM_THREADS -x LD_LIBRARY_PATH "
           "./main_parallel.out input.bin 2>&1") % HOSTFILE
    t0 = time.time()
    rc, out = sh(cmd)
    wall_serial = time.time() - t0
    with open(os.path.join(out_dir, "serial.log"), "w") as f:
        f.write("# cmd: %s\n# wall=%.4fs rc=%d\n%s" % (cmd, wall_serial, rc, out))
    if rc != 0:
        print(out); sys.exit("serial run failed for %s" % cfg_id)
    serial_t = parse_log(out)
    print("[serial] total=%.4fs interp=%.4f mover=%.4f voids=%d wall=%.2fs" %
          (serial_t["total"], serial_t["interp"], serial_t["mover"],
           serial_t["voids"], wall_serial))
    # Save serial mesh as reference
    ref_mesh = os.path.join(out_dir, "Mesh_serial.out")
    shutil.copyfile(os.path.join(LAB_DIR, "Mesh.out"), ref_mesh)

    rows = []
    rows.append(dict(config=cfg_id, total_cores=1, ranks=1, threads=1, **serial_t))
    rows[-1]["wall"] = wall_serial
    rows[-1]["max_abs"] = 0.0
    rows[-1]["max_rel"] = 0.0

    for tc in cores_list:
        ranks, threads, num_hosts = CORE_LAYOUTS[tc]
        log_path = os.path.join(out_dir, "p%d_t%d.log" % (ranks, threads))
        print("[run] %d cores  (np=%d threads/rank=%d hosts=%d)..." %
              (tc, ranks, threads, num_hosts))
        rc, out, wall = run_one(tc, log_path)
        if rc != 0:
            print("[FAIL] cores=%d:" % tc)
            print(out)
            continue
        t = parse_log(out)
        # validate
        cur_mesh = os.path.join(LAB_DIR, "Mesh.out")
        cmp_res = compare_mesh(ref_mesh, cur_mesh)
        if cmp_res[0] == "len-mismatch":
            print("    LENGTH MISMATCH ref=%d cur=%d" % (cmp_res[1], cmp_res[2]))
            ok = False; max_abs = max_rel = float("nan")
        else:
            max_abs, max_rel, _n = cmp_res
            ok = (max_abs <= TOL_ABS and max_rel <= TOL_REL)
        speedup = serial_t["total"] / t["total"] if t.get("total", 0) > 0 else 0.0
        eff = speedup / tc * 100.0
        print("    total=%.4fs  speedup=%.2fx  eff=%.1f%%  "
              "interp=%.4f mover=%.4f ar_mesh=%.4f  diff(abs/rel)=%.2e/%.2e %s" %
              (t.get("total", 0), speedup, eff,
               t.get("interp", 0), t.get("mover", 0), t.get("ar_mesh", 0),
               max_abs, max_rel, "OK" if ok else "**MISMATCH**"))
        rows.append(dict(config=cfg_id, total_cores=tc, ranks=ranks, threads=threads,
                         wall=wall, max_abs=max_abs, max_rel=max_rel,
                         speedup=speedup, efficiency=eff, **t))
        # Save per-run mesh? Skip to save disk; just keep serial reference + log

    # Write CSV per config
    csv_path = os.path.join(out_dir, "timings.csv")
    fieldnames = ["config", "total_cores", "ranks", "threads",
                  "interp", "norm", "mover", "denorm", "total",
                  "scat", "ar_mesh", "ar_mm",
                  "voids", "wall", "max_abs", "max_rel",
                  "speedup", "efficiency"]
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            for fn in fieldnames:
                row.setdefault(fn, "")
            w.writerow(row)
    print("[csv] wrote %s" % csv_path)

    # cleanup generated input + per-iter mesh
    try: os.remove(os.path.join(LAB_DIR, "input.bin"))
    except OSError: pass
    return rows


def maybe_plot(all_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[plot] matplotlib unavailable, skipping: %s" % e)
        return

    plot_dir = os.path.join(OUTPUTS_DIR, "plots")
    if not os.path.isdir(plot_dir): os.makedirs(plot_dir)

    by_cfg = {}
    for r in all_rows:
        by_cfg.setdefault(r["config"], []).append(r)

    for cfg_id, rows in by_cfg.items():
        rows.sort(key=lambda r: r["total_cores"])
        cores  = [r["total_cores"] for r in rows]
        total  = [r["total"]       for r in rows]
        speed  = [r.get("speedup", 1.0) for r in rows]

        # exec time vs cores
        plt.figure(figsize=(7, 5))
        plt.plot(cores, total, "o-", label="total")
        plt.xscale("log", basex=2); plt.yscale("log")
        plt.xlabel("Total cores"); plt.ylabel("Time (s)")
        plt.title("Config %s : execution time" % cfg_id)
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "time_%s.png" % cfg_id), dpi=120,
                    bbox_inches="tight")
        plt.close()

        # speedup vs cores
        plt.figure(figsize=(7, 5))
        plt.plot(cores, speed, "o-", label="actual")
        plt.plot(cores, cores, "k--", alpha=0.5, label="ideal")
        plt.xscale("log", basex=2); plt.yscale("log")
        plt.xlabel("Total cores"); plt.ylabel("Speedup")
        plt.title("Config %s : speedup" % cfg_id)
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "speedup_%s.png" % cfg_id), dpi=120,
                    bbox_inches="tight")
        plt.close()

    print("[plot] wrote plots to %s" % plot_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="a,b,c,d,e",
                   help="comma list, default a,b,c,d,e")
    p.add_argument("--cores",   default="2,4,8,16,32,64",
                   help="comma list of total cores")
    p.add_argument("--no-compile", action="store_true",
                   help="skip recompilation (assume binaries built)")
    p.add_argument("--quick", action="store_true",
                   help="reduced sweep: configs=a,c cores=2,8,32")
    args = p.parse_args()

    if args.quick:
        args.configs = "a,c"
        args.cores   = "2,8,32"

    configs    = [c.strip() for c in args.configs.split(",") if c.strip()]
    cores_list = [int(c.strip()) for c in args.cores.split(",") if c.strip()]
    for c in cores_list:
        if c not in CORE_LAYOUTS:
            sys.exit("unknown core count %d (valid: %s)" % (c, sorted(CORE_LAYOUTS)))

    if not os.path.isdir(OUTPUTS_DIR): os.makedirs(OUTPUTS_DIR)

    if not args.no_compile:
        compile_all()

    all_rows = []
    for cid in configs:
        if cid not in CONFIGS:
            print("skip unknown config %s" % cid); continue
        all_rows.extend(run_config(cid, cores_list))

    # Aggregate summary CSV
    summary = os.path.join(OUTPUTS_DIR, "summary.csv")
    fieldnames = ["config", "total_cores", "ranks", "threads",
                  "interp", "norm", "mover", "denorm", "total",
                  "scat", "ar_mesh", "ar_mm",
                  "voids", "wall", "max_abs", "max_rel",
                  "speedup", "efficiency"]
    with open(summary, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            for fn in fieldnames: row.setdefault(fn, "")
            w.writerow(row)
    print("\n[summary] %s (%d rows)" % (summary, len(all_rows)))

    maybe_plot(all_rows)
    print("\n[done] outputs/ ready.")


if __name__ == "__main__":
    main()
