/*
 * utils.cpp — Haswell-optimized implementations for HPC Assignment 4
 *
 * Target CPU: Intel Xeon E5-2640 v3 (Haswell microarchitecture)
 *   L1d:  32 KiB per core (8-way, 64B line)
 *   L2:   256 KiB per core (8-way, 64B line)
 *   L3:   20 MiB shared (20-way, 64B line)
 *   Clock: 2.60 GHz base, ~3.3 GHz turbo
 *   ROB:  192 entries
 *   Load/Store: 2 loads + 1 store per cycle
 *   DRAM: DDR4, ~65 ns latency, ~8-9 GB/s measured sequential BW
 *
 * Tuning decisions derived from on-cluster benchmarks:
 *
 *   PREFETCH DISTANCE = 32
 *   ----------------------
 *   Benchmark sweep (PF = 0..32) showed:
 *     250x100 (198KB mesh, fits L2): PF irrelevant, ~16.3 cyc/pt
 *     500x200 (787KB mesh, L3):      PF=0 → 36.0 cyc/pt
 *                                    PF=32 → 21.6 cyc/pt (best, 40% gain)
 *     1000x400 (3.1MB mesh, L3):     PF=0 → 36.7 cyc/pt
 *                                    PF=32 → 23.4 cyc/pt (best, 36% gain)
 *   PF=32 costs only 0.3 cyc/pt on the small grid while saving 13-15
 *   cyc/pt on the large grids.  Universal PF=32 is the clear winner.
 *
 *   Why 32? DRAM latency ~65 ns × 3.3 GHz = ~215 cycles.  At ~28
 *   cycles per 4-point iteration, PF=32 gives 32/4=8 iterations of
 *   look-ahead = 224 cycles, just covering DRAM latency.  Haswell's
 *   simpler hardware prefetcher (vs Alder Lake's 13-stream detector)
 *   benefits more from explicit software prefetch at longer distances.
 *
 *   PREFETCH HINT = T0 (L1d)
 *   -------------------------
 *   T1 (L2 only) performed identically — points are consumed immediately,
 *   so bringing them into L1 is strictly better.  No benefit to T1.
 *
 *   NO MESH PREFETCHING
 *   --------------------
 *   Benchmark showed mesh prefetch HURTS (+1.5-2.0 cyc/pt).  Particles
 *   are uniformly random, so mesh access is essentially random within the
 *   grid.  Prefetched lines get evicted by other random accesses before
 *   the scatter-add reaches them.  The 8 extra prefetch instructions per
 *   4-point group also consume issue bandwidth on Haswell's narrower
 *   frontend (4-wide decode vs 6-wide on Golden Cove).
 *
 *   4-POINT UNROLL (not 8)
 *   -----------------------
 *   8-point unroll tested neutral or slightly worse.  Haswell's 192-entry
 *   ROB limits in-flight instructions: 8-point unroll needs ~56+ live
 *   temporaries, forcing register spills.  4-point keeps ~28 temporaries,
 *   well within the 168 physical integer + 168 FP register budget.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>
#include "utils.h"

/* ============================================================
 * Bilinear Interpolation — Haswell-optimized serial
 * ============================================================ */
void interpolation(double *mesh_value, Points *points) {
    const double inv_dx  = 1.0 / dx;
    const double inv_dy  = 1.0 / dy;
    const double ldx     = dx;
    const double ldy     = dy;
    const int lGRID_X    = GRID_X;
    const int lNX        = NX;
    const int lNY        = NY;
    const int N          = NUM_Points;

    const int PF = 32;

    int i = 0;
    const int n4 = N & ~3;

    for (; i < n4; i += 4) {

        /*
         * Two prefetches targeting two DISTINCT cache lines:
         *   points[i+PF]   → cache line holding points[i+PF .. i+PF+3]
         *   points[i+PF+4] → next cache line holding points[i+PF+4 .. i+PF+7]
         * Each Points is 16 bytes; 4 Points = 64 bytes = 1 cache line.
         */
        if (i + PF + 7 < N) {
            _mm_prefetch((const char *)&points[i + PF],     _MM_HINT_T0);
            _mm_prefetch((const char *)&points[i + PF + 4], _MM_HINT_T0);
        }

        /* Load 4 particle positions */
        const double px0 = points[i  ].x,  py0 = points[i  ].y;
        const double px1 = points[i+1].x,  py1 = points[i+1].y;
        const double px2 = points[i+2].x,  py2 = points[i+2].y;
        const double px3 = points[i+3].x,  py3 = points[i+3].y;

        /* Cell index via multiply-by-reciprocal (avoids 14-20 cycle division) */
        int ix0 = (int)(px0 * inv_dx),  iy0 = (int)(py0 * inv_dy);
        int ix1 = (int)(px1 * inv_dx),  iy1 = (int)(py1 * inv_dy);
        int ix2 = (int)(px2 * inv_dx),  iy2 = (int)(py2 * inv_dy);
        int ix3 = (int)(px3 * inv_dx),  iy3 = (int)(py3 * inv_dy);

        /* Clamp: particles at x=1.0 produce ix=NX → map to last cell */
        if (ix0 >= lNX) ix0 = lNX - 1;
        if (iy0 >= lNY) iy0 = lNY - 1;
        if (ix1 >= lNX) ix1 = lNX - 1;
        if (iy1 >= lNY) iy1 = lNY - 1;
        if (ix2 >= lNX) ix2 = lNX - 1;
        if (iy2 >= lNY) iy2 = lNY - 1;
        if (ix3 >= lNX) ix3 = lNX - 1;
        if (iy3 >= lNY) iy3 = lNY - 1;

        /* Local offsets within cell */
        const double lx0 = px0 - ix0 * ldx,  ly0 = py0 - iy0 * ldy;
        const double lx1 = px1 - ix1 * ldx,  ly1 = py1 - iy1 * ldy;
        const double lx2 = px2 - ix2 * ldx,  ly2 = py2 - iy2 * ldy;
        const double lx3 = px3 - ix3 * ldx,  ly3 = py3 - iy3 * ldy;

        const double rx0 = ldx - lx0,  ry0 = ldy - ly0;
        const double rx1 = ldx - lx1,  ry1 = ldy - ly1;
        const double rx2 = ldx - lx2,  ry2 = ldy - ly2;
        const double rx3 = ldx - lx3,  ry3 = ldy - ly3;

        /* Base grid offset (row-major flat index) */
        const int b0 = iy0 * lGRID_X + ix0;
        const int b1 = iy1 * lGRID_X + ix1;
        const int b2 = iy2 * lGRID_X + ix2;
        const int b3 = iy3 * lGRID_X + ix3;

        /*
         * Interleaved scatter-accumulate: group by node offset so the
         * OOO engine sees 4 independent stores to (likely) different
         * cache lines, maximizing MSHR utilization.
         */
        mesh_value[b0]               += rx0 * ry0;
        mesh_value[b1]               += rx1 * ry1;
        mesh_value[b2]               += rx2 * ry2;
        mesh_value[b3]               += rx3 * ry3;

        mesh_value[b0 + 1]           += lx0 * ry0;
        mesh_value[b1 + 1]           += lx1 * ry1;
        mesh_value[b2 + 1]           += lx2 * ry2;
        mesh_value[b3 + 1]           += lx3 * ry3;

        mesh_value[b0 + lGRID_X]     += rx0 * ly0;
        mesh_value[b1 + lGRID_X]     += rx1 * ly1;
        mesh_value[b2 + lGRID_X]     += rx2 * ly2;
        mesh_value[b3 + lGRID_X]     += rx3 * ly3;

        mesh_value[b0 + lGRID_X + 1] += lx0 * ly0;
        mesh_value[b1 + lGRID_X + 1] += lx1 * ly1;
        mesh_value[b2 + lGRID_X + 1] += lx2 * ly2;
        mesh_value[b3 + lGRID_X + 1] += lx3 * ly3;
    }

    /* Scalar tail: remaining 0-3 particles */
    for (; i < N; i++) {
        const double px = points[i].x;
        const double py = points[i].y;

        int ix = (int)(px * inv_dx);
        int iy = (int)(py * inv_dy);
        if (ix >= lNX) ix = lNX - 1;
        if (iy >= lNY) iy = lNY - 1;

        const double lx = px - ix * ldx;
        const double ly = py - iy * ldy;
        const double rx = ldx - lx;
        const double ry = ldy - ly;

        const int base = iy * lGRID_X + ix;
        mesh_value[base]               += rx * ry;
        mesh_value[base + 1]           += lx * ry;
        mesh_value[base + lGRID_X]     += rx * ly;
        mesh_value[base + lGRID_X + 1] += lx * ly;
    }
}


/* ============================================================
 * Stochastic Mover — Serial
 * ============================================================ */
void mover_serial(Points *points, double deltaX, double deltaY) {
    const int N = NUM_Points;

    for (int i = 0; i < N; i++) {
        double x_new, y_new;

        do {
            double rx = (double)rand() / RAND_MAX;
            double ry = (double)rand() / RAND_MAX;

            double disp_x = rx * 2.0 * deltaX - deltaX;
            double disp_y = ry * 2.0 * deltaY - deltaY;

            x_new = points[i].x + disp_x;
            y_new = points[i].y + disp_y;

        } while (x_new < 0.0 || x_new > 1.0 || y_new < 0.0 || y_new > 1.0);

        points[i].x = x_new;
        points[i].y = y_new;
    }
}


/* ============================================================
 * Stochastic Mover — Parallel (4-thread OpenMP)
 *
 * Benchmark showed schedule(static) and schedule(guided) are
 * nearly identical (~0.041s for 14M particles).  Using static
 * for deterministic partitioning and minimal overhead.
 *
 * rand_r() with per-thread seeds avoids glibc's global mutex
 * on rand(), which would serialize all 4 threads.
 * ============================================================ */
void mover_parallel(Points *points, double deltaX, double deltaY) {
    const int N = NUM_Points;

    #pragma omp parallel num_threads(4)
    {
        unsigned int seed = (unsigned int)(omp_get_thread_num() * 1073741827u
                                           + 2654435761u);

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            double x_new, y_new;

            do {
                double rx = (double)rand_r(&seed) / RAND_MAX;
                double ry = (double)rand_r(&seed) / RAND_MAX;

                double disp_x = rx * 2.0 * deltaX - deltaX;
                double disp_y = ry * 2.0 * deltaY - deltaY;

                x_new = points[i].x + disp_x;
                y_new = points[i].y + disp_y;

            } while (x_new < 0.0 || x_new > 1.0 || y_new < 0.0 || y_new > 1.0);

            points[i].x = x_new;
            points[i].y = y_new;
        }
    }
}


/* ============================================================
 * Write structured mesh to file
 * ============================================================ */
void save_mesh(double *mesh_value) {
    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }

    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}
