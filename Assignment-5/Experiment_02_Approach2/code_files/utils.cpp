#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include "init.h"
#include "utils.h"

/* ================================================================
 *  INTERPOLATION — bilinear CIC scatter (serial, Haswell-optimized)
 *  Exact code from Assignment 04. PF=32, 4x unroll proven optimal
 *  for the scatter pattern (random mesh access benefits from prefetch).
 * ================================================================ */
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
        if (i + PF + 7 < N) {
            _mm_prefetch((const char *)&points[i + PF],     _MM_HINT_T0);
            _mm_prefetch((const char *)&points[i + PF + 4], _MM_HINT_T0);
        }

        const double px0 = points[i  ].x,  py0 = points[i  ].y;
        const double px1 = points[i+1].x,  py1 = points[i+1].y;
        const double px2 = points[i+2].x,  py2 = points[i+2].y;
        const double px3 = points[i+3].x,  py3 = points[i+3].y;

        int ix0 = (int)(px0 * inv_dx),  iy0 = (int)(py0 * inv_dy);
        int ix1 = (int)(px1 * inv_dx),  iy1 = (int)(py1 * inv_dy);
        int ix2 = (int)(px2 * inv_dx),  iy2 = (int)(py2 * inv_dy);
        int ix3 = (int)(px3 * inv_dx),  iy3 = (int)(py3 * inv_dy);

        if (ix0 >= lNX) ix0 = lNX - 1;
        if (iy0 >= lNY) iy0 = lNY - 1;
        if (ix1 >= lNX) ix1 = lNX - 1;
        if (iy1 >= lNY) iy1 = lNY - 1;
        if (ix2 >= lNX) ix2 = lNX - 1;
        if (iy2 >= lNY) iy2 = lNY - 1;
        if (ix3 >= lNX) ix3 = lNX - 1;
        if (iy3 >= lNY) iy3 = lNY - 1;

        const double lx0 = px0 - ix0 * ldx,  ly0 = py0 - iy0 * ldy;
        const double lx1 = px1 - ix1 * ldx,  ly1 = py1 - iy1 * ldy;
        const double lx2 = px2 - ix2 * ldx,  ly2 = py2 - iy2 * ldy;
        const double lx3 = px3 - ix3 * ldx,  ly3 = py3 - iy3 * ldy;

        const double rx0 = ldx - lx0,  ry0 = ldy - ly0;
        const double rx1 = ldx - lx1,  ry1 = ldy - ly1;
        const double rx2 = ldx - lx2,  ry2 = ldy - ly2;
        const double rx3 = ldx - lx3,  ry3 = ldy - ly3;

        const int b0 = iy0 * lGRID_X + ix0;
        const int b1 = iy1 * lGRID_X + ix1;
        const int b2 = iy2 * lGRID_X + ix2;
        const int b3 = iy3 * lGRID_X + ix3;

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

/* ================================================================
 *  APPROACH A: IMMEDIATE REPLACEMENT
 *
 *  Benchmark findings (Haswell E5-2640 v3):
 *  - Prefetch: flat across PF=0..128 (rand_r dominates), keep PF=32
 *  - Unroll: no benefit (branch + rand_r per particle kills ILP)
 *  - Serial: 172.8 ms / iter for 14M particles
 *  - Parallel: scales 5.8x at 8 threads, NUMA wall at 16
 *  - Beats deferred by 10-40% at all thread counts
 * ================================================================ */

/* --- Serial Immediate --- */
void mover_serial_immediate(Points *points, double deltaX, double deltaY) {
    const int N = NUM_Points;
    const int PF = 32;
    unsigned int seed = 42u;

    for (int i = 0; i < N; i++) {
        if (i + PF < N) {
            _mm_prefetch((const char *)&points[i + PF], _MM_HINT_T0);
        }

        double dx_rand = deltaX * (2.0 * ((double)rand_r(&seed) / RAND_MAX) - 1.0);
        double dy_rand = deltaY * (2.0 * ((double)rand_r(&seed) / RAND_MAX) - 1.0);
        double x_new = points[i].x + dx_rand;
        double y_new = points[i].y + dy_rand;

        if (x_new < 0.0 || x_new > 1.0 || y_new < 0.0 || y_new > 1.0) {
            /* DELETE + INSERT: replace with random position in domain */
            points[i].x = (double)rand_r(&seed) / RAND_MAX;
            points[i].y = (double)rand_r(&seed) / RAND_MAX;
        } else {
            points[i].x = x_new;
            points[i].y = y_new;
        }
    }
}

/* --- Parallel Immediate --- */
void mover_parallel_immediate(Points *points, double deltaX, double deltaY, int nthreads) {
    const int N = NUM_Points;
    const int PF = 32;

    omp_set_num_threads(nthreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int seed = (unsigned int)tid * 1073741827u + 2654435761u;

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            if (i + PF < N) {
                _mm_prefetch((const char *)&points[i + PF], _MM_HINT_T0);
            }

            double dx_rand = deltaX * (2.0 * ((double)rand_r(&seed) / RAND_MAX) - 1.0);
            double dy_rand = deltaY * (2.0 * ((double)rand_r(&seed) / RAND_MAX) - 1.0);
            double x_new = points[i].x + dx_rand;
            double y_new = points[i].y + dy_rand;

            if (x_new < 0.0 || x_new > 1.0 || y_new < 0.0 || y_new > 1.0) {
                points[i].x = (double)rand_r(&seed) / RAND_MAX;
                points[i].y = (double)rand_r(&seed) / RAND_MAX;
            } else {
                points[i].x = x_new;
                points[i].y = y_new;
            }
        }
    }
}

/* ================================================================
 *  APPROACH B: DEFERRED INSERTION
 *
 *  Benchmark findings (Haswell E5-2640 v3):
 *  - Serial swap compaction: 18-21 ms for 14M (dominates all parallel
 *    strategies even at 16 threads — prefix_inplace best parallel is 34ms)
 *  - Parallel compaction adds overhead: extra pass over array + malloc +
 *    barrier + memcpy, all for ~0.2-0.5% deletion rate
 *  - Decision: serial swap for compaction in both serial and parallel
 *    deferred variants. Phase 1 and Phase 3 are parallel.
 * ================================================================ */

/* --- Serial Deferred --- */
void mover_serial_deferred(Points *points, double deltaX, double deltaY) {
    const int N = NUM_Points;
    const int PF = 32;
    unsigned int seed = 42u;
    int deleted_count = 0;

    /* Phase 1: Move and mark deleted (sentinel x = -1.0) */
    for (int i = 0; i < N; i++) {
        if (i + PF < N) {
            _mm_prefetch((const char *)&points[i + PF], _MM_HINT_T0);
        }

        double dx_rand = deltaX * (2.0 * ((double)rand_r(&seed) / RAND_MAX) - 1.0);
        double dy_rand = deltaY * (2.0 * ((double)rand_r(&seed) / RAND_MAX) - 1.0);
        double x_new = points[i].x + dx_rand;
        double y_new = points[i].y + dy_rand;

        if (x_new < 0.0 || x_new > 1.0 || y_new < 0.0 || y_new > 1.0) {
            points[i].x = -1.0;
            points[i].y = -1.0;
            deleted_count++;
        } else {
            points[i].x = x_new;
            points[i].y = y_new;
        }
    }

    /* Phase 2: Compact — two-pointer swap pushes voids to end.
     * Serial swap: 18ms for 14M, faster than any parallel strategy. */
    int left = 0, right = N - 1;
    while (left < right) {
        while (left < right && points[left].x >= 0.0) left++;
        while (left < right && points[right].x < 0.0) right--;
        if (left < right) {
            Points tmp = points[left];
            points[left] = points[right];
            points[right] = tmp;
            left++;
            right--;
        }
    }

    /* Phase 3: Insert new particles at the void positions [N-deleted, N) */
    int start_insert = N - deleted_count;
    for (int i = start_insert; i < N; i++) {
        points[i].x = (double)rand_r(&seed) / RAND_MAX;
        points[i].y = (double)rand_r(&seed) / RAND_MAX;
    }
}

/* --- Parallel Deferred --- */
void mover_parallel_deferred(Points *points, double deltaX, double deltaY, int nthreads) {
    const int N = NUM_Points;
    const int PF = 32;

    omp_set_num_threads(nthreads);

    int total_deleted = 0;

    /* Phase 1: Move and mark (parallel) — embarrassingly parallel */
    #pragma omp parallel reduction(+:total_deleted)
    {
        int tid = omp_get_thread_num();
        unsigned int seed = (unsigned int)tid * 1073741827u + 2654435761u;

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            if (i + PF < N) {
                _mm_prefetch((const char *)&points[i + PF], _MM_HINT_T0);
            }

            double dx_rand = deltaX * (2.0 * ((double)rand_r(&seed) / RAND_MAX) - 1.0);
            double dy_rand = deltaY * (2.0 * ((double)rand_r(&seed) / RAND_MAX) - 1.0);
            double x_new = points[i].x + dx_rand;
            double y_new = points[i].y + dy_rand;

            if (x_new < 0.0 || x_new > 1.0 || y_new < 0.0 || y_new > 1.0) {
                points[i].x = -1.0;
                points[i].y = -1.0;
                total_deleted++;
            } else {
                points[i].x = x_new;
                points[i].y = y_new;
            }
        }
    }

    /* Phase 2: Compact — serial two-pointer swap.
     *
     * Benchmarked all 4 strategies on cluster (E5-2640 v3):
     *   serial_swap:     18 ms  (14M, 0.5% deletion)
     *   prefix_inplace:  34 ms  (14M, 0.5% deletion, 8 threads — best parallel)
     *   prefix_buf:      54 ms  (14M, 0.5% deletion, 8 threads)
     *   atomic_cursor:   912 ms (14M, 0.5% deletion, 8 threads — catastrophic)
     *
     * Serial swap wins because:
     *   - Deletion rate is 0.15-0.5%, so very few swaps needed
     *   - Two-pointer converges quickly from both ends
     *   - No malloc, no barrier, no prefix sum overhead
     *   - Parallel strategies pay full O(N) scan cost just to count/copy
     */
    int left = 0, right = N - 1;
    while (left < right) {
        while (left < right && points[left].x >= 0.0) left++;
        while (left < right && points[right].x < 0.0) right--;
        if (left < right) {
            Points tmp = points[left];
            points[left] = points[right];
            points[right] = tmp;
            left++;
            right--;
        }
    }

    /* Phase 3: Insert new particles (parallel) */
    int start_insert = N - total_deleted;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int seed = (unsigned int)tid * 2654435761u + 1073741827u;

        #pragma omp for schedule(static)
        for (int i = start_insert; i < N; i++) {
            points[i].x = (double)rand_r(&seed) / RAND_MAX;
            points[i].y = (double)rand_r(&seed) / RAND_MAX;
        }
    }
}

/* ================================================================
 *  OLD ASSIGNMENT 04 MOVER — no deletion, periodic wrap-around
 * ================================================================ */
void mover_parallel_no_delete(Points *points, double deltaX, double deltaY, int nthreads) {
    const int N = NUM_Points;
    const int PF = 32;

    omp_set_num_threads(nthreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int seed = (unsigned int)tid * 1073741827u + 2654435761u;

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            if (i + PF < N) {
                _mm_prefetch((const char *)&points[i + PF], _MM_HINT_T0);
            }

            double dx_rand = deltaX * (2.0 * ((double)rand_r(&seed) / RAND_MAX) - 1.0);
            double dy_rand = deltaY * (2.0 * ((double)rand_r(&seed) / RAND_MAX) - 1.0);
            double x_new = points[i].x + dx_rand;
            double y_new = points[i].y + dy_rand;

            /* Periodic wrap-around: x - floor(x) maps any real to [0, 1) */
            x_new = x_new - floor(x_new);
            y_new = y_new - floor(y_new);

            points[i].x = x_new;
            points[i].y = y_new;
        }
    }
}

/* ================================================================
 *  LEGACY WRAPPERS (called by main.cpp)
 *
 *  mover_serial backs onto immediate — benchmarked 172.8 ms vs
 *  191.7 ms deferred (14M particles). No compaction overhead.
 * ================================================================ */
void mover_serial(Points *points, double deltaX, double deltaY) {
    mover_serial_immediate(points, deltaX, deltaY);
}

void mover_parallel(Points *points, double deltaX, double deltaY) {
    mover_parallel_immediate(points, deltaX, deltaY, 4);
}

/* ================================================================
 *  SAVE MESH
 * ================================================================ */
void save_mesh(double *mesh_value) {
    FILE *fp = fopen("mesh_output.csv", "w");
    if (!fp) {
        fprintf(stderr, "Error: cannot open mesh_output.csv for writing\n");
        return;
    }
    for (int j = 0; j < GRID_Y; j++) {
        for (int i = 0; i < GRID_X; i++) {
            if (i > 0) fprintf(fp, ",");
            fprintf(fp, "%.6e", mesh_value[j * GRID_X + i]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}