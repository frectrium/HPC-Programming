/*
 * utils.cpp — Parallel PIC Interpolation (OpenMP)
 *
 * Compile-time method selection via -DMETHOD=N:
 *   METHOD=0 : Original serial (no OpenMP needed)
 *   METHOD=1 : Atomic updates on shared mesh
 *   METHOD=2 : Thread-private mesh reduction (simple loop)
 *   METHOD=3 : Thread-private mesh reduction + 4-pt unroll + prefetch  [DEFAULT]
 *
 * Hardware target: 2× Xeon E5-2640 v3 (Haswell), 16 cores, 2 NUMA nodes
 * Compiler: GCC 4.8.2, OpenMP 3.1
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "init.h"
#include "utils.h"

/* Default to method 3 (fastest) if not specified */
#ifndef METHOD
#define METHOD 3
#endif

/* ======================================================================
 * Persistent buffer management for thread-private meshes.
 * Only needed for METHOD 2 and 3 (reduction-based approaches).
 * NUMA-aware: first-touch policy places each buffer on the correct node.
 * ====================================================================== */

#if METHOD >= 2

static double **g_local_mesh  = NULL;
static int      g_num_buffers = 0;
static int      g_buf_size    = 0;

static void ensure_buffers(int nthreads, int mesh_size) {
    if (g_local_mesh != NULL && g_num_buffers == nthreads && g_buf_size == mesh_size)
        return;

    /* Free old buffers if layout changed */
    if (g_local_mesh) {
        for (int t = 0; t < g_num_buffers; t++)
            free(g_local_mesh[t]);
        free(g_local_mesh);
    }

    g_local_mesh  = (double **)malloc(nthreads * sizeof(double *));
    g_num_buffers = nthreads;
    g_buf_size    = mesh_size;

    /* NUMA-aware allocation: each thread touches its own buffer first
     * so the OS maps pages to the local NUMA node (first-touch policy). */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        g_local_mesh[tid] = (double *)malloc(mesh_size * sizeof(double));
        memset(g_local_mesh[tid], 0, mesh_size * sizeof(double));
    }
}

#endif /* METHOD >= 2 */

void interpolation_cleanup(void) {
#if METHOD >= 2
    if (g_local_mesh) {
        for (int t = 0; t < g_num_buffers; t++)
            free(g_local_mesh[t]);
        free(g_local_mesh);
        g_local_mesh  = NULL;
        g_num_buffers = 0;
        g_buf_size    = 0;
    }
#endif
}


/* ======================================================================
 * METHOD 0: Original serial baseline (no OpenMP)
 * ====================================================================== */
#if METHOD == 0

void interpolation(double *mesh_value, Points *points) {
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double ldx = dx, ldy = dy;
    const int lGRID_X = GRID_X;
    const int lNX = NX, lNY = NY;
    const int N = NUM_Points;

    for (int i = 0; i < N; i++) {
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
        mesh_value[base]                += rx * ry;
        mesh_value[base + 1]            += lx * ry;
        mesh_value[base + lGRID_X]      += rx * ly;
        mesh_value[base + lGRID_X + 1]  += lx * ly;
    }
}


/* ======================================================================
 * METHOD 1: Parallel with atomic updates on shared mesh
 *
 * Simplest parallel approach. Each thread processes a chunk of points
 * and uses #pragma omp atomic for the 4 mesh updates per point.
 *
 * Pros:  No extra memory, simple code
 * Cons:  High contention on small grids, atomic overhead per point
 * ====================================================================== */
#elif METHOD == 1

void interpolation(double *mesh_value, Points *points) {
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double ldx = dx, ldy = dy;
    const int lGRID_X = GRID_X;
    const int lNX = NX, lNY = NY;
    const int N = NUM_Points;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
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

        #pragma omp atomic
        mesh_value[base]                += rx * ry;
        #pragma omp atomic
        mesh_value[base + 1]            += lx * ry;
        #pragma omp atomic
        mesh_value[base + lGRID_X]      += rx * ly;
        #pragma omp atomic
        mesh_value[base + lGRID_X + 1]  += lx * ly;
    }
}


/* ======================================================================
 * METHOD 2: Thread-private mesh reduction (simple per-point loop)
 *
 * Each thread accumulates into its own mesh copy, then a parallel
 * merge sums all copies into the output mesh.
 *
 * Pros:  Zero contention during interpolation, simple inner loop
 * Cons:  Extra memory (nthreads × mesh_size), merge overhead
 *
 * Memory per thread for largest config (1001×401):
 *   401,401 × 8B = 3.13 MB — fits in L3 (20 MB/socket)
 * ====================================================================== */
#elif METHOD == 2

void interpolation(double *mesh_value, Points *points) {
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double ldx = dx, ldy = dy;
    const int lGRID_X = GRID_X;
    const int lNX = NX, lNY = NY;
    const int N = NUM_Points;
    const int mesh_size = GRID_X * GRID_Y;

    int nthreads;
    #pragma omp parallel
    { nthreads = omp_get_num_threads(); }

    ensure_buffers(nthreads, mesh_size);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *my_mesh = g_local_mesh[tid];

        /* Zero local mesh (NUMA-local memset) */
        memset(my_mesh, 0, mesh_size * sizeof(double));

        /* ---- Interpolate: each thread handles its chunk of points ---- */
        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
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
            my_mesh[base]                += rx * ry;
            my_mesh[base + 1]            += lx * ry;
            my_mesh[base + lGRID_X]      += rx * ly;
            my_mesh[base + lGRID_X + 1]  += lx * ly;
        }

        /* ---- Parallel merge: sum all local meshes into output ---- */
        #pragma omp for schedule(static)
        for (int j = 0; j < mesh_size; j++) {
            double sum = 0.0;
            for (int t = 0; t < nthreads; t++)
                sum += g_local_mesh[t][j];
            mesh_value[j] = sum;
        }
    }
}


/* ======================================================================
 * METHOD 3: Thread-private reduction + 4-point unroll + prefetch
 *
 * The fastest method. Combines:
 *   1. Thread-private meshes (zero contention)
 *   2. Manual work partitioning (no omp-for overhead in hot loop)
 *   3. 4-point loop unroll for ILP
 *   4. Software prefetch 16 points ahead
 *   5. Interleaved scatter pattern (reduces store-forwarding stalls)
 *   6. NUMA-aware persistent buffers
 *   7. Local copies of globals for register allocation
 *
 * Expected: near-linear speedup up to ~8 cores per socket,
 * then memory-bandwidth-limited scaling on 2nd socket.
 * ====================================================================== */
#elif METHOD == 3

void interpolation(double *mesh_value, Points *points) {
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const int N = NUM_Points;
    const int mesh_size = GRID_X * GRID_Y;

    int nthreads;
    #pragma omp parallel
    { nthreads = omp_get_num_threads(); }

    ensure_buffers(nthreads, mesh_size);

    #pragma omp parallel
    {
        const int tid  = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
        double *my_mesh = g_local_mesh[tid];

        /* Zero local mesh (NUMA-local) */
        memset(my_mesh, 0, mesh_size * sizeof(double));

        /* Local copies for register allocation */
        const double ldx = dx, ldy = dy;
        const int lGRID_X = GRID_X;
        const int lNX = NX, lNY = NY;
        const int PF = 16;   /* prefetch distance in points */

        /* ---- Manual work partition ---- */
        const int chunk = N / nthr;
        const int start = tid * chunk;
        const int end   = (tid == nthr - 1) ? N : start + chunk;
        const int len   = end - start;
        const int n4    = start + (len & ~3);   /* unroll boundary */

        /* ---- 4-point unrolled hot loop ---- */
        int i = start;
        for (; i < n4; i += 4) {
            /* Prefetch future point data into L1 */
            if (i + PF + 3 < end) {
                __builtin_prefetch(&points[i + PF],     0, 3);
                __builtin_prefetch(&points[i + PF + 2], 0, 3);
            }

            /* Load 4 points */
            const double px0 = points[i].x,   py0 = points[i].y;
            const double px1 = points[i+1].x, py1 = points[i+1].y;
            const double px2 = points[i+2].x, py2 = points[i+2].y;
            const double px3 = points[i+3].x, py3 = points[i+3].y;

            /* Cell indices */
            int ix0 = (int)(px0 * inv_dx), iy0 = (int)(py0 * inv_dy);
            int ix1 = (int)(px1 * inv_dx), iy1 = (int)(py1 * inv_dy);
            int ix2 = (int)(px2 * inv_dx), iy2 = (int)(py2 * inv_dy);
            int ix3 = (int)(px3 * inv_dx), iy3 = (int)(py3 * inv_dy);

            /* Clamp */
            if (ix0 >= lNX) ix0 = lNX - 1;  if (iy0 >= lNY) iy0 = lNY - 1;
            if (ix1 >= lNX) ix1 = lNX - 1;  if (iy1 >= lNY) iy1 = lNY - 1;
            if (ix2 >= lNX) ix2 = lNX - 1;  if (iy2 >= lNY) iy2 = lNY - 1;
            if (ix3 >= lNX) ix3 = lNX - 1;  if (iy3 >= lNY) iy3 = lNY - 1;

            /* Local offsets */
            const double lx0 = px0 - ix0*ldx, ly0 = py0 - iy0*ldy;
            const double lx1 = px1 - ix1*ldx, ly1 = py1 - iy1*ldy;
            const double lx2 = px2 - ix2*ldx, ly2 = py2 - iy2*ldy;
            const double lx3 = px3 - ix3*ldx, ly3 = py3 - iy3*ldy;

            const double rx0 = ldx - lx0, ry0 = ldy - ly0;
            const double rx1 = ldx - lx1, ry1 = ldy - ly1;
            const double rx2 = ldx - lx2, ry2 = ldy - ly2;
            const double rx3 = ldx - lx3, ry3 = ldy - ly3;

            /* Base grid indices */
            const int b0 = iy0 * lGRID_X + ix0;
            const int b1 = iy1 * lGRID_X + ix1;
            const int b2 = iy2 * lGRID_X + ix2;
            const int b3 = iy3 * lGRID_X + ix3;

            /* Scatter-accumulate (interleaved to spread cache-line pressure) */
            my_mesh[b0]            += rx0 * ry0;
            my_mesh[b1]            += rx1 * ry1;
            my_mesh[b2]            += rx2 * ry2;
            my_mesh[b3]            += rx3 * ry3;

            my_mesh[b0 + 1]        += lx0 * ry0;
            my_mesh[b1 + 1]        += lx1 * ry1;
            my_mesh[b2 + 1]        += lx2 * ry2;
            my_mesh[b3 + 1]        += lx3 * ry3;

            my_mesh[b0 + lGRID_X]  += rx0 * ly0;
            my_mesh[b1 + lGRID_X]  += rx1 * ly1;
            my_mesh[b2 + lGRID_X]  += rx2 * ly2;
            my_mesh[b3 + lGRID_X]  += rx3 * ly3;

            my_mesh[b0 + lGRID_X + 1] += lx0 * ly0;
            my_mesh[b1 + lGRID_X + 1] += lx1 * ly1;
            my_mesh[b2 + lGRID_X + 1] += lx2 * ly2;
            my_mesh[b3 + lGRID_X + 1] += lx3 * ly3;
        }

        /* ---- Scalar remainder (0-3 points) ---- */
        for (; i < end; i++) {
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
            my_mesh[base]                += rx * ry;
            my_mesh[base + 1]            += lx * ry;
            my_mesh[base + lGRID_X]      += rx * ly;
            my_mesh[base + lGRID_X + 1]  += lx * ly;
        }

        /* implicit barrier from end of parallel region's work section */
        #pragma omp barrier

        /* ---- Parallel merge: sum local meshes into output ---- */
        {
            const int gchunk = mesh_size / nthr;
            const int gstart = tid * gchunk;
            const int gend   = (tid == nthr - 1) ? mesh_size : gstart + gchunk;

            for (int j = gstart; j < gend; j++) {
                double sum = 0.0;
                for (int t = 0; t < nthr; t++)
                    sum += g_local_mesh[t][j];
                mesh_value[j] = sum;
            }
        }
    }
}

#else
#error "Unknown METHOD value. Use 0 (serial), 1 (atomic), 2 (reduction), or 3 (reduction+unrolled)."
#endif


/* ======================================================================
 * Save mesh to "Mesh.out" — unchanged from original
 * ====================================================================== */
void save_mesh(double *mesh_value) {
    FILE *fp = fopen("Mesh.out", "w");
    if (!fp) {
        printf("Error: cannot open Mesh.out for writing\n");
        return;
    }

    for (int j = 0; j < GRID_Y; j++) {
        for (int i = 0; i < GRID_X; i++) {
            if (i > 0) fprintf(fp, " ");
            fprintf(fp, "%.6f", mesh_value[j * GRID_X + i]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}
