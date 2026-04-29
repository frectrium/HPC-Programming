/*
 * Lab_08 hybrid MPI + OpenMP utilities.
 *
 * Per-rank kernels: each rank operates on its LOCAL particle slice (NUM_Points
 * is set per-rank by main.cpp after MPI_Scatterv) and on a FULL grid copy.
 * main.cpp orchestrates the MPI Allreduce of grid contributions and global
 * min/max between calls; the kernels themselves contain no MPI.
 *
 * OpenMP strategy (within a rank):
 *   - Scatter (interpolation): per-thread private grids + parallel cell-wise
 *     reduction. No atomics / locks. Thread-private grids are pooled and
 *     padded to a cache line boundary.
 *   - Normalization: parallel min/max reduction + parallel rescale.
 *   - Mover: read-only grid accesses, independent per-particle writes.
 *   - Denormalization: parallel-for rescale.
 *
 * NUMA tricks:
 *   - mesh_numa_first_touch(): drops the master-thread pages of mesh_value
 *     and re-touches them in parallel so they land on the worker thread's
 *     NUMA node (Linux first-touch policy).
 *   - pts_ensure(): copies the per-rank points buffer into a posix_memalign'd
 *     pool whose pages are first-touched in parallel.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include "utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

double min_val, max_val;

#define CACHE_LINE 64

/* Below this grid size, the parallel-region fork/join cost outweighs the
 * benefit of threading the linear sweeps in normalize/denormalize. */
#define PAR_GRID_MIN 200000

/* ---------- per-thread grid pool ---------- */
static double  *tg_pool    = NULL;
static size_t   tg_stride  = 0;
static int      tg_threads = 0;
static int      tg_n       = 0;

static inline double *tg_get(int tid) { return tg_pool + (size_t)tid * tg_stride; }
static void tg_release(void) {
    if (tg_pool) { free(tg_pool); tg_pool = NULL; }
    tg_stride = 0; tg_threads = 0; tg_n = 0;
}

static void tg_ensure(int N) {
#ifdef _OPENMP
    int t = omp_get_max_threads();
#else
    int t = 1;
#endif
    if (tg_pool && tg_threads == t && tg_n == N) return;
    tg_release();

    size_t elems_per_line = CACHE_LINE / sizeof(double);
    size_t stride = (size_t)N;
    if (stride % elems_per_line) stride += elems_per_line - (stride % elems_per_line);

    size_t total_bytes = (size_t)t * stride * sizeof(double);
    if (posix_memalign((void **)&tg_pool, CACHE_LINE, total_bytes) != 0 || !tg_pool) {
        fprintf(stderr, "tg_ensure: allocation failed (%d threads x %d doubles)\n", t, N);
        exit(1);
    }
    tg_stride  = stride;
    tg_threads = t;
    tg_n       = N;

#ifdef _OPENMP
    #pragma omp parallel num_threads(t)
    {
        const int tid = omp_get_thread_num();
        memset(tg_pool + (size_t)tid * stride, 0, stride * sizeof(double));
    }
#else
    memset(tg_pool, 0, total_bytes);
#endif

    static int registered = 0;
    if (!registered) { atexit(tg_release); registered = 1; }
}

/* ---------- NUMA-local point cache ---------- */
static Points *pts_local    = NULL;
static int     pts_local_n  = 0;
static void   *pts_last_src = NULL;

static void pts_release(void) {
    if (pts_local) { free(pts_local); pts_local = NULL; }
    pts_local_n = 0;
    pts_last_src = NULL;
}

static void pts_ensure(Points *user_points) {
    if (pts_local != NULL && pts_local_n == NUM_Points
        && pts_last_src == (void *)user_points) return;
    pts_release();
    if (NUM_Points <= 0) return;

    if (posix_memalign((void **)&pts_local, CACHE_LINE,
                       (size_t)NUM_Points * sizeof(Points)) != 0 || !pts_local) {
        fprintf(stderr, "pts_ensure: allocation failed (%d points)\n", NUM_Points);
        exit(1);
    }
    pts_local_n  = NUM_Points;
    pts_last_src = (void *)user_points;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < NUM_Points; i++) pts_local[i] = user_points[i];

    static int registered = 0;
    if (!registered) { atexit(pts_release); registered = 1; }
}

/* ---------- mesh NUMA first-touch ---------- */
static void *g_mesh_seen = NULL;

static void mesh_numa_first_touch(double *mesh_value, int N) {
#ifdef _OPENMP
    if ((void *)mesh_value == g_mesh_seen) return;
    g_mesh_seen = (void *)mesh_value;
    if (omp_get_max_threads() < 2) return;

    const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t base = (uintptr_t)mesh_value;
    const size_t   bytes = sizeof(double) * (size_t)N;
    const uintptr_t lo = (base + pg - 1) & ~(pg - 1);
    const uintptr_t hi = (base + bytes) & ~(pg - 1);
    if (hi > lo) (void)madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) mesh_value[i] = 0.0;
#else
    (void)mesh_value; (void)N;
#endif
}

/* -------------------------------------------------------------------------- */
/*  Scatter: particles -> mesh                                                */
/* -------------------------------------------------------------------------- */
void interpolation(double *mesh_value, Points *points) {
    const int N  = GRID_X * GRID_Y;
    const int nx = NX, ny = NY;
    const double DX = dx, DY = dy;
    const double inv_dx = 1.0 / DX;
    const double inv_dy = 1.0 / DY;

    tg_ensure(N);
    mesh_numa_first_touch(mesh_value, N);
    pts_ensure(points);
    Points * __restrict__ pts = pts_local;

#ifdef _OPENMP
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double * __restrict__ tg = tg_get(tid);
        memset(tg, 0, sizeof(double) * (size_t)N);

        #pragma omp for schedule(static)
        for (int p = 0; p < NUM_Points; p++) {
            if (pts[p].is_void) continue;
            const double x = pts[p].x;
            const double y = pts[p].y;

            int ix = (int)(x * inv_dx);
            int iy = (int)(y * inv_dy);
            if (ix >= nx) ix = nx - 1;
            if (iy >= ny) iy = ny - 1;
            if (ix < 0) ix = 0;
            if (iy < 0) iy = 0;

            const double lx = x - ix * DX;
            const double ly = y - iy * DY;
            const double rx = DX - lx;
            const double ry = DY - ly;

            const int base = iy * GRID_X + ix;
            tg[base]              += rx * ry;
            tg[base + 1]          += lx * ry;
            tg[base + GRID_X]     += rx * ly;
            tg[base + GRID_X + 1] += lx * ly;
        }

        const int nt = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            double s = 0.0;
            for (int k = 0; k < nt; k++) s += tg_pool[(size_t)k * tg_stride + i];
            mesh_value[i] = s;
        }
    }
#else
    memset(mesh_value, 0, sizeof(double) * N);
    for (int p = 0; p < NUM_Points; p++) {
        if (pts[p].is_void) continue;
        const double x = pts[p].x;
        const double y = pts[p].y;
        int ix = (int)(x * inv_dx);
        int iy = (int)(y * inv_dy);
        if (ix >= nx) ix = nx - 1;
        if (iy >= ny) iy = ny - 1;
        if (ix < 0) ix = 0;
        if (iy < 0) iy = 0;
        const double lx = x - ix * DX;
        const double ly = y - iy * DY;
        const double rx = DX - lx;
        const double ry = DY - ly;
        const int base = iy * GRID_X + ix;
        mesh_value[base]              += rx * ry;
        mesh_value[base + 1]          += lx * ry;
        mesh_value[base + GRID_X]     += rx * ly;
        mesh_value[base + GRID_X + 1] += lx * ly;
    }
#endif
}

/* -------------------------------------------------------------------------- */
/*  Min/max + normalize (split for MPI Allreduce between the two)             */
/* -------------------------------------------------------------------------- */
void mesh_minmax(const double *mesh_value, double *out_min, double *out_max) {
    const int N = GRID_X * GRID_Y;
    double mn =  DBL_MAX;
    double mx = -DBL_MAX;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) \
            reduction(min:mn) reduction(max:mx) if(N >= PAR_GRID_MIN)
#endif
    for (int i = 0; i < N; i++) {
        const double v = mesh_value[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    *out_min = mn;
    *out_max = mx;
}

void normalize_with_minmax(double *mesh_value, double mn, double mx) {
    const int N = GRID_X * GRID_Y;
    min_val = mn;
    max_val = mx;
    const double range = mx - mn;
    if (range == 0.0) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(N >= PAR_GRID_MIN)
#endif
        for (int i = 0; i < N; i++) mesh_value[i] = 0.0;
        return;
    }
    const double inv_range = 1.0 / range;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(N >= PAR_GRID_MIN)
#endif
    for (int i = 0; i < N; i++) {
        mesh_value[i] = 2.0 * (mesh_value[i] - mn) * inv_range - 1.0;
    }
}

void normalization(double *mesh_value) {
    double mn, mx;
    mesh_minmax(mesh_value, &mn, &mx);
    normalize_with_minmax(mesh_value, mn, mx);
}

/* -------------------------------------------------------------------------- */
/*  Mover: gather + position update                                           */
/* -------------------------------------------------------------------------- */
void mover(double *mesh_value, Points *points) {
    const int nx = NX, ny = NY;
    const double DX = dx, DY = dy;
    const double inv_dx = 1.0 / DX;
    const double inv_dy = 1.0 / DY;

    pts_ensure(points);
    Points * __restrict__ pts = pts_local;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int p = 0; p < NUM_Points; p++) {
        if (pts[p].is_void) continue;
        const double x = pts[p].x;
        const double y = pts[p].y;

        int ix = (int)(x * inv_dx);
        int iy = (int)(y * inv_dy);
        if (ix >= nx) ix = nx - 1;
        if (iy >= ny) iy = ny - 1;
        if (ix < 0) ix = 0;
        if (iy < 0) iy = 0;

        const double lx = x - ix * DX;
        const double ly = y - iy * DY;
        const double rx = DX - lx;
        const double ry = DY - ly;

        const double w00 = rx * ry;
        const double w10 = lx * ry;
        const double w01 = rx * ly;
        const double w11 = lx * ly;

        const int base = iy * GRID_X + ix;
        const double Fi = w00 * mesh_value[base]
                        + w10 * mesh_value[base + 1]
                        + w01 * mesh_value[base + GRID_X]
                        + w11 * mesh_value[base + GRID_X + 1];

        const double xn = x + Fi * DX;
        const double yn = y + Fi * DY;

        if (xn < 0.0 || xn >= 1.0 || yn < 0.0 || yn >= 1.0) {
            pts[p].is_void = true;
        } else {
            pts[p].x = xn;
            pts[p].y = yn;
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Denormalization                                                           */
/* -------------------------------------------------------------------------- */
void denormalization(double *mesh_value) {
    const int N = GRID_X * GRID_Y;
    const double range = max_val - min_val;
    if (range == 0.0) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(N >= PAR_GRID_MIN)
#endif
        for (int i = 0; i < N; i++) mesh_value[i] = min_val;
        return;
    }
    const double half_range = 0.5 * range;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(N >= PAR_GRID_MIN)
#endif
    for (int i = 0; i < N; i++) {
        mesh_value[i] = (mesh_value[i] + 1.0) * half_range + min_val;
    }
}

/* -------------------------------------------------------------------------- */
/*  Void counting + output                                                    */
/* -------------------------------------------------------------------------- */
long long int void_count(Points *points) {
    Points *src = (pts_local && pts_last_src == (void *)points) ? pts_local : points;
    long long int voids = 0;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) reduction(+:voids)
#endif
    for (int i = 0; i < NUM_Points; i++) {
        voids += (long long int)src[i].is_void;
    }
    return voids;
}

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
