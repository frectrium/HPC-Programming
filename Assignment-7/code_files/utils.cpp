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

/* ----------------------------------------------------------------------------
 * Parallel Particle-in-Cell pipeline
 *
 * Strategy summary:
 *   - Scatter (interpolation): per-thread private grids + parallel cell-wise
 *     reduction. Eliminates race conditions without atomics or locks.
 *   - Normalization: parallel min/max reduction + parallel rescale.
 *   - Mover: read-only grid accesses, independent per-particle writes -> a
 *     trivial parallel-for.
 *   - Denormalization: parallel-for rescale.
 *
 * Thread-local grids are allocated lazily once (the first call) and reused
 * across all iterations of the time loop. Each is padded to a multiple of a
 * cache line so adjacent thread buffers cannot contend.
 * -------------------------------------------------------------------------- */

#define CACHE_LINE 64

/* Grid sizes below this threshold go serial for the norm/denorm sweeps: the
 * fork-join cost of an omp parallel region dwarfs 200k doubles of linear
 * memory traffic on modern CPUs, so spawning 16 threads there is a pessim. */
#define PAR_GRID_MIN 200000

static double  *tg_pool    = NULL;  /* contiguous pool of per-thread grids    */
static size_t   tg_stride  = 0;     /* number of doubles per thread (padded)  */
static int      tg_threads = 0;     /* number of per-thread grids allocated   */
static int      tg_n       = 0;     /* logical grid length (GRID_X*GRID_Y)    */

static inline double *tg_get(int tid) {
    return tg_pool + (size_t)tid * tg_stride;
}

static void tg_release(void) {
    if (tg_pool) { free(tg_pool); tg_pool = NULL; }
    tg_stride = 0; tg_threads = 0; tg_n = 0;
}

/* ----------------------------------------------------------------------------
 * NUMA first-touch trick for the output mesh
 *
 * main.cpp calls calloc() on mesh_value from the master thread, which on
 * Linux first-touches every page and binds it to NUMA node 0. With 16 worker
 * threads spread across two sockets, half the threads then pay remote-DRAM
 * traffic on every mesh access. We can't modify main.cpp, but we can drop the
 * pages with madvise(MADV_DONTNEED) (legal because the mesh is re-zeroed
 * every iteration anyway) and re-touch them in parallel. The subsequent page
 * faults use the standard first-touch policy and place each page on the node
 * of the thread that touched it. This is a zero-libnuma NUMA optimization.
 * -------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * NUMA-local particle cache
 *
 * main.cpp allocates the points array with a single calloc() on the master
 * thread, and read_points() streams the whole file into it. The entire array
 * therefore lives on NUMA node 0. For configs with millions of particles the
 * array comfortably exceeds any single socket's L3, so every iteration the
 * 8 worker threads on node 1 stream it over QPI.
 *
 * We break this by copying the user's points into our own posix_memalign'd
 * buffer on the first interpolation call, first-touching each page in
 * parallel so every slice lives on its worker thread's own node. From there
 * on, scatter/mover read and write this internal array. void_count is also
 * rewired to read it, so nothing downstream of utils ever touches the
 * original main.cpp buffer again.
 * -------------------------------------------------------------------------- */

static Points *pts_local    = NULL;
static int     pts_local_n  = 0;
static void   *pts_last_src = NULL;

static void pts_release(void) {
    if (pts_local) { free(pts_local); pts_local = NULL; }
    pts_local_n = 0;
    pts_last_src = NULL;
}

static void pts_ensure(Points *user_points) {
    /* Reset if either never initialized or a different points array came in
     * (defensive; main.cpp passes the same pointer throughout). */
    if (pts_local != NULL && pts_local_n == NUM_Points
        && pts_last_src == (void *)user_points) return;

    pts_release();

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

static void *g_mesh_seen = NULL;

static void mesh_numa_first_touch(double *mesh_value, int N) {
#ifdef _OPENMP
    if ((void *)mesh_value == g_mesh_seen) return;
    g_mesh_seen = (void *)mesh_value;
    if (omp_get_max_threads() < 2) return;

    const size_t pg = (size_t)sysconf(_SC_PAGESIZE);
    const uintptr_t base = (uintptr_t)mesh_value;
    const size_t   bytes = sizeof(double) * (size_t)N;
    const uintptr_t lo = (base + pg - 1) & ~(pg - 1);                 /* round up */
    const uintptr_t hi = (base + bytes) & ~(pg - 1);                  /* round down */
    if (hi > lo) {
        /* Drop only whole, aligned, interior pages so we don't stomp on
         * neighboring heap metadata. Safe because calloc() returned these
         * pages and they're entirely owned by mesh_value. */
        (void)madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);
    }
    /* Touch every cell in parallel; static schedule matches the scatter
     * reduction and normalization sweeps, so each thread owns the same
     * cache-contiguous slice across phases. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) mesh_value[i] = 0.0;
#else
    (void)mesh_value; (void)N;
#endif
}

static void tg_ensure(int N) {
#ifdef _OPENMP
    int t = omp_get_max_threads();
#else
    int t = 1;
#endif
    if (tg_pool && tg_threads == t && tg_n == N) return;

    tg_release();

    /* Pad per-thread stride up to a cache line to kill false sharing on the
     * boundary between threads' buffers. */
    size_t elems_per_line = CACHE_LINE / sizeof(double);
    size_t stride = (size_t)N;
    if (stride % elems_per_line) stride += elems_per_line - (stride % elems_per_line);

    /* posix_memalign keeps the buffer page-aligned but leaves pages untouched,
     * so first-touch is determined by whichever thread writes first. We then
     * parallel-memset each thread's slice so its pages land on that thread's
     * own NUMA node. calloc() would have first-touched everything on master. */
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

    /* Register a one-shot cleanup at exit. */
    static int registered = 0;
    if (!registered) { atexit(tg_release); registered = 1; }
}

/* -------------------------------------------------------------------------- */
/*  Scatter: particles -> mesh (bilinear)                                     */
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

        /* Zero this thread's private grid. */
        memset(tg, 0, sizeof(double) * (size_t)N);

        /* Scatter pass. schedule(static) is fine since work per particle is
         * uniform; even as particles go void, the balance barely drifts. */
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

            const double w00 = rx * ry;
            const double w10 = lx * ry;
            const double w01 = rx * ly;
            const double w11 = lx * ly;

            const int base = iy * GRID_X + ix;
            tg[base]              += w00;
            tg[base + 1]          += w10;
            tg[base + GRID_X]     += w01;
            tg[base + GRID_X + 1] += w11;
        }

        /* Cell-wise reduction: each thread accumulates a contiguous slice of
         * the output grid by summing over all thread-private grids at the
         * same cell index. The outer stride is chosen so thread blocks of the
         * pool are accessed sequentially -> streaming reads, one write. */
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
/*  Normalization: rescale grid to [-1, 1]                                    */
/* -------------------------------------------------------------------------- */
void normalization(double *mesh_value) {
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

/* -------------------------------------------------------------------------- */
/*  Mover: gather from grid, update particle positions                        */
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
/*  Denormalization: invert normalization                                     */
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
    /* Read from the NUMA-local cache if the mover has been running there;
     * the user's original array may be stale since pts_ensure() is the only
     * path we write back through. Fall back to the argument if we never
     * cached (i.e. someone called void_count before interpolation). */
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
