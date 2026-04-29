/*
 * Lab_08 hybrid MPI + OpenMP driver.
 *
 * Distribution:
 *   - rank 0 reads input.bin (header + global particle array)
 *   - header (NX, NY, NUM_Points global, Maxiter) is broadcast
 *   - particles are MPI_Scatterv'd; each rank sets the global NUM_Points to
 *     its LOCAL slice size so the (per-rank) kernels in utils.cpp run on
 *     just that slice.
 *   - each rank holds the FULL mesh; after local interpolation we do a
 *     single MPI_Allreduce(SUM) on the mesh, then normalization (with
 *     Allreduce of min/max), mover, denormalize. mover/denorm read-only on
 *     mesh -> no MPI in those phases.
 *
 * Per-phase timings are reported as the max over ranks (the bottleneck rank).
 * Allreduce/Bcast/Scatter wall time is timed separately.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

/* Compute rank's [start, end) slice of [0, total). */
static void slice(int total, int rank, int size, int *start, int *count) {
    int q = total / size, r = total % size;
    *start = rank * q + (rank < r ? rank : r);
    *count = q + (rank < r ? 1 : 0);
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s <input_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int header[4];   /* NX, NY, NUM_Points (global), Maxiter */
    Points *all_points = NULL;
    int global_N = 0;

    if (rank == 0) {
        FILE *file = fopen(argv[1], "rb");
        if (!file) { printf("Error opening input file\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        fread(&header[0], sizeof(int), 1, file);
        fread(&header[1], sizeof(int), 1, file);
        fread(&header[2], sizeof(int), 1, file);
        fread(&header[3], sizeof(int), 1, file);
        global_N = header[2];

        all_points = (Points *) calloc(global_N, sizeof(Points));
        for (int i = 0; i < global_N; i++) {
            fread(&all_points[i].x, sizeof(double), 1, file);
            fread(&all_points[i].y, sizeof(double), 1, file);
            all_points[i].is_void = false;
        }
        fclose(file);
    }

    MPI_Bcast(header, 4, MPI_INT, 0, MPI_COMM_WORLD);
    NX = header[0]; NY = header[1]; global_N = header[2]; Maxiter = header[3];
    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    /* per-rank slice */
    int my_start, my_count;
    slice(global_N, rank, size, &my_start, &my_count);

    /* gather counts/displs at rank 0 for Scatterv */
    int *counts_pts  = NULL;  /* in number of doubles (each Point = 2 doubles + bool, but we'll use a custom MPI type) */
    int *displs_pts  = NULL;
    if (rank == 0) {
        counts_pts = (int *) malloc(size * sizeof(int));
        displs_pts = (int *) malloc(size * sizeof(int));
        for (int r = 0; r < size; r++) {
            int s, c; slice(global_N, r, size, &s, &c);
            counts_pts[r] = c;
            displs_pts[r] = s;
        }
    }

    /* Build a contiguous MPI type for Points (struct with 2 doubles + bool). */
    MPI_Datatype MPI_POINT;
    MPI_Type_contiguous(sizeof(Points), MPI_BYTE, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    Points *local_points = (Points *) calloc(my_count > 0 ? my_count : 1, sizeof(Points));

    double t_scatter0 = MPI_Wtime();
    MPI_Scatterv(all_points, counts_pts, displs_pts, MPI_POINT,
                 local_points, my_count, MPI_POINT,
                 0, MPI_COMM_WORLD);
    double t_scatter = MPI_Wtime() - t_scatter0;

    if (rank == 0) { free(all_points); free(counts_pts); free(displs_pts); }

    /* Per-rank: NUM_Points becomes the LOCAL count (used by utils.cpp kernels). */
    NUM_Points = my_count;

    /* Allocate full mesh on every rank. */
    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));

    double total_int   = 0.0, total_norm = 0.0, total_move = 0.0, total_denorm = 0.0;
    double total_allreduce_mesh = 0.0, total_allreduce_minmax = 0.0;

    /* warm up cache pools (interpolation does this on first call; do an empty
     * iteration first so first-iter timing is not skewed). */
    /* We won't do a warm-up iteration: the test compares results, so just run. */

    for (int iter = 0; iter < Maxiter; iter++) {

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        interpolation(mesh_value, local_points);

        double t1 = MPI_Wtime();

        /* Sum local mesh contributions across all ranks. */
        MPI_Allreduce(MPI_IN_PLACE, mesh_value, GRID_X * GRID_Y,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double t2 = MPI_Wtime();

        /* Normalization: local min/max -> Allreduce -> apply. */
        double mn_local, mx_local;
        mesh_minmax(mesh_value, &mn_local, &mx_local);
        double mn_global, mx_global;
        double tA = MPI_Wtime();
        MPI_Allreduce(&mn_local, &mn_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&mx_local, &mx_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        double tB = MPI_Wtime();
        normalize_with_minmax(mesh_value, mn_global, mx_global);

        double t3 = MPI_Wtime();

        mover(mesh_value, local_points);

        double t4 = MPI_Wtime();

        denormalization(mesh_value);

        double t5 = MPI_Wtime();

        total_int    += (t1 - t0);
        total_allreduce_mesh   += (t2 - t1);
        total_norm   += (t3 - t2);              /* includes Allreduce(min/max) */
        total_allreduce_minmax += (tB - tA);
        total_move   += (t4 - t3);
        total_denorm += (t5 - t4);
    }

    /* Reduce timings: report the MAX across ranks (slowest rank = wall time). */
    double r_int, r_norm, r_move, r_denorm, r_arm, r_amm, r_scat;
    MPI_Reduce(&total_int,    &r_int,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_norm,   &r_norm,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_move,   &r_move,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_denorm, &r_denorm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_allreduce_mesh,   &r_arm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_allreduce_minmax, &r_amm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_scatter,    &r_scat,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Aggregate void count: sum across ranks. */
    long long int local_voids = void_count(local_points);
    long long int global_voids = 0;
    MPI_Reduce(&local_voids, &global_voids, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        save_mesh(mesh_value);
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        printf("MPI ranks = %d, OMP threads/rank = %d, total cores = %d\n",
               size, nthreads, size * nthreads);
        printf("Total Interpolation Time = %lf seconds\n", r_int);
        printf("Total Normalization Time = %lf seconds\n", r_norm);
        printf("Total Mover Time = %lf seconds\n", r_move);
        printf("Total Denormalization Time = %lf seconds\n", r_denorm);
        printf("Total Algorithm Time = %lf seconds\n",
               r_int + r_norm + r_move + r_denorm);
        printf("MPI Scatter time = %lf seconds\n", r_scat);
        printf("MPI Allreduce(mesh) time = %lf seconds\n", r_arm);
        printf("MPI Allreduce(min/max) time = %lf seconds\n", r_amm);
        printf("Total Number of Voids = %lld\n", global_voids);
    }

    MPI_Type_free(&MPI_POINT);
    free(local_points);
    free(mesh_value);
    MPI_Finalize();
    return 0;
}
