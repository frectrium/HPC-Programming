#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "init.h"
#include "utils.h"

// Check for C++11 availability for std::chrono, else fall back to monotonic
#if __cplusplus >= 201103L
    #include <chrono>
    using namespace std::chrono;
    #define USE_CHRONO
#else
    #define CLK CLOCK_MONOTONIC
#endif

int main(int argc, char *argv[]) {
    // DISABLE BUFFERING: Fixes the Python "no output" issue
    setbuf(stdout, NULL);

    if (argc < 2) return 1;
    int kernel_id = atoi(argv[1]);

    // Setup Problem Sizes
    size_t minProbSize = 1 << 8;
    size_t maxProbSize = 1 << 29; 
    size_t totalParticles = maxProbSize;

    double *x, *y, *v, *S;
    
    // Volatile sink to prevent optimization
    volatile double sink_variable = 0.0;

    printf("Np,RUNS,TotalElements,AlgoTime\n");

    for (size_t Np = minProbSize; Np <= maxProbSize; Np *= 2) {
        
        init_vectors(Np, &x, &y, &v, &S);

        // Intelligent RUNS calculation
        size_t RUNS = totalParticles / Np;
        if (RUNS < 1) RUNS = 1;
        // Cap runs for small sizes to save time (2 million is enough)
        if (RUNS > 2000000) RUNS = 2000000; 

        // Warmup
        if (kernel_id == 0) copy_kernel(x, y, Np);

        double alg_time = 0.0;

        #ifdef USE_CHRONO
            // --- METHOD 1: C++11 std::chrono (Cleaner) ---
            auto start = high_resolution_clock::now();
            
            for (size_t r = 0; r < RUNS; r++) {
                switch(kernel_id) {
                    case 0: copy_kernel(x, y, Np); break;
                    case 1: scale_kernel(x, y, 3.0, Np); break;
                    case 2: add_kernel(x, y, S, Np); break;
                    case 3: triad_kernel(x, y, v, S, Np); break;
                    case 4: energy_kernel(v, S, 2.0, Np); break;
                    case 5: triad_memory_only(x, y, v, S, Np); break;
                    case 6: triad_compute_only(Np); break;
                }
            }
            
            auto end = high_resolution_clock::now();
            duration<double> diff = end - start;
            alg_time = diff.count();

        #else
            // --- METHOD 2: POSIX CLOCK_MONOTONIC (Legacy/C style) ---
            struct timespec start, end;
            clock_gettime(CLK, &start);

            for (size_t r = 0; r < RUNS; r++) {
                switch(kernel_id) {
                    case 0: copy_kernel(x, y, Np); break;
                    case 1: scale_kernel(x, y, 3.0, Np); break;
                    case 2: add_kernel(x, y, S, Np); break;
                    case 3: triad_kernel(x, y, v, S, Np); break;
                    case 4: energy_kernel(v, S, 2.0, Np); break;
                    case 5: triad_memory_only(x, y, v, S, Np); break;
                    case 6: triad_compute_only(Np); break;
                }
            }

            clock_gettime(CLK, &end);
            alg_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
        #endif

        // --- THE TRICK ---
        // We must USE the result so the compiler doesn't delete the loop above.
        // We sum a few elements and write to a volatile variable.
        double temp_sum = 0.0;
        // Sample just the first few elements to avoid overhead
        size_t limit = (Np > 5) ? 5 : Np; 
        if (kernel_id == 0 || kernel_id == 1) { // x is output
            for(size_t i=0; i<limit; i++) temp_sum += x[i];
        } else { // S is output
            for(size_t i=0; i<limit; i++) temp_sum += S[i];
        }
        sink_variable = temp_sum; // Compiler cannot optimize this away
        // -----------------

        printf("%lu,%lu,%lu,%.9lf\n", Np, RUNS, Np*RUNS, alg_time);

        free(x); free(y); free(v); free(S);
    }
    return 0;
}