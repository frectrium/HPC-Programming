# Assignment 03: Serial Interpolation on Scattered Data

This assignment implements interpolation of scattered 2D data points onto a structured mesh grid.  
The project includes utilities for generating input data, initializing data structures, running the interpolation pipeline, and verifying correctness using sample test files.

---

## Project Structure

- `input_file_maker.cpp`  : Utility program to generate binary input files containing scattered data points.
- `init.cpp` : Handles data structure initialization.
- `main.cpp` : Contains the main interpolation pipeline.
- `utils.cpp` : Contains helper/utility functions used across the project.
- `init.h`, `utils.h` : Header files for corresponding source files.
- `Test_input.bin` : Sample binary input file containing scattered data points.
- `Test_Mesh.out` : Sample output file showing correct interpolation results for `Test_input.bin`.
---

## Compilation & Execution

### Step 1: Generate Input File

First, compile the input file generator:

```bash
gcc input_file_maker.cpp -o input_maker.out
```

Run the executable:

```bash
./input_maker.out
```

You will be prompted to enter required parameters. This program will generate a binary file named `input.bin`.

---

### Step 2: Compile the Main Program

Compile the interpolation pipeline:

```bash
gcc main.cpp utils.cpp init.cpp -lm -o main.out
```

---

### Step 3: Run the Program

```bash
./main.out input.bin
```

This will read scattered data points from `input.bin`, perform interpolation and store the output mesh grid in `Mesh.out`

---

## Correctness Verification

To verify your implementation, run the program using the provided test file:

```bash
./main.out Test_input.bin
```

After execution, compare your generated `Mesh.out` with `Test_Mesh.out`. If both files match exactly, your implementation is correct.

---

## Workflow 

1. Compile and run `input_file_maker.cpp`
2. Generate `input.bin`
3. Compile main interpolation program
4. Run with input file
5. Compare output with test file for validation

---

