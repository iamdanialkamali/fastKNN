# FastKNN: High-Performance K-Nearest Neighbors Library

FastKNN is a production-grade library for K-Nearest Neighbors classification with multiple parallel backends including OpenMP, MPI, CUDA, and hybrid multi-GPU implementations.

## Features

- **Multiple Backends**: Serial, OpenMP, MPI (train/test decomposition), CUDA, CUDA+MPI, CUDA+MPI+OpenMP
- **Clean C API**: Simple, stable API suitable for integration into other projects
- **CMake Build System**: Feature-gated compilation with automatic dependency detection
- **High Performance**: Achieves up to 1189x speedup vs serial on realistic workloads
- **Production Ready**: Comprehensive tests, benchmarks, and documentation

## Quick Start

### Build (Serial only)

```bash
mkdir build && cd build
cmake .. -DFASTKNN_ENABLE_OPENMP=OFF -DFASTKNN_ENABLE_MPI=OFF -DFASTKNN_ENABLE_CUDA=OFF
make
make test
```

### Build (All backends)

```bash
mkdir build && cd build
cmake ..
make
make test
```

### Run Example

```bash
./examples/knn_serial 10000 1000
# Train Size: 10000 Test Size: 1000 Accuracy: 97.15% Time: 9.707s
```

## System Requirements

### Minimum (Serial only)
- C compiler (GCC, Clang, MSVC)
- CMake 3.18+

### Optional Dependencies
- **OpenMP**: GCC 4.9+, Clang 3.8+, or Intel Compiler
- **MPI**: OpenMPI 3.0+, MPICH 3.2+, or Intel MPI
- **CUDA**: CUDA Toolkit 11.0+ with compute capability 6.0+ GPU

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `FASTKNN_ENABLE_OPENMP` | ON | Enable OpenMP backend |
| `FASTKNN_ENABLE_MPI` | ON | Enable MPI backends |
| `FASTKNN_ENABLE_CUDA` | ON | Enable CUDA backends |
| `FASTKNN_BUILD_EXAMPLES` | ON | Build example programs |
| `FASTKNN_BUILD_TESTS` | ON | Build test suite |
| `FASTKNN_BUILD_BENCH` | ON | Build benchmark harness |

## Building for Specific Backends

### Serial + OpenMP (shared memory only)

```bash
cmake .. -DFASTKNN_ENABLE_MPI=OFF -DFASTKNN_ENABLE_CUDA=OFF
make
./examples/knn_openmp 10000 1000
```

### MPI (distributed CPU)

```bash
cmake .. -DFASTKNN_ENABLE_CUDA=OFF
make
mpirun -np 4 ./examples/knn_mpi 56000 1000
```

### CUDA (single GPU)

```bash
cmake .. -DFASTKNN_ENABLE_MPI=OFF
make
./examples/knn_cuda 56000 1000
```

### CUDA + MPI (multi-GPU)

```bash
cmake ..
make
mpirun -np 4 ./examples/knn_cuda_mpi 56000 5000
```

## Installation

```bash
cd build
sudo make install
```

This installs:
- Headers to `/usr/local/include/fastknn/`
- Library to `/usr/local/lib/`
- CMake package config to `/usr/local/lib/cmake/FastKNN/`
- Examples to `/usr/local/bin/examples/`

## Using the Library in Your Project

### With CMake

```cmake
find_package(fastknn REQUIRED)
add_executable(myapp main.c)
target_link_libraries(myapp fastknn::fastknn)
```

### Example Code

```c
#include <fastknn/fastknn.h>

int main() {
    fastknn_init();

    // Read data
    fastknn_matrix_f32 train, test;
    fastknn_labels_i32 train_labels, test_labels;
    fastknn_matrix_read_csv("train.csv", &train, &train_labels, -1);
    fastknn_matrix_read_csv("test.csv", &test, &test_labels, -1);

    // Configure backend
    fastknn_config config = fastknn_config_default();
    config.k = 3;
    config.backend = FASTKNN_BACKEND_CUDA;

    // Predict
    fastknn_result result;
    fastknn_predict(&train, &train_labels, &test, &config, &result);

    // Evaluate
    float acc = fastknn_accuracy(&result.predictions, &test_labels);
    printf("Accuracy: %.2f%%\n", acc);

    // Cleanup
    fastknn_result_free(&result);
    fastknn_matrix_free(&train);
    fastknn_matrix_free(&test);
    fastknn_labels_free(&train_labels);
    fastknn_labels_free(&test_labels);
    fastknn_cleanup();

    return 0;
}
```

## Running Examples

All examples accept optional command-line arguments:

```bash
<example> [train_size] [test_size]
```

If omitted, reads entire dataset.

### Serial

```bash
./examples/knn_serial 10000 1000
```

### OpenMP

```bash
export OMP_NUM_THREADS=8
./examples/knn_openmp 56000 1000
```

### MPI (train decomposition)

```bash
mpirun -np 4 ./examples/knn_mpi 56000 5000
```

### MPI (test decomposition)

```bash
mpirun -np 4 ./examples/knn_mpi_test 56000 5000
```

### CUDA

```bash
./examples/knn_cuda 56000 5000
```

### CUDA + MPI (multi-GPU)

```bash
mpirun -np 4 ./examples/knn_cuda_mpi 56000 5000
```

## Running Benchmarks

The benchmark tool supports flexible configuration:

```bash
./bench/benchmark --backend cuda --train 56000 --test 5000 --k 3 --output results.json
```

Options:
- `--backend <name>`: serial, openmp, mpi_train, mpi_test, cuda, cuda_mpi
- `--train <N>`: Number of training samples
- `--test <N>`: Number of test samples
- `--k <N>`: Number of neighbors (default: 3)
- `--output <file>`: Output JSON results
- `--repeat <N>`: Number of repetitions for averaging

## Performance Guide

### Backend Selection

| Scenario | Recommended Backend |
|----------|---------------------|
| Small dataset (<10K train) | Serial or OpenMP |
| Medium dataset (10K-50K train) | CUDA (single GPU) |
| Large dataset (>50K train) | CUDA+MPI (multi-GPU) |
| Limited GPU memory | MPI (train decomposition) |
| Batch processing (many test) | MPI (test decomposition) |

### Expected Performance (MNIST, k=3)

| Backend | Train Size | Test Size | Time | Speedup |
|---------|-----------|-----------|------|---------|
| Serial | 56K | 1K | 554s | 1.0x |
| OpenMP (8 cores) | 56K | 1K | 45s | 12.1x |
| CUDA (1 GPU) | 56K | 1K | 20s | 27.5x |
| MPI (4 ranks) | 56K | 1K | 22s | 25.2x |
| CUDA+MPI (4 GPUs) | 56K | 5K | 2.3s | **240x** |

Full details in [docs/blog.md](docs/blog.md).

## Testing

```bash
cd build
make test
# or
ctest -V
```

Tests include:
- Correctness on synthetic 2D data
- Backend consistency (all produce same results)
- Optional MNIST subset test (if data available)

## Troubleshooting

### CMake doesn't find MPI

```bash
cmake .. -DMPI_C_COMPILER=$(which mpicc)
```

### CMake doesn't find CUDA

Ensure `nvcc` is in PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake ..
```

### Build fails with "undefined reference to MPI_*"

MPI was detected but not linked properly. Try:
```bash
cmake .. -DMPI_C_COMPILER=$(which mpicc)
```

### Runtime error: "Backend not available"

The requested backend was not compiled. Check CMake output:
```bash
cmake .. 2>&1 | grep "enabled"
```

### GPU out of memory

Reduce dataset size or use MPI to distribute across multiple GPUs:
```bash
mpirun -np 4 ./examples/knn_cuda_mpi 56000 5000
```

## Project Structure

```
.
├── include/fastknn/     # Public API header
├── src/
│   ├── core/            # Core library (API, I/O, dispatch)
│   └── backend/         # Backend implementations
├── examples/            # CLI tools (drop-in replacements for old binaries)
├── tests/               # Correctness tests
├── bench/               # Benchmark harness
├── docs/                # API docs and blog
├── data/                # MNIST dataset
└── CMakeLists.txt       # Build configuration
```

## Documentation

- [API Reference](docs/API.md)
- [Technical Blog Post](docs/blog.md)
- [Original Report](Report.pdf)

## License

See original project license.

## Citation

If you use this library in research, please cite:

```
Fast KNN on CPUs and GPUs: MPI + OpenMP + CUDA from Scratch
CMSE 822 Final Project, Michigan State University, 2024
```

## Contributing

This is a refactored version of a course project. The original implementations have been preserved in the git history. To see evolution:

```bash
git log --all --oneline --graph
```
