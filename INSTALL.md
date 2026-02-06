# FastKNN Installation Guide

**Status: ✅ Tested and Working**

This guide provides tested, step-by-step instructions for installing and using FastKNN.

## Prerequisites

**Required:**
- CMake 3.18+
- C/C++ compiler (gcc recommended)
- Python 3.7+
- NumPy

**Optional (for accelerated backends):**
- OpenMP (usually included with gcc)
- MPI (OpenMPI or MPICH)
- CUDA Toolkit 11.0+ with compute capability 7.0+

## Installation (3 Steps)

### Step 1: Build the C Library

```bash
# Navigate to library directory
cd fastknn-library

# Create and enter build directory
mkdir build && cd build

# Configure with CMake
# Note: We disable CUDA by default for compatibility
cmake .. -DBUILD_SHARED_LIBS=ON -DFASTKNN_ENABLE_CUDA=OFF

# Build (use -j8 for parallel build with 8 cores)
make -j8

# Return to library root
cd ..
```

**Expected output:**
```
-- The C compiler identification is GNU X.X.X
-- Found OpenMP_C: -fopenmp (found version "4.5")
-- OpenMP enabled
-- Could NOT find MPI_C (missing: ...)
-- ==================================================
-- FastKNN Configuration Summary
-- ==================================================
-- Version: 1.0.0
-- Library type: ON
-- OpenMP backend: TRUE
-- MPI backends: FALSE
-- CUDA backends: FALSE
-- ==================================================
[100%] Built target fastknn
```

**Verify build:**
```bash
ls build/libfastknn.so  # Should exist (~26KB)
```

### Step 2: Install Python Package

```bash
cd python
pip install -e .
cd ..
```

**Expected output:**
```
Looking in indexes: https://pypi.org/simple
Obtaining file:///path/to/fastknn-library/python
...
Successfully installed fastknn-1.0.0
```

**Verify installation:**
```bash
python -c "from fastknn import KNeighborsClassifier; print('✓ Import successful!')"
```

### Step 3: Run Tests

```bash
# Quick test (30 seconds, 1K train / 100 test)
python test_mnist_quick.py
```

**Expected output:**
```
FastKNN - MNIST Quick Test
============================================================

Loading data...
  Train: (1000, 784) Test: (100, 784)

Testing Serial Backend
  Fit time: 0.0018s
  Predict time: 0.2711s
  Accuracy: 89.00%

Testing OpenMP Backend
  Fit time: 0.0019s
  Predict time: 0.0684s
  Accuracy: 89.00%
  Speedup vs Serial: 3.96x

✓ All tests passed!
```

## Verify Installation

### Test 1: Small Synthetic Data

```python
from fastknn import KNeighborsClassifier
import numpy as np

X_train = np.array([[1, 1], [1.5, 1.2], [5, 5], [5.5, 5.2]], dtype=np.float32)
y_train = np.array([0, 0, 1, 1], dtype=np.int32)
X_test = np.array([[1.2, 1.1], [5.2, 5.1]], dtype=np.float32)

clf = KNeighborsClassifier(n_neighbors=2, backend='serial')
clf.fit(X_train, y_train)
print(clf.predict(X_test))  # Should print: [0 1]
```

### Test 2: MNIST Example

```bash
# Test with small MNIST subset (500 train, 50 test)
python example_mnist.py --train 500 --test 50 --backend serial
```

**Expected:**
```
Results:
  Accuracy: 86.00%
  Total time: 0.0688s

✓ EXPERIMENT COMPLETE
```

### Test 3: OpenMP Backend

```bash
python example_mnist.py --train 1000 --test 100 --backend openmp
```

**Expected speedup:** 3-4x faster than serial.

## Build Options

### Minimal Build (Serial Only)

No dependencies required:

```bash
cmake .. -DBUILD_SHARED_LIBS=ON \
         -DFASTKNN_ENABLE_OPENMP=OFF \
         -DFASTKNN_ENABLE_MPI=OFF \
         -DFASTKNN_ENABLE_CUDA=OFF
make
```

**Backends available:** Serial only

### CPU Parallelism (Serial + OpenMP)

Requires: OpenMP (usually included with gcc)

```bash
cmake .. -DBUILD_SHARED_LIBS=ON \
         -DFASTKNN_ENABLE_MPI=OFF \
         -DFASTKNN_ENABLE_CUDA=OFF
make
```

**Backends available:** Serial, OpenMP

### With CUDA (GPU Acceleration)

Requires: CUDA Toolkit 11.0+, GPU with compute capability 7.0+

```bash
cmake .. -DBUILD_SHARED_LIBS=ON
make
```

**Backends available:** Serial, OpenMP, CUDA (if available)

### Full Build (All Backends)

Requires: OpenMP, MPI, CUDA

```bash
cmake .. -DBUILD_SHARED_LIBS=ON \
         -DFASTKNN_ENABLE_OPENMP=ON \
         -DFASTKNN_ENABLE_MPI=ON \
         -DFASTKNN_ENABLE_CUDA=ON
make
```

CMake auto-detects which dependencies are available.

## Troubleshooting

### Problem: Library not found during import

**Error:**
```
OSError: libfastknn.so: cannot open shared object file
```

**Solution:**
```bash
# Make sure you built with -DBUILD_SHARED_LIBS=ON
cd build
rm -rf *
cmake .. -DBUILD_SHARED_LIBS=ON -DFASTKNN_ENABLE_CUDA=OFF
make
```

### Problem: CUDA architecture not supported

**Error:**
```
nvcc fatal: Unsupported gpu architecture 'compute_60'
```

**Solution:** Build without CUDA:
```bash
cmake .. -DBUILD_SHARED_LIBS=ON -DFASTKNN_ENABLE_CUDA=OFF
make
```

### Problem: Data files not found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: '../data/train_data.csv'
```

**Solution:** Make sure you're running from the `fastknn-library/` directory:
```bash
cd /path/to/fastknn-library
ls ../data/train_data.csv  # Should exist
python test_mnist_quick.py
```

### Problem: Import error after installation

**Error:**
```
ModuleNotFoundError: No module named 'fastknn'
```

**Solution:** Reinstall the package:
```bash
cd python
pip uninstall fastknn -y
pip install -e .
```

### Problem: Undefined symbol errors

**Error:**
```
OSError: undefined symbol: fastknn_predict_cuda_mpi_openmp
```

**Solution:** Rebuild the library to include all stubs:
```bash
cd build
make clean
cmake .. -DBUILD_SHARED_LIBS=ON -DFASTKNN_ENABLE_CUDA=OFF
make
```

## Uninstallation

```bash
# Remove Python package
pip uninstall fastknn

# Remove built library
rm -rf build

# Remove Python package build artifacts
rm -rf python/fastknn.egg-info
rm -rf python/build
```

## Next Steps

After successful installation:

1. **Read the tutorial:** `TUTORIAL_SUMMARY.md`
2. **Check the API docs:** `docs/PYTHON_API.md`
3. **Read the technical blog:** `docs/blog.md`
4. **Try larger datasets:** Increase `--train` and `--test` in examples

## Quick Reference

**Build commands:**
```bash
cd fastknn-library
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DFASTKNN_ENABLE_CUDA=OFF
make -j8
cd ..
```

**Install Python package:**
```bash
cd python && pip install -e . && cd ..
```

**Run tests:**
```bash
python test_mnist_quick.py              # Quick test
python example_mnist.py --train 1000    # Full example
```

**Basic usage:**
```python
from fastknn import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3, backend='openmp')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

---

**Need help?** See `README.md` or `TUTORIAL_SUMMARY.md`
