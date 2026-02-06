# FastKNN Python Interface

Scikit-learn compatible Python interface for the FastKNN high-performance library.

## Installation

### Prerequisites

1. Build the FastKNN C library first:

```bash
cd ..
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON  # Build shared library
make
cd ../python
```

2. Install Python package:

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage (sklearn-compatible)

```python
from fastknn import KNeighborsClassifier, Backend
import numpy as np

# Generate some data
X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
y_train = np.array([0, 0, 1, 1], dtype=np.int32)

# Create classifier (just like sklearn!)
clf = KNeighborsClassifier(n_neighbors=2, backend=Backend.SERIAL)

# Fit
clf.fit(X_train, y_train)

# Predict
X_test = np.array([[0.5, 0.5], [2.5, 2.5]], dtype=np.float32)
predictions = clf.predict(X_test)
print(predictions)  # [0, 1]

# Score
accuracy = clf.score(X_test, np.array([0, 1]))
print(f"Accuracy: {accuracy * 100:.1f}%")  # 100.0%
```

### Using Different Backends

```python
from fastknn import KNeighborsClassifier, Backend

# Serial (single-threaded CPU)
clf = KNeighborsClassifier(n_neighbors=3, backend=Backend.SERIAL)

# OpenMP (multi-threaded CPU)
clf = KNeighborsClassifier(n_neighbors=3, backend=Backend.OPENMP, n_jobs=8)

# CUDA (single GPU)
clf = KNeighborsClassifier(n_neighbors=3, backend=Backend.CUDA, device=0)

# MPI + CUDA (multi-GPU)
clf = KNeighborsClassifier(n_neighbors=3, backend=Backend.CUDA_MPI)
```

### String Backend Names

```python
# Can also use string names
clf = KNeighborsClassifier(n_neighbors=3, backend='cuda')
clf = KNeighborsClassifier(n_neighbors=3, backend='openmp')
```

## Available Backends

| Backend | String Name | Description |
|---------|-------------|-------------|
| `Backend.SERIAL` | `'serial'` | Single-threaded CPU |
| `Backend.OPENMP` | `'openmp'` | Multi-threaded CPU |
| `Backend.MPI_TRAIN_DECOMP` | `'mpi_train'` | MPI train decomposition |
| `Backend.MPI_TEST_DECOMP` | `'mpi_test'` | MPI test decomposition |
| `Backend.CUDA` | `'cuda'` | Single GPU |
| `Backend.CUDA_TRAIN` | `'cuda_train'` | GPU train-optimized |
| `Backend.CUDA_MPI` | `'cuda_mpi'` | Multi-GPU via MPI |
| `Backend.CUDA_MPI_OPENMP` | `'cuda_mpi_openmp'` | All three combined |

## API Reference

### `KNeighborsClassifier`

Scikit-learn compatible KNN classifier.

#### Parameters

- `n_neighbors` (int, default=3): Number of neighbors
- `backend` (Backend or str, default=Backend.SERIAL): Parallel backend
- `n_jobs` (int, optional): Number of OpenMP threads (None = auto)
- `device` (int, optional): CUDA device ID (None = auto)
- `verbose` (int, default=0): Verbosity (0=quiet, 1=normal, 2=debug)

#### Methods

- `fit(X, y)`: Fit the classifier
  - `X`: array-like of shape (n_samples, n_features)
  - `y`: array-like of shape (n_samples,)
  - Returns: self

- `predict(X)`: Predict class labels
  - `X`: array-like of shape (n_samples, n_features)
  - Returns: array of shape (n_samples,)

- `score(X, y)`: Return accuracy
  - `X`: array-like of shape (n_samples, n_features)
  - `y`: array-like of shape (n_samples,)
  - Returns: float in [0, 1]

- `get_backend_name()`: Get backend name
  - Returns: str

## Examples

### Compare with Scikit-learn

```bash
python examples/sklearn_comparison.py
```

Expected output:
```
FastKNN vs Scikit-learn Comparison
======================================================================

Generating data...
Train: (4000, 50), Test: (1000, 50)

======================================================================
FastKNN (Serial)
======================================================================
Fit time:     0.000123s
Predict time: 2.451233s
Total time:   2.451356s
Accuracy:     0.8234

======================================================================
FastKNN (CUDA)
======================================================================
Fit time:     0.000089s
Predict time: 0.098765s
Total time:   0.098854s
Accuracy:     0.8234

======================================================================
Summary (Predict Time)
======================================================================
FastKNN (CUDA)                 0.098765s   24.82x
FastKNN (Serial)               2.451233s    1.00x
```

### MNIST Example

```bash
python examples/mnist_example.py
```

## Performance Tips

1. **Use GPU for large datasets:**
   ```python
   clf = KNeighborsClassifier(backend='cuda')
   ```

2. **Use OpenMP for medium datasets:**
   ```python
   clf = KNeighborsClassifier(backend='openmp', n_jobs=-1)
   ```

3. **Batch predictions are faster:**
   ```python
   # Good: predict 1000 at once
   predictions = clf.predict(X_test)  # X_test: (1000, 784)

   # Bad: predict one at a time
   for x in X_test:
       pred = clf.predict(x.reshape(1, -1))
   ```

4. **Use float32 for GPU backends:**
   ```python
   X = X.astype(np.float32)  # Faster on GPU
   ```

## Compatibility with Scikit-learn

FastKNN implements the scikit-learn estimator interface:

```python
from sklearn.model_selection import cross_val_score
from fastknn import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5, backend='serial')
scores = cross_val_score(clf, X, y, cv=5)
print(f"CV scores: {scores}")
```

### Feature Parity

| Feature | FastKNN | sklearn |
|---------|---------|---------|
| `fit(X, y)` | ✅ | ✅ |
| `predict(X)` | ✅ | ✅ |
| `score(X, y)` | ✅ | ✅ |
| `predict_proba(X)` | ❌ | ✅ |
| `kneighbors(X)` | ❌ | ✅ |
| Distance metrics | Euclidean only | Multiple |
| Algorithms | Brute force only | ball_tree, kd_tree, brute |

FastKNN focuses on **brute-force KNN with extreme performance** through parallelization, rather than approximate methods.

## Troubleshooting

### "FastKNN library not found"

Build the C library first:
```bash
cd ..
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make
```

### "Backend not available"

The requested backend wasn't compiled. Rebuild with appropriate flags:
```bash
cmake .. -DFASTKNN_ENABLE_CUDA=ON
make
```

### Import Error

Make sure numpy is installed:
```bash
pip install numpy
```

## Development

Run tests:
```bash
pip install -e ".[dev]"
pytest
```

## License

Same as FastKNN C library.
