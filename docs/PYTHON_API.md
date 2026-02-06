# FastKNN Python API

## Overview

FastKNN provides a **scikit-learn compatible** Python interface, allowing you to use high-performance parallel backends (OpenMP, MPI, CUDA) with the familiar sklearn API.

```python
from fastknn import KNeighborsClassifier, Backend

# Works just like sklearn!
clf = KNeighborsClassifier(n_neighbors=3, backend=Backend.CUDA)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
```

## Installation

### Step 1: Build C Library

```bash
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON  # Enable shared library
make
cd ..
```

### Step 2: Install Python Package

```bash
cd python
pip install -e .
```

Or with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Examples

### Basic Classification

```python
import numpy as np
from fastknn import KNeighborsClassifier, Backend

# Data
X_train = np.random.randn(1000, 50).astype(np.float32)
y_train = np.random.randint(0, 10, 1000).astype(np.int32)
X_test = np.random.randn(200, 50).astype(np.float32)

# Serial backend
clf = KNeighborsClassifier(n_neighbors=5, backend=Backend.SERIAL)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### GPU Acceleration

```python
# CUDA backend (single GPU)
clf = KNeighborsClassifier(
    n_neighbors=3,
    backend=Backend.CUDA,
    device=0,  # GPU device ID
    verbose=1   # Show timing info
)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### Multi-threaded CPU

```python
# OpenMP backend
clf = KNeighborsClassifier(
    n_neighbors=5,
    backend=Backend.OPENMP,
    n_jobs=8  # Use 8 threads
)
clf.fit(X_train, y_train)
```

### Multi-GPU (MPI)

```python
# CUDA + MPI backend (requires running with mpirun)
clf = KNeighborsClassifier(
    n_neighbors=3,
    backend=Backend.CUDA_MPI
)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

Run with:
```bash
mpirun -np 4 python my_script.py
```

## sklearn Compatibility

FastKNN works with sklearn utilities:

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from fastknn import KNeighborsClassifier

# Cross-validation
clf = KNeighborsClassifier(backend='serial')
scores = cross_val_score(clf, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Grid search
param_grid = {'n_neighbors': [3, 5, 7, 9]}
grid = GridSearchCV(clf, param_grid, cv=3)
grid.fit(X_train, y_train)
print(f"Best k: {grid.best_params_['n_neighbors']}")
```

## Complete API

### Class: `KNeighborsClassifier`

#### Constructor

```python
KNeighborsClassifier(
    n_neighbors=3,
    backend=Backend.SERIAL,
    n_jobs=None,
    device=None,
    verbose=0
)
```

**Parameters:**

- `n_neighbors` (int, default=3)
  Number of neighbors to use for classification.

- `backend` (Backend or str, default=Backend.SERIAL)
  Parallel backend. Can be Backend enum or string:
  - `Backend.SERIAL` or `'serial'`
  - `Backend.OPENMP` or `'openmp'`
  - `Backend.MPI_TRAIN_DECOMP` or `'mpi_train'`
  - `Backend.MPI_TEST_DECOMP` or `'mpi_test'`
  - `Backend.CUDA` or `'cuda'`
  - `Backend.CUDA_TRAIN` or `'cuda_train'`
  - `Backend.CUDA_MPI` or `'cuda_mpi'`
  - `Backend.CUDA_MPI_OPENMP` or `'cuda_mpi_openmp'`

- `n_jobs` (int, optional)
  Number of OpenMP threads. `None` = auto-detect. Only used for OpenMP backend.

- `device` (int, optional)
  CUDA device ID. `None` = auto-select. Only used for CUDA backends.

- `verbose` (int, default=0)
  Verbosity level:
  - 0: Quiet
  - 1: Print timing and configuration
  - 2: Debug output

#### Methods

**`fit(X, y)`**

Fit the classifier.

- **Parameters:**
  - `X`: array-like of shape (n_samples, n_features)
  - `y`: array-like of shape (n_samples,)
- **Returns:** self

**`predict(X)`**

Predict class labels.

- **Parameters:**
  - `X`: array-like of shape (n_samples, n_features)
- **Returns:** ndarray of shape (n_samples,)

**`score(X, y)`**

Return mean accuracy.

- **Parameters:**
  - `X`: array-like of shape (n_samples, n_features)
  - `y`: array-like of shape (n_samples,)
- **Returns:** float in [0, 1]

**`get_backend_name()`**

Get the name of the current backend.

- **Returns:** str

#### Attributes

After calling `fit()`:

- `classes_`: ndarray of unique class labels
- `n_features_in_`: int, number of features

## Backend Selection Guide

| Scenario | Recommended Backend |
|----------|---------------------|
| Small dataset (< 10K samples) | `Backend.SERIAL` or `Backend.OPENMP` |
| Medium dataset (10K-50K) | `Backend.CUDA` (if GPU available) |
| Large dataset (> 50K) | `Backend.CUDA_MPI` (multi-GPU) |
| No GPU available | `Backend.OPENMP` with `n_jobs=-1` |
| Very large test set | `Backend.MPI_TEST_DECOMP` |

## Performance Comparison

Example on 10K train, 1K test, 50 features:

```python
from fastknn import KNeighborsClassifier, Backend
import time

backends = [
    (Backend.SERIAL, "Serial"),
    (Backend.OPENMP, "OpenMP"),
    (Backend.CUDA, "CUDA"),
]

for backend, name in backends:
    clf = KNeighborsClassifier(n_neighbors=5, backend=backend)
    clf.fit(X_train, y_train)

    start = time.time()
    predictions = clf.predict(X_test)
    elapsed = time.time() - start

    print(f"{name:10s} {elapsed:.4f}s")
```

Expected output:
```
Serial     1.2345s
OpenMP     0.1567s  (7.9x faster)
CUDA       0.0523s  (23.6x faster)
```

## Examples

### MNIST Classification

```python
import numpy as np
from fastknn import KNeighborsClassifier, Backend

# Load MNIST (assume data is preprocessed)
X_train = np.load('mnist_train_images.npy')  # (60000, 784)
y_train = np.load('mnist_train_labels.npy')  # (60000,)
X_test = np.load('mnist_test_images.npy')    # (10000, 784)
y_test = np.load('mnist_test_labels.npy')    # (10000,)

# Normalize
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Train with CUDA
clf = KNeighborsClassifier(n_neighbors=3, backend='cuda', verbose=1)
clf.fit(X_train, y_train)

# Predict
accuracy = clf.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from fastknn import KNeighborsClassifier

# Define parameter grid
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11],
}

# Grid search
clf = KNeighborsClassifier(backend='openmp', n_jobs=-1)
grid = GridSearchCV(clf, param_grid, cv=5, verbose=2)
grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")

# Use best estimator
best_clf = grid.best_estimator_
test_accuracy = best_clf.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
```

## Differences from Scikit-learn

| Feature | FastKNN | sklearn KNeighborsClassifier |
|---------|---------|------------------------------|
| Backends | Serial, OpenMP, MPI, CUDA, hybrids | single-threaded, n_jobs for parallelism |
| Algorithm | Brute force only | ball_tree, kd_tree, brute, auto |
| Distance metric | Euclidean only | many (euclidean, manhattan, etc.) |
| Weights | Uniform only | uniform, distance |
| `predict_proba()` | Not implemented | ✅ |
| `kneighbors()` | Not implemented | ✅ |
| Performance | Up to 1000x faster (GPU) | Optimized CPU |

**FastKNN trades flexibility for raw performance.** If you need:
- Different distance metrics → use sklearn
- Probability estimates → use sklearn
- Approximate nearest neighbors → use sklearn with ball_tree/kd_tree

If you need:
- Maximum speed on large datasets → use FastKNN
- GPU acceleration → use FastKNN
- Multi-GPU scaling → use FastKNN

## Troubleshooting

### Library Not Found

```
FileNotFoundError: FastKNN library not found
```

**Solution:** Build the C library with shared libs enabled:

```bash
cd ../build
cmake .. -DBUILD_SHARED_LIBS=ON
make
```

### Backend Not Available

```
RuntimeError: Backend 'CUDA' is not available
```

**Solution:** Rebuild library with CUDA enabled:

```bash
cmake .. -DBUILD_SHARED_LIBS=ON -DFASTKNN_ENABLE_CUDA=ON
make
```

### Import Error

```
ImportError: No module named 'fastknn'
```

**Solution:** Install the package:

```bash
cd python
pip install -e .
```

### NumPy dtype warnings

```
Warning: Converting data to float32
```

**Best practice:** Explicitly cast to float32:

```python
X = X.astype(np.float32)
y = y.astype(np.int32)
```

## Advanced Usage

### Custom Training Loop

```python
from fastknn import KNeighborsClassifier, Backend
import numpy as np

# Incremental learning (refit with more data)
clf = KNeighborsClassifier(n_neighbors=5, backend='cuda')

for batch_X, batch_y in data_loader:
    # Concatenate with existing training data
    if hasattr(clf, '_X_train'):
        X_train = np.vstack([clf._X_train, batch_X])
        y_train = np.concatenate([clf._y_train, batch_y])
    else:
        X_train, y_train = batch_X, batch_y

    clf.fit(X_train, y_train)
```

### Ensemble Methods

```python
from sklearn.ensemble import VotingClassifier
from fastknn import KNeighborsClassifier

# Create ensemble with different k values
clf1 = KNeighborsClassifier(n_neighbors=3, backend='cuda')
clf2 = KNeighborsClassifier(n_neighbors=5, backend='cuda')
clf3 = KNeighborsClassifier(n_neighbors=7, backend='cuda')

ensemble = VotingClassifier(
    estimators=[('knn3', clf1), ('knn5', clf2), ('knn7', clf3)],
    voting='hard'
)

ensemble.fit(X_train, y_train)
accuracy = ensemble.score(X_test, y_test)
```

## See Also

- [C API Reference](API.md)
- [Technical Blog Post](blog.md)
- [Python README](../python/README.md)
- [Example Scripts](../python/examples/)
