# FastKNN: High-Performance K-Nearest Neighbors Library

A scikit-learn compatible KNN library with **GPU and multi-GPU acceleration**, achieving up to **200× speedup** with CUDA on large datasets.

## Features

- 🚀 **Scikit-learn compatible API** - drop-in replacement
- ⚡ **Multiple backends**: Serial, OpenMP, CUDA, Multi-GPU (CUDA+MPI)
- 🎯 **Up to 200× speedup** with CUDA on large datasets
- 💎 **Multi-GPU support** for massive datasets (2-3× additional speedup)
- 📦 **Easy installation** - Python package with C backend
- 🔧 **Minimal dependencies** - works with or without GPU

---

## 📓 Interactive Tutorial (START HERE!)

**Try our Jupyter notebook with beautiful visualizations:**

```bash
# Install Jupyter and visualization packages
pip install jupyter matplotlib seaborn pandas

# Launch the tutorial notebook
jupyter notebook FastKNN_Tutorial.ipynb
```

**What's inside:**
- 🎨 Beautiful speedup visualizations comparing all backends
- 📊 Performance analysis on 30K+ MNIST samples
- 📈 Scalability analysis (1K to 50K samples)
- 🔬 Interactive parameter tuning
- 💎 Multi-GPU performance demonstration
- 🎯 Confusion matrix and accuracy analysis

See [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) for detailed instructions.

---

## Quick Start

### Step 1: Build the C Library

```bash
# From the fastknn-library/ directory
mkdir build && cd build

# Build with CUDA support (recommended)
cmake .. -DBUILD_SHARED_LIBS=ON -DFASTKNN_ENABLE_CUDA=ON
make -j8

# OR build without CUDA (CPU-only)
cmake .. -DBUILD_SHARED_LIBS=ON -DFASTKNN_ENABLE_CUDA=OFF
make -j8

cd ..
```

**Expected output:**
```
-- OpenMP backend: TRUE   ✓
-- MPI backends: FALSE    (optional, requires MPI)
-- CUDA backends: TRUE    ✓
[100%] Built target fastknn
```

### Step 2: Install Python Package

```bash
cd python
pip install -e .
cd ..
```

### Step 3: Try It!

**Option A: Interactive Tutorial (Recommended)**
```bash
jupyter notebook FastKNN_Tutorial.ipynb
```

**Option B: Run MNIST Example**
```bash
# 30K samples - shows CUDA dominance
python example_mnist.py --train 30000 --test 2000 --backend cuda
```

---

## Usage (Just Like Scikit-learn!)

### Basic Usage

```python
from fastknn import KNeighborsClassifier
import numpy as np

# Load your data
X_train = np.random.randn(30000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 30000).astype(np.int32)
X_test = np.random.randn(2000, 784).astype(np.float32)
y_test = np.random.randint(0, 10, 2000).astype(np.int32)

# Create and train classifier (choose backend)
clf = KNeighborsClassifier(n_neighbors=3, backend='cuda')
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

### Choosing the Right Backend

```python
# Small datasets (<5K): Use OpenMP
clf = KNeighborsClassifier(n_neighbors=3, backend='openmp')

# Large datasets (>20K): Use CUDA for massive speedup
clf = KNeighborsClassifier(n_neighbors=3, backend='cuda', device=0)

# Very large datasets (>100K): Use Multi-GPU
clf = KNeighborsClassifier(n_neighbors=3, backend='cuda_mpi')
# Run with: mpirun -np 3 python your_script.py
```

### Works with Scikit-learn

```python
from sklearn.model_selection import cross_val_score
from fastknn import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3, backend='cuda')
scores = cross_val_score(clf, X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
```

---

## Available Backends

| Backend | Hardware | Best For | Typical Speedup |
|---------|----------|----------|-----------------|
| `serial` | CPU (1 core) | Testing, small datasets | 1× (baseline) |
| `openmp` | CPU (multi-core) | Medium datasets (1K-20K) | 8-12× |
| `cuda` | Single GPU | Large datasets (20K+) | 50-200× |
| `cuda_mpi` | Multi-GPU | Very large datasets (100K+) | 100-500× |

---

## Performance

### Scalability (MNIST Dataset)

**Test Setup:** MNIST digits (784 features), k=3

| Train Size | OpenMP (8 cores) | CUDA (RTX A6000) | CUDA Speedup |
|-----------|------------------|-------------------|--------------|
| 1K | 0.2s | 0.3s | 0.7× (overhead) |
| 5K | 2.5s | 0.4s | 6× |
| 10K | 9s | 0.5s | 18× |
| 20K | 36s | 0.7s | **51×** |
| 30K | 81s | 0.9s | **90×** |
| 50K | 225s | 1.5s | **150×** |

**Key Insight:** CUDA dominates for datasets >20K samples. For smaller datasets, OpenMP is faster due to GPU overhead.

### Multi-GPU Performance (3 GPUs)

| Train Size | Single GPU | 3 GPUs (CUDA+MPI) | Multi-GPU Speedup |
|-----------|-----------|-------------------|-------------------|
| 15K | 0.54s | 0.21s | 2.5× |
| 50K | 1.5s | 0.6s | 2.5× |
| 100K | 3.0s | 1.2s | 2.5× |

**Parallel Efficiency:** ~83% (excellent for 3 GPUs)

---

## Requirements

### Core Requirements
- Python 3.7+
- NumPy
- CMake 3.18+
- C/C++ compiler (gcc, clang, or MSVC)

### Optional (for different backends)
- **OpenMP**: Usually included with compiler
- **CUDA**: CUDA Toolkit 12.0+ (for GPU acceleration)
- **MPI**: OpenMPI or MPICH (for multi-GPU)

### For Tutorial
- Jupyter
- Matplotlib
- Seaborn
- Pandas

```bash
pip install jupyter matplotlib seaborn pandas
```

---

## Build Options

### Maximum Performance (All Backends)
```bash
cmake .. -DBUILD_SHARED_LIBS=ON \
         -DFASTKNN_ENABLE_OPENMP=ON \
         -DFASTKNN_ENABLE_CUDA=ON \
         -DFASTKNN_ENABLE_MPI=ON
```

### GPU Acceleration Only
```bash
cmake .. -DBUILD_SHARED_LIBS=ON \
         -DFASTKNN_ENABLE_CUDA=ON \
         -DFASTKNN_ENABLE_MPI=OFF
```

### CPU Only (No GPU)
```bash
cmake .. -DBUILD_SHARED_LIBS=ON \
         -DFASTKNN_ENABLE_OPENMP=ON \
         -DFASTKNN_ENABLE_CUDA=OFF \
         -DFASTKNN_ENABLE_MPI=OFF
```

### Minimal Build (Serial Only)
```bash
cmake .. -DBUILD_SHARED_LIBS=ON \
         -DFASTKNN_ENABLE_OPENMP=OFF \
         -DFASTKNN_ENABLE_CUDA=OFF \
         -DFASTKNN_ENABLE_MPI=OFF
```

---

## Troubleshooting

### Library Not Found Error

```bash
# Ensure you built with shared library enabled
cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j8
cd ..
```

### CUDA Architecture Not Supported

If you see `Unsupported gpu architecture 'compute_XX'`:

```bash
# Build without CUDA
cmake .. -DBUILD_SHARED_LIBS=ON -DFASTKNN_ENABLE_CUDA=OFF
make -j8
```

Or specify your GPU architecture in CMakeLists.txt (line 49).

### Import Error

```bash
# Reinstall Python package
cd python
pip install -e .
cd ..
```

### MNIST Data Not Found

The example script expects MNIST data in `../data/`:

```bash
ls ../data/train_data.csv  # Should exist
ls ../data/test_data.csv   # Should exist
```

If missing, check the parent directory or adjust the path in `example_mnist.py`.

---

## Documentation

- **[FastKNN_Tutorial.ipynb](FastKNN_Tutorial.ipynb)** - 📓 Interactive tutorial (START HERE!)
- **[NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md)** - Guide for using the notebook
- **[INSTALL.md](INSTALL.md)** - Detailed installation instructions
- **[README_LIBRARY.md](README_LIBRARY.md)** - C library API reference
- **[docs/PYTHON_API.md](docs/PYTHON_API.md)** - Python API documentation
- **[docs/blog.md](docs/blog.md)** - Technical deep dive
- **[example_mnist.py](example_mnist.py)** - Example usage script

---

## Project Structure

```
fastknn-library/
├── include/fastknn/      # Public C API headers
├── src/                  # Library implementation
│   ├── core/             # Core algorithms
│   └── backend/          # Parallel backends
├── python/               # Python bindings
│   ├── fastknn/          # Python package
│   └── examples/         # Python examples
├── docs/                 # Documentation
├── cmake/                # CMake modules
└── FastKNN_Tutorial.ipynb # Interactive tutorial
```

---

## About This Project

This library was developed as part of the **CMSE 822: Parallel Computing** course at Michigan State University, taught by **Dr. Sean Couch**. The project demonstrates practical applications of parallel computing techniques across multiple paradigms:

- **Shared Memory:** OpenMP for multi-core CPU parallelization
- **Distributed Memory:** MPI for multi-GPU coordination
- **GPU Computing:** CUDA for massive parallel acceleration
- **Hybrid Approaches:** CUDA+MPI for multi-GPU systems

### Acknowledgments

- **Dr. Sean Couch** - For an excellent course on parallel computing that inspired this project
- **CMSE 822** - Parallel Computing course at Michigan State University
- **Claude (Anthropic)** - For assistance in refactoring, documentation, and code organization

The original implementation explored various parallel strategies for K-Nearest Neighbors. This library packages that work into a reusable, production-ready tool with a user-friendly Python interface.

---

## Citation

If you use this library in your research, please cite:

```bibtex
@software{fastknn2025,
  title = {FastKNN: High-Performance K-Nearest Neighbors Library},
  author = {[Your Name]},
  year = {2025},
  note = {CMSE 822 Final Project, Michigan State University}
}
```

---

## Contributing

Contributions are welcome! Areas for improvement:

- Additional distance metrics (Manhattan, Cosine, etc.)
- Optimized CUDA kernels (shared memory, coalesced access)
- Support for more GPU architectures
- Additional example datasets
- Performance optimizations

Please open an issue or pull request on GitHub.

---

## License

MIT License - See LICENSE file for details.

**Original Project:** CMSE 822 Final Project
**Institution:** Michigan State University
**Course Instructor:** Dr. Sean Couch

---

**FastKNN** - Bringing GPU-accelerated machine learning to everyone! 🚀
