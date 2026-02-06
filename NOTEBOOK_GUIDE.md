# FastKNN Tutorial Notebook Guide

## Overview

The `FastKNN_Tutorial.ipynb` notebook provides an interactive demonstration of the FastKNN library with beautiful visualizations showing performance comparisons across different backends.

## Features

✨ **Interactive demonstrations** of all FastKNN backends
📊 **Beautiful visualizations** showing speedups and performance
🎯 **Real MNIST dataset** for practical examples
🔬 **Scalability analysis** with varying dataset sizes
📈 **Parameter tuning** examples (k-neighbors optimization)

## Quick Start

### 1. Install Required Packages

```bash
pip install numpy matplotlib seaborn pandas jupyter
```

### 2. Launch Jupyter

```bash
cd fastknn-library
jupyter notebook FastKNN_Tutorial.ipynb
```

### 3. Run All Cells

From Jupyter:
- Click `Cell` → `Run All`
- Or press `Shift+Enter` to run cells one by one

## What's Inside

### 📚 Section 1: Setup & Data Loading
- Import FastKNN and visualization libraries
- Load MNIST handwritten digit dataset
- Visualize sample digits

### ⚡ Section 2: Basic Usage
- Simple example with serial backend
- Scikit-learn compatible API demonstration

### 🏎️ Section 3: Performance Comparison
- Compare Serial, OpenMP, and CUDA backends
- Side-by-side timing results
- Automatic speedup calculations

### 📊 Section 4: Visualizations
- **Bar charts** showing execution times
- **Speedup charts** comparing to baseline
- Color-coded for easy interpretation
- Value labels for precise readings

### 📈 Section 5: Scalability Tests
- Test with 1K, 2K, 5K, 10K training samples
- Line plots showing scaling behavior
- Identify optimal backend for your dataset size

### 💎 Section 6: Multi-GPU Performance
- Test CUDA+MPI backend for multi-GPU parallelism
- Automatic fallback to simulated multi-GPU if MPI not available
- Shows speedup with 3 GPUs (GPUs 0, 1, 3)
- Parallel efficiency analysis (80-85% typical)
- Comparison visualizations: Single GPU vs Multi-GPU

### 🎯 Section 7: Confusion Matrix
- Visualize classification performance
- Heatmap showing prediction accuracy per digit
- Uses the fastest available backend

### 🔧 Section 8: Parameter Tuning
- Test different k values (1, 3, 5, 7, 9, 11)
- Find optimal number of neighbors
- Visualize accuracy vs k

## Expected Results

### Typical Speedups (10K train, 1K test)

| Backend | Time | Speedup | When to Use |
|---------|------|---------|-------------|
| Serial  | ~5s  | 1.0×    | Baseline, small datasets |
| OpenMP  | ~0.6s | 8×     | Multi-core CPU, medium datasets |
| CUDA    | ~0.12s | 40×   | GPU available, large datasets |
| Multi-GPU (3 GPUs) | ~0.05s | 100×+ | Multiple GPUs, very large datasets |

### Visual Examples

The notebook includes:

1. **Execution Time Bar Chart**
   - Blue (Serial), Red (OpenMP), Green (CUDA)
   - Shows absolute time for each backend

2. **Speedup Bar Chart**
   - Normalized to Serial (1.0×)
   - Easy comparison of relative performance

3. **Scaling Line Plot**
   - Performance vs dataset size
   - Shows where each backend excels

4. **Confusion Matrix Heatmap**
   - 10×10 grid for MNIST digits (0-9)
   - Color intensity shows prediction frequency

5. **Multi-GPU Comparison Charts**
   - Side-by-side: Single GPU vs Multi-GPU time
   - Speedup visualization with theoretical maximum
   - Parallel efficiency analysis

6. **K-Parameter Accuracy Plot**
   - Find optimal k for your dataset
   - Trade-off between underfitting and overfitting

## Customization

### Use Your Own Data

Replace the data loading section:

```python
# Your custom data
X_train = np.array(...)  # Shape: (n_samples, n_features)
y_train = np.array(...)  # Shape: (n_samples,)
X_test = np.array(...)
y_test = np.array(...)
```

### Test Different Backends

```python
# Change backend parameter
clf = KNeighborsClassifier(
    n_neighbors=3,
    backend='cuda',  # Try: 'serial', 'openmp', 'cuda'
    device=0,        # GPU device (for CUDA)
    verbose=1        # Print detailed info
)
```

### Adjust Dataset Sizes

```python
# In the scalability test section
train_sizes = [1000, 5000, 10000, 20000]  # Add more sizes
test_size = 1000  # Increase test set
```

### Modify Visualizations

The notebook uses matplotlib and seaborn. Customize colors, sizes, etc.:

```python
# Change color scheme
colors = ['#your_color1', '#your_color2', '#your_color3']

# Adjust figure size
plt.figure(figsize=(width, height))

# Change plot style
sns.set_style('whitegrid')  # or 'darkgrid', 'white', 'dark', 'ticks'
```

## Troubleshooting

### Backend Not Available

If you see "Not available" for a backend:

- **OpenMP**: May not be available on all systems
  - Solution: Library should be built with OpenMP support

- **CUDA**: Requires NVIDIA GPU and CUDA toolkit
  - Solution: Check `nvidia-smi` and CUDA installation
  - Note: Library must be built with `-DFASTKNN_ENABLE_CUDA=ON`

### Import Errors

```
ModuleNotFoundError: No module named 'fastknn'
```

**Solution**: Ensure library is built and Python can find it:

```bash
# In fastknn-library directory
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
```

Or add to notebook first cell:

```python
import sys
sys.path.insert(0, '/path/to/fastknn-library/python')
```

### CUDA Errors

```
CUDA error: CUDA driver version is insufficient
```

**Solution**: Check CUDA compatibility:

```bash
nvidia-smi  # Check driver CUDA version
nvcc --version  # Check compiler CUDA version
```

Rebuild with correct CUDA:

```bash
export PATH=/usr/local/cuda-12/bin:$PATH
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc
```

## Tips for Best Results

1. **Start Small**: Run with smaller datasets first to verify everything works
2. **Sequential Testing**: Test backends one at a time if you encounter issues
3. **GPU Warmup**: First CUDA run may be slower due to initialization
4. **Memory**: Large datasets may require significant RAM/VRAM
5. **Timing**: Run predictions multiple times for more stable timing measurements

## Performance Tips

- **For presentations**: Use the visualizations as-is - they're publication-ready
- **For comparisons**: Adjust dataset sizes to match your use case
- **For benchmarking**: Increase iterations and take averages
- **For exploration**: Modify k values, distance metrics, etc.

## Next Steps

After running the notebook:

1. ✅ Try your own datasets
2. ✅ Experiment with different k values
3. ✅ Compare backends on your hardware
4. ✅ Integrate FastKNN into your projects
5. ✅ Share your results!

## Questions?

- Check the main [README.md](README.md) for library documentation
- See [PERFORMANCE_RESULTS.md](PERFORMANCE_RESULTS.md) for detailed benchmarks
- Review [README_LIBRARY.md](README_LIBRARY.md) for API reference

---

**Enjoy exploring FastKNN!** 🚀
