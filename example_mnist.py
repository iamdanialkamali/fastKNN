#!/usr/bin/env python3
"""
FastKNN MNIST Classification Example
=====================================

This example demonstrates how to use FastKNN for handwritten digit
classification on the MNIST dataset.

Compares different backends (Serial, OpenMP, CUDA) and shows sklearn compatibility.

Usage:
    python example_mnist.py
    python example_mnist.py --backend cuda
    python example_mnist.py --train 10000 --test 1000
    mpirun -np 4 python example_mnist.py --backend cuda_mpi
"""

import argparse
import time
import numpy as np

# Import FastKNN
try:
    from fastknn import KNeighborsClassifier, Backend
    HAS_FASTKNN = True
except ImportError:
    print("Error: FastKNN not installed. Run: cd python && pip install -e .")
    HAS_FASTKNN = False
    exit(1)

# Optional: compare with sklearn
try:
    from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_mnist_csv(train_path='../data/train_data.csv',
                    test_path='../data/test_data.csv',
                    train_limit=None,
                    test_limit=None):
    """
    Load MNIST data from CSV files.

    CSV format: Each row is [pixel1, pixel2, ..., pixel784, label]

    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        train_limit: Max training samples (None = all)
        test_limit: Max test samples (None = all)

    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    print("Loading MNIST data...")

    # Load training data
    print(f"  Reading {train_path}...")
    train_data = []
    with open(train_path, 'r') as f:
        for i, line in enumerate(f):
            if train_limit and i >= train_limit:
                break
            values = [float(x) for x in line.strip().split(',')]
            train_data.append(values)

    train_data = np.array(train_data, dtype=np.float32)
    X_train = train_data[:, :-1]  # All columns except last
    y_train = train_data[:, -1].astype(np.int32)  # Last column

    # Load test data
    print(f"  Reading {test_path}...")
    test_data = []
    with open(test_path, 'r') as f:
        for i, line in enumerate(f):
            if test_limit and i >= test_limit:
                break
            values = [float(x) for x in line.strip().split(',')]
            test_data.append(values)

    test_data = np.array(test_data, dtype=np.float32)
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].astype(np.int32)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")

    return X_train, y_train, X_test, y_test


def run_fastknn(X_train, y_train, X_test, y_test, backend='serial', k=3, verbose=1):
    """
    Run FastKNN with specified backend.

    Args:
        X_train, y_train, X_test, y_test: Data arrays
        backend: Backend name ('serial', 'openmp', 'cuda', 'cuda_mpi')
        k: Number of neighbors
        verbose: Verbosity level

    Returns:
        dict with keys: accuracy, train_time, predict_time, total_time
    """
    print(f"\n{'='*60}")
    print(f"FastKNN with {backend.upper()} backend (k={k})")
    print(f"{'='*60}")

    # Create classifier
    clf = KNeighborsClassifier(
        n_neighbors=k,
        backend=backend,
        verbose=verbose
    )

    # Fit
    print("Fitting model...")
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Completed in {train_time:.4f}s")

    # Predict
    print("Making predictions...")
    start = time.time()
    predictions = clf.predict(X_test)
    predict_time = time.time() - start
    print(f"  Completed in {predict_time:.4f}s")

    # Score
    accuracy = clf.score(X_test, y_test)
    total_time = train_time + predict_time

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Train time: {train_time:.4f}s")
    print(f"  Predict time: {predict_time:.4f}s")
    print(f"  Total time: {total_time:.4f}s")

    return {
        'accuracy': accuracy,
        'train_time': train_time,
        'predict_time': predict_time,
        'total_time': total_time,
        'backend': backend
    }


def run_sklearn(X_train, y_train, X_test, y_test, k=3, n_jobs=-1):
    """
    Run scikit-learn KNN for comparison.
    """
    if not HAS_SKLEARN:
        print("\nSkipping sklearn comparison (not installed)")
        return None

    print(f"\n{'='*60}")
    print(f"Scikit-learn KNN (k={k}, n_jobs={n_jobs})")
    print(f"{'='*60}")

    clf = SklearnKNN(n_neighbors=k, n_jobs=n_jobs)

    print("Fitting model...")
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Completed in {train_time:.4f}s")

    print("Making predictions...")
    start = time.time()
    predictions = clf.predict(X_test)
    predict_time = time.time() - start
    print(f"  Completed in {predict_time:.4f}s")

    accuracy = clf.score(X_test, y_test)
    total_time = train_time + predict_time

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Total time: {total_time:.4f}s")

    return {
        'accuracy': accuracy,
        'total_time': total_time,
        'backend': 'sklearn'
    }


def compare_backends(X_train, y_train, X_test, y_test, k=3):
    """
    Compare multiple FastKNN backends.
    """
    backends_to_test = ['serial', 'openmp', 'cuda']

    results = []

    for backend in backends_to_test:
        try:
            result = run_fastknn(X_train, y_train, X_test, y_test,
                                 backend=backend, k=k, verbose=0)
            results.append(result)
        except Exception as e:
            print(f"\n{backend.upper()} backend not available: {e}")
            continue

    # Print comparison table
    if results:
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"{'Backend':<15} {'Time (s)':>10} {'Speedup':>10} {'Accuracy':>10}")
        print('-' * 60)

        baseline_time = results[0]['total_time']
        for r in results:
            speedup = baseline_time / r['total_time']
            print(f"{r['backend']:<15} {r['total_time']:>10.4f} "
                  f"{speedup:>10.2f}x {r['accuracy']*100:>9.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='FastKNN MNIST Example')
    parser.add_argument('--train', type=int, default=10000,
                        help='Number of training samples (default: 10000)')
    parser.add_argument('--test', type=int, default=1000,
                        help='Number of test samples (default: 1000)')
    parser.add_argument('--k', type=int, default=3,
                        help='Number of neighbors (default: 3)')
    parser.add_argument('--backend', type=str, default='compare',
                        choices=['serial', 'openmp', 'cuda', 'cuda_mpi', 'compare'],
                        help='Backend to use (default: compare all)')
    parser.add_argument('--compare-sklearn', action='store_true',
                        help='Also run sklearn for comparison')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level (0=quiet, 1=normal, 2=debug)')

    args = parser.parse_args()

    # Load data
    X_train, y_train, X_test, y_test = load_mnist_csv(
        train_limit=args.train,
        test_limit=args.test
    )

    print(f"\nDataset: {X_train.shape[0]} train, {X_test.shape[0]} test, "
          f"{X_train.shape[1]} features")

    # Run experiments
    if args.backend == 'compare':
        results = compare_backends(X_train, y_train, X_test, y_test, k=args.k)
    else:
        result = run_fastknn(X_train, y_train, X_test, y_test,
                             backend=args.backend, k=args.k, verbose=args.verbose)

    # Optional sklearn comparison
    if args.compare_sklearn:
        sklearn_result = run_sklearn(X_train, y_train, X_test, y_test, k=args.k)

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
