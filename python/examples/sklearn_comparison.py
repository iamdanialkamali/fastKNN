#!/usr/bin/env python3
"""
Compare FastKNN with scikit-learn KNeighborsClassifier

This example demonstrates:
1. API compatibility with sklearn
2. Performance comparison
3. Accuracy verification
"""

import numpy as np
import time
from fastknn import KNeighborsClassifier, Backend

try:
    from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available, using synthetic data only")


def generate_data(n_samples=1000, n_features=20, n_classes=5):
    """Generate synthetic classification data"""
    if SKLEARN_AVAILABLE:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=10,
            n_redundant=5,
            random_state=42
        )
    else:
        # Simple synthetic data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, n_classes, size=n_samples).astype(np.int32)

    # Split into train/test
    n_train = int(0.8 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def benchmark_fastknn(X_train, y_train, X_test, backend=Backend.SERIAL):
    """Benchmark FastKNN with specified backend"""
    clf = KNeighborsClassifier(n_neighbors=5, backend=backend, verbose=0)

    # Fit
    start = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - start

    # Predict
    start = time.time()
    predictions = clf.predict(X_test)
    predict_time = time.time() - start

    return clf, predictions, fit_time, predict_time


def main():
    print("=" * 70)
    print("FastKNN vs Scikit-learn Comparison")
    print("=" * 70)

    # Generate data
    print("\nGenerating data...")
    X_train, X_test, y_train, y_test = generate_data(
        n_samples=5000, n_features=50, n_classes=10
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results = []

    # Test FastKNN backends
    for backend in [Backend.SERIAL, Backend.OPENMP, Backend.CUDA]:
        try:
            backend_name = {
                Backend.SERIAL: "Serial",
                Backend.OPENMP: "OpenMP",
                Backend.CUDA: "CUDA"
            }[backend]

            print(f"\n{'='*70}")
            print(f"FastKNN ({backend_name})")
            print(f"{'='*70}")

            clf, predictions, fit_time, predict_time = benchmark_fastknn(
                X_train, y_train, X_test, backend
            )

            accuracy = clf.score(X_test, y_test)

            print(f"Fit time:     {fit_time:.6f}s")
            print(f"Predict time: {predict_time:.6f}s")
            print(f"Total time:   {fit_time + predict_time:.6f}s")
            print(f"Accuracy:     {accuracy:.4f}")

            results.append({
                'name': f'FastKNN ({backend_name})',
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_time': fit_time + predict_time,
                'accuracy': accuracy
            })

        except RuntimeError as e:
            print(f"Skipped: {e}")

    # Test sklearn for comparison
    if SKLEARN_AVAILABLE:
        print(f"\n{'='*70}")
        print("Scikit-learn KNeighborsClassifier")
        print(f"{'='*70}")

        clf = SklearnKNN(n_neighbors=5, algorithm='brute', n_jobs=-1)

        start = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - start

        start = time.time()
        predictions = clf.predict(X_test)
        predict_time = time.time() - start

        accuracy = clf.score(X_test, y_test)

        print(f"Fit time:     {fit_time:.6f}s")
        print(f"Predict time: {predict_time:.6f}s")
        print(f"Total time:   {fit_time + predict_time:.6f}s")
        print(f"Accuracy:     {accuracy:.4f}")

        results.append({
            'name': 'Scikit-learn',
            'fit_time': fit_time,
            'predict_time': predict_time,
            'total_time': fit_time + predict_time,
            'accuracy': accuracy
        })

    # Summary
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("Summary (Predict Time)")
        print(f"{'='*70}")

        # Sort by predict time
        results_sorted = sorted(results, key=lambda x: x['predict_time'])
        baseline = results_sorted[-1]['predict_time']  # Slowest

        for r in results_sorted:
            speedup = baseline / r['predict_time']
            print(f"{r['name']:30s} {r['predict_time']:8.6f}s  {speedup:6.2f}x")

        print(f"\n{'='*70}")
        print("Accuracy Comparison")
        print(f"{'='*70}")
        for r in results:
            print(f"{r['name']:30s} {r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
