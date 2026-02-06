#!/usr/bin/env python3
"""
MNIST classification with FastKNN

Demonstrates sklearn-compatible API on real-world dataset.
"""

import numpy as np
import time
from pathlib import Path
from fastknn import KNeighborsClassifier, Backend


def load_mnist_csv(data_dir="../data", max_train=None, max_test=None):
    """Load MNIST from CSV files"""
    data_dir = Path(data_dir)

    # Load training data
    print(f"Loading training data from {data_dir / 'train_data.csv'}...")
    train_data = []
    with open(data_dir / "train_data.csv", "r") as f:
        for i, line in enumerate(f):
            if max_train and i >= max_train:
                break
            values = [float(x) for x in line.strip().split(',')]
            train_data.append(values)

    train_data = np.array(train_data, dtype=np.float32)
    X_train = train_data[:, :-1]  # Features
    y_train = train_data[:, -1].astype(np.int32)  # Labels

    # Load test data
    print(f"Loading test data from {data_dir / 'test_data.csv'}...")
    test_data = []
    with open(data_dir / "test_data.csv", "r") as f:
        for i, line in enumerate(f):
            if max_test and i >= max_test:
                break
            values = [float(x) for x in line.strip().split(',')]
            test_data.append(values)

    test_data = np.array(test_data, dtype=np.float32)
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].astype(np.int32)

    return X_train, X_test, y_train, y_test


def main():
    print("=" * 70)
    print("MNIST Classification with FastKNN")
    print("=" * 70)

    # Load data
    X_train, X_test, y_train, y_test = load_mnist_csv(
        max_train=10000,  # Use subset for faster demo
        max_test=1000
    )

    print(f"\nDataset:")
    print(f"  Training:   {X_train.shape} samples")
    print(f"  Test:       {X_test.shape} samples")
    print(f"  Features:   {X_train.shape[1]}")
    print(f"  Classes:    {len(np.unique(y_train))}")

    # Try different backends
    backends_to_test = [
        (Backend.SERIAL, "Serial"),
        (Backend.OPENMP, "OpenMP"),
        (Backend.CUDA, "CUDA"),
    ]

    for backend, name in backends_to_test:
        print(f"\n{'='*70}")
        print(f"{name} Backend")
        print(f"{'='*70}")

        try:
            # Create classifier
            clf = KNeighborsClassifier(
                n_neighbors=3,
                backend=backend,
                verbose=1
            )

            # Fit
            print("\nFitting...")
            start = time.time()
            clf.fit(X_train, y_train)
            fit_time = time.time() - start
            print(f"Fit time: {fit_time:.4f}s")

            # Predict
            print("\nPredicting...")
            start = time.time()
            predictions = clf.predict(X_test)
            predict_time = time.time() - start
            print(f"Predict time: {predict_time:.4f}s")

            # Score
            accuracy = clf.score(X_test, y_test)
            print(f"\nAccuracy: {accuracy * 100:.2f}%")

            # Show some predictions
            print("\nSample predictions:")
            for i in range(min(10, len(predictions))):
                print(f"  True: {y_test[i]}, Predicted: {predictions[i]}")

        except RuntimeError as e:
            print(f"Skipped: {e}")


if __name__ == "__main__":
    main()
