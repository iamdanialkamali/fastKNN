"""Scikit-learn compatible KNN classifier using FastKNN library"""

import os
import ctypes
import numpy as np
from pathlib import Path
from .backends import Backend


# Locate the shared library
def _find_library():
    """Find the FastKNN shared library"""
    # Try common locations
    lib_name = "libfastknn.so"  # Linux
    if os.name == "nt":
        lib_name = "fastknn.dll"  # Windows
    elif os.name == "darwin":
        lib_name = "libfastknn.dylib"  # macOS

    # Check build directory
    search_paths = [
        Path(__file__).parent.parent.parent / "build" / lib_name,
        Path(__file__).parent.parent.parent / "build" / "libfastknn.a",
        Path("/usr/local/lib") / lib_name,
        Path("/usr/lib") / lib_name,
    ]

    for path in search_paths:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"FastKNN library not found. Please build the library first:\n"
        f"  mkdir build && cd build && cmake .. && make"
    )


# Load the library
_lib_path = _find_library()
_lib = ctypes.CDLL(_lib_path)


# Define C structures
class CMatrix(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("rows", ctypes.c_int),
        ("cols", ctypes.c_int),
        ("owns_data", ctypes.c_int),
    ]


class CLabels(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_int)),
        ("count", ctypes.c_int),
        ("owns_data", ctypes.c_int),
    ]


class CConfig(ctypes.Structure):
    _fields_ = [
        ("k", ctypes.c_int),
        ("backend", ctypes.c_int),
        ("num_threads", ctypes.c_int),
        ("cuda_device", ctypes.c_int),
        ("verbose", ctypes.c_int),
    ]


class CResult(ctypes.Structure):
    _fields_ = [
        ("predictions", CLabels),
        ("distances", ctypes.POINTER(ctypes.c_float)),
        ("indices", ctypes.POINTER(ctypes.c_int)),
        ("owns_distances", ctypes.c_int),
        ("owns_indices", ctypes.c_int),
    ]


# Define C functions
_lib.fastknn_init.argtypes = []
_lib.fastknn_init.restype = ctypes.c_int

_lib.fastknn_cleanup.argtypes = []
_lib.fastknn_cleanup.restype = None

_lib.fastknn_backend_available.argtypes = [ctypes.c_int]
_lib.fastknn_backend_available.restype = ctypes.c_int

_lib.fastknn_backend_name.argtypes = [ctypes.c_int]
_lib.fastknn_backend_name.restype = ctypes.c_char_p

_lib.fastknn_matrix_alloc.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.fastknn_matrix_alloc.restype = CMatrix

_lib.fastknn_matrix_free.argtypes = [ctypes.POINTER(CMatrix)]
_lib.fastknn_matrix_free.restype = None

_lib.fastknn_labels_alloc.argtypes = [ctypes.c_int]
_lib.fastknn_labels_alloc.restype = CLabels

_lib.fastknn_labels_free.argtypes = [ctypes.POINTER(CLabels)]
_lib.fastknn_labels_free.restype = None

_lib.fastknn_predict.argtypes = [
    ctypes.POINTER(CMatrix),
    ctypes.POINTER(CLabels),
    ctypes.POINTER(CMatrix),
    ctypes.POINTER(CConfig),
    ctypes.POINTER(CResult),
]
_lib.fastknn_predict.restype = ctypes.c_int

_lib.fastknn_result_free.argtypes = [ctypes.POINTER(CResult)]
_lib.fastknn_result_free.restype = None

_lib.fastknn_accuracy.argtypes = [ctypes.POINTER(CLabels), ctypes.POINTER(CLabels)]
_lib.fastknn_accuracy.restype = ctypes.c_float

_lib.fastknn_config_default.argtypes = []
_lib.fastknn_config_default.restype = CConfig


class KNeighborsClassifier:
    """K-Nearest Neighbors classifier with multiple parallel backends.

    Scikit-learn compatible interface with high-performance backends including
    OpenMP, MPI, CUDA, and hybrid multi-GPU implementations.

    Parameters
    ----------
    n_neighbors : int, default=3
        Number of neighbors to use for classification.

    backend : Backend or str, default=Backend.SERIAL
        Parallel backend to use. Options:
        - Backend.SERIAL or 'serial': Single-threaded CPU
        - Backend.OPENMP or 'openmp': Multi-threaded CPU
        - Backend.MPI_TRAIN_DECOMP or 'mpi_train': MPI train decomposition
        - Backend.MPI_TEST_DECOMP or 'mpi_test': MPI test decomposition
        - Backend.CUDA or 'cuda': Single GPU
        - Backend.CUDA_TRAIN or 'cuda_train': GPU train-optimized
        - Backend.CUDA_MPI or 'cuda_mpi': Multi-GPU via MPI
        - Backend.CUDA_MPI_OPENMP or 'cuda_mpi_openmp': All three combined

    n_jobs : int, default=None
        Number of threads for OpenMP backend (None = auto-detect).

    device : int, default=None
        CUDA device ID (None = auto-select).

    verbose : int, default=0
        Verbosity level (0=quiet, 1=normal, 2=debug).

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.

    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from fastknn import KNeighborsClassifier, Backend
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = KNeighborsClassifier(n_neighbors=2, backend=Backend.SERIAL)
    >>> clf.fit(X, y)
    >>> clf.predict([[1.5, 1.5]])
    array([0])
    >>> clf.score(X, y)
    1.0

    Multi-GPU example:
    >>> clf = KNeighborsClassifier(n_neighbors=3, backend=Backend.CUDA_MPI)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(self, n_neighbors=3, backend=Backend.SERIAL, n_jobs=None,
                 device=None, verbose=0):
        self.n_neighbors = n_neighbors

        # Handle string backend names
        if isinstance(backend, str):
            backend_map = {
                'serial': Backend.SERIAL,
                'openmp': Backend.OPENMP,
                'mpi_train': Backend.MPI_TRAIN_DECOMP,
                'mpi_test': Backend.MPI_TEST_DECOMP,
                'cuda': Backend.CUDA,
                'cuda_train': Backend.CUDA_TRAIN,
                'cuda_mpi': Backend.CUDA_MPI,
                'cuda_mpi_openmp': Backend.CUDA_MPI_OPENMP,
            }
            backend = backend_map.get(backend.lower(), Backend.SERIAL)

        self.backend = backend
        self.n_jobs = n_jobs
        self.device = device
        self.verbose = verbose

        self._X_train = None
        self._y_train = None
        self.classes_ = None
        self.n_features_in_ = None

        # Initialize library
        _lib.fastknn_init()

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, '_X_train') and self._X_train is not None:
            # Cleanup handled by numpy array lifecycle
            pass

    def fit(self, X, y):
        """Fit the KNN classifier from the training dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")

        if len(y) != len(X):
            raise ValueError("X and y must have the same length")

        # Check backend availability
        if not _lib.fastknn_backend_available(self.backend):
            backend_name = _lib.fastknn_backend_name(self.backend).decode('utf-8')
            raise RuntimeError(
                f"Backend '{backend_name}' is not available. "
                f"Please rebuild the library with appropriate flags."
            )

        # Store training data
        self._X_train = np.ascontiguousarray(X, dtype=np.float32)
        self._y_train = np.ascontiguousarray(y, dtype=np.int32)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Class labels for each data sample.
        """
        if self._X_train is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but classifier expects {self.n_features_in_}"
            )

        X = np.ascontiguousarray(X, dtype=np.float32)

        # Create C structures
        train_matrix = CMatrix()
        train_matrix.data = self._X_train.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        train_matrix.rows = self._X_train.shape[0]
        train_matrix.cols = self._X_train.shape[1]
        train_matrix.owns_data = 0

        train_labels = CLabels()
        train_labels.data = self._y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        train_labels.count = len(self._y_train)
        train_labels.owns_data = 0

        test_matrix = CMatrix()
        test_matrix.data = X.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        test_matrix.rows = X.shape[0]
        test_matrix.cols = X.shape[1]
        test_matrix.owns_data = 0

        config = CConfig()
        config.k = self.n_neighbors
        config.backend = self.backend
        config.num_threads = self.n_jobs if self.n_jobs is not None else 0
        config.cuda_device = self.device if self.device is not None else -1
        config.verbose = self.verbose

        result = CResult()

        # Call C library
        error = _lib.fastknn_predict(
            ctypes.byref(train_matrix),
            ctypes.byref(train_labels),
            ctypes.byref(test_matrix),
            ctypes.byref(config),
            ctypes.byref(result)
        )

        if error != 0:
            raise RuntimeError(f"Prediction failed with error code {error}")

        # Copy predictions to numpy array
        predictions = np.zeros(X.shape[0], dtype=np.int32)
        if result.predictions.count > 0:
            ctypes.memmove(
                predictions.ctypes.data,
                result.predictions.data,
                predictions.nbytes
            )

        # Free result
        _lib.fastknn_result_free(ctypes.byref(result))

        return predictions

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y (in [0, 1]).
        """
        y = np.asarray(y, dtype=np.int32)
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_backend_name(self):
        """Get the name of the current backend.

        Returns
        -------
        name : str
            Human-readable backend name.
        """
        return _lib.fastknn_backend_name(self.backend).decode('utf-8')
