"""Backend enumeration for FastKNN"""

from enum import IntEnum


class Backend(IntEnum):
    """Available parallel backends for KNN computation.

    Attributes:
        SERIAL: Single-threaded CPU implementation
        OPENMP: Multi-threaded CPU (shared memory)
        MPI_TRAIN_DECOMP: MPI with training data decomposition
        MPI_TEST_DECOMP: MPI with test data decomposition
        CUDA: Single GPU
        CUDA_TRAIN: Single GPU (train-optimized)
        CUDA_MPI: Multi-GPU via MPI
        CUDA_MPI_OPENMP: Multi-GPU + MPI + OpenMP (all three)
    """
    SERIAL = 0
    OPENMP = 1
    MPI_TRAIN_DECOMP = 2
    MPI_TEST_DECOMP = 3
    CUDA = 4
    CUDA_TRAIN = 5
    CUDA_MPI = 6
    CUDA_MPI_OPENMP = 7
