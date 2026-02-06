#include "fastknn/fastknn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Global initialization state */
static int g_initialized = 0;

fastknn_error fastknn_init(void) {
    if (g_initialized) {
        return FASTKNN_SUCCESS;
    }
    g_initialized = 1;
    return FASTKNN_SUCCESS;
}

void fastknn_cleanup(void) {
    g_initialized = 0;
}

fastknn_config fastknn_config_default(void) {
    fastknn_config config;
    config.k = 3;
    config.backend = FASTKNN_BACKEND_SERIAL;
    config.num_threads = 0;  /* Auto-detect */
    config.cuda_device = -1; /* Auto-detect */
    config.verbose = 0;      /* Quiet */
    return config;
}

int fastknn_backend_available(fastknn_backend backend) {
    switch (backend) {
        case FASTKNN_BACKEND_SERIAL:
            return 1;  /* Always available */
#ifdef FASTKNN_ENABLE_OPENMP
        case FASTKNN_BACKEND_OPENMP:
            return 1;
#endif
#ifdef FASTKNN_ENABLE_MPI
        case FASTKNN_BACKEND_MPI_TRAIN_DECOMP:
        case FASTKNN_BACKEND_MPI_TEST_DECOMP:
            return 1;
#endif
#ifdef FASTKNN_ENABLE_CUDA
        case FASTKNN_BACKEND_CUDA:
        case FASTKNN_BACKEND_CUDA_TRAIN:
            return 1;
#endif
#if defined(FASTKNN_ENABLE_CUDA) && defined(FASTKNN_ENABLE_MPI)
        case FASTKNN_BACKEND_CUDA_MPI:
            return 1;
#endif
#if defined(FASTKNN_ENABLE_CUDA) && defined(FASTKNN_ENABLE_MPI) && defined(FASTKNN_ENABLE_OPENMP)
        case FASTKNN_BACKEND_CUDA_MPI_OPENMP:
            return 1;
#endif
        default:
            return 0;
    }
}

const char* fastknn_backend_name(fastknn_backend backend) {
    switch (backend) {
        case FASTKNN_BACKEND_SERIAL: return "Serial";
        case FASTKNN_BACKEND_OPENMP: return "OpenMP";
        case FASTKNN_BACKEND_MPI_TRAIN_DECOMP: return "MPI (Train Decomposition)";
        case FASTKNN_BACKEND_MPI_TEST_DECOMP: return "MPI (Test Decomposition)";
        case FASTKNN_BACKEND_CUDA: return "CUDA";
        case FASTKNN_BACKEND_CUDA_TRAIN: return "CUDA (Train Optimized)";
        case FASTKNN_BACKEND_CUDA_MPI: return "CUDA + MPI";
        case FASTKNN_BACKEND_CUDA_MPI_OPENMP: return "CUDA + MPI + OpenMP";
        default: return "Unknown";
    }
}

const char* fastknn_error_string(fastknn_error error) {
    switch (error) {
        case FASTKNN_SUCCESS: return "Success";
        case FASTKNN_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case FASTKNN_ERROR_BACKEND_NOT_AVAILABLE: return "Backend not available";
        case FASTKNN_ERROR_ALLOCATION_FAILED: return "Memory allocation failed";
        case FASTKNN_ERROR_CUDA_ERROR: return "CUDA error";
        case FASTKNN_ERROR_MPI_ERROR: return "MPI error";
        case FASTKNN_ERROR_IO_ERROR: return "I/O error";
        default: return "Unknown error";
    }
}

fastknn_matrix_f32 fastknn_matrix_wrap(float *data, int rows, int cols) {
    fastknn_matrix_f32 matrix;
    matrix.data = data;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.owns_data = 0;
    return matrix;
}

fastknn_matrix_f32 fastknn_matrix_alloc(int rows, int cols) {
    fastknn_matrix_f32 matrix;
    matrix.data = (float*)malloc(rows * cols * sizeof(float));
    matrix.rows = (matrix.data != NULL) ? rows : 0;
    matrix.cols = (matrix.data != NULL) ? cols : 0;
    matrix.owns_data = 1;
    return matrix;
}

void fastknn_matrix_free(fastknn_matrix_f32 *matrix) {
    if (matrix && matrix->owns_data && matrix->data) {
        free(matrix->data);
        matrix->data = NULL;
    }
    if (matrix) {
        matrix->rows = 0;
        matrix->cols = 0;
        matrix->owns_data = 0;
    }
}

fastknn_labels_i32 fastknn_labels_wrap(int *data, int count) {
    fastknn_labels_i32 labels;
    labels.data = data;
    labels.count = count;
    labels.owns_data = 0;
    return labels;
}

fastknn_labels_i32 fastknn_labels_alloc(int count) {
    fastknn_labels_i32 labels;
    labels.data = (int*)malloc(count * sizeof(int));
    labels.count = (labels.data != NULL) ? count : 0;
    labels.owns_data = 1;
    return labels;
}

void fastknn_labels_free(fastknn_labels_i32 *labels) {
    if (labels && labels->owns_data && labels->data) {
        free(labels->data);
        labels->data = NULL;
    }
    if (labels) {
        labels->count = 0;
        labels->owns_data = 0;
    }
}

void fastknn_result_free(fastknn_result *result) {
    if (!result) return;

    fastknn_labels_free(&result->predictions);

    if (result->owns_distances && result->distances) {
        free(result->distances);
        result->distances = NULL;
    }

    if (result->owns_indices && result->indices) {
        free(result->indices);
        result->indices = NULL;
    }

    result->owns_distances = 0;
    result->owns_indices = 0;
}

float fastknn_accuracy(const fastknn_labels_i32 *predictions,
                       const fastknn_labels_i32 *true_labels) {
    if (!predictions || !true_labels ||
        !predictions->data || !true_labels->data ||
        predictions->count != true_labels->count) {
        return 0.0f;
    }

    int correct = 0;
    for (int i = 0; i < predictions->count; i++) {
        if (predictions->data[i] == true_labels->data[i]) {
            correct++;
        }
    }

    return (100.0f * correct) / predictions->count;
}
