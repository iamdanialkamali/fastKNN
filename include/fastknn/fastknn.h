#ifndef FASTKNN_H
#define FASTKNN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* Version information */
#define FASTKNN_VERSION_MAJOR 1
#define FASTKNN_VERSION_MINOR 0
#define FASTKNN_VERSION_PATCH 0

/* Error codes */
typedef enum {
    FASTKNN_SUCCESS = 0,
    FASTKNN_ERROR_INVALID_ARGUMENT = -1,
    FASTKNN_ERROR_BACKEND_NOT_AVAILABLE = -2,
    FASTKNN_ERROR_ALLOCATION_FAILED = -3,
    FASTKNN_ERROR_CUDA_ERROR = -4,
    FASTKNN_ERROR_MPI_ERROR = -5,
    FASTKNN_ERROR_IO_ERROR = -6
} fastknn_error;

/* Backend selection */
typedef enum {
    FASTKNN_BACKEND_SERIAL = 0,
    FASTKNN_BACKEND_OPENMP,
    FASTKNN_BACKEND_MPI_TRAIN_DECOMP,
    FASTKNN_BACKEND_MPI_TEST_DECOMP,
    FASTKNN_BACKEND_CUDA,
    FASTKNN_BACKEND_CUDA_TRAIN,
    FASTKNN_BACKEND_CUDA_MPI,
    FASTKNN_BACKEND_CUDA_MPI_OPENMP
} fastknn_backend;

/* Row-major matrix of 32-bit floats */
typedef struct {
    float *data;      /* Row-major data (rows * cols elements) */
    int rows;         /* Number of rows */
    int cols;         /* Number of columns */
    int owns_data;    /* 1 if library should free data, 0 otherwise */
} fastknn_matrix_f32;

/* Integer label array */
typedef struct {
    int *data;        /* Array of integer labels */
    int count;        /* Number of labels */
    int owns_data;    /* 1 if library should free data, 0 otherwise */
} fastknn_labels_i32;

/* Configuration for KNN execution */
typedef struct {
    int k;                          /* Number of nearest neighbors */
    fastknn_backend backend;        /* Backend to use */
    int num_threads;                /* OpenMP threads (0 = auto) */
    int cuda_device;                /* CUDA device ID (-1 = auto) */
    int verbose;                    /* Logging verbosity (0 = quiet, 1 = normal, 2 = debug) */
} fastknn_config;

/* Result of KNN prediction */
typedef struct {
    fastknn_labels_i32 predictions;  /* Predicted labels */
    float *distances;                 /* Optional: top-k distances (can be NULL) */
    int *indices;                     /* Optional: top-k indices (can be NULL) */
    int owns_distances;               /* 1 if library should free distances */
    int owns_indices;                 /* 1 if library should free indices */
} fastknn_result;

/* Initialization and cleanup */
fastknn_error fastknn_init(void);
void fastknn_cleanup(void);

/* Configuration */
fastknn_config fastknn_config_default(void);
int fastknn_backend_available(fastknn_backend backend);
const char* fastknn_backend_name(fastknn_backend backend);

/* Matrix operations */
fastknn_matrix_f32 fastknn_matrix_wrap(float *data, int rows, int cols);
fastknn_matrix_f32 fastknn_matrix_alloc(int rows, int cols);
void fastknn_matrix_free(fastknn_matrix_f32 *matrix);
fastknn_error fastknn_matrix_read_csv(const char *filename,
                                       fastknn_matrix_f32 *matrix,
                                       fastknn_labels_i32 *labels,
                                       int max_rows);

/* Label operations */
fastknn_labels_i32 fastknn_labels_wrap(int *data, int count);
fastknn_labels_i32 fastknn_labels_alloc(int count);
void fastknn_labels_free(fastknn_labels_i32 *labels);

/* KNN prediction */
fastknn_error fastknn_predict(const fastknn_matrix_f32 *train_data,
                               const fastknn_labels_i32 *train_labels,
                               const fastknn_matrix_f32 *test_data,
                               const fastknn_config *config,
                               fastknn_result *result);

/* Result management */
void fastknn_result_free(fastknn_result *result);

/* Utility functions */
float fastknn_accuracy(const fastknn_labels_i32 *predictions,
                       const fastknn_labels_i32 *true_labels);
const char* fastknn_error_string(fastknn_error error);

#ifdef __cplusplus
}
#endif

#endif /* FASTKNN_H */
