#include "fastknn/fastknn.h"
#include <stdio.h>

/* Forward declarations for backend implementations */
fastknn_error fastknn_predict_serial(const fastknn_matrix_f32 *train_data,
                                      const fastknn_labels_i32 *train_labels,
                                      const fastknn_matrix_f32 *test_data,
                                      const fastknn_config *config,
                                      fastknn_result *result);

fastknn_error fastknn_predict_openmp(const fastknn_matrix_f32 *train_data,
                                      const fastknn_labels_i32 *train_labels,
                                      const fastknn_matrix_f32 *test_data,
                                      const fastknn_config *config,
                                      fastknn_result *result);

fastknn_error fastknn_predict_mpi_train(const fastknn_matrix_f32 *train_data,
                                         const fastknn_labels_i32 *train_labels,
                                         const fastknn_matrix_f32 *test_data,
                                         const fastknn_config *config,
                                         fastknn_result *result);

fastknn_error fastknn_predict_mpi_test(const fastknn_matrix_f32 *train_data,
                                        const fastknn_labels_i32 *train_labels,
                                        const fastknn_matrix_f32 *test_data,
                                        const fastknn_config *config,
                                        fastknn_result *result);

fastknn_error fastknn_predict_cuda(const fastknn_matrix_f32 *train_data,
                                    const fastknn_labels_i32 *train_labels,
                                    const fastknn_matrix_f32 *test_data,
                                    const fastknn_config *config,
                                    fastknn_result *result);

fastknn_error fastknn_predict_cuda_train(const fastknn_matrix_f32 *train_data,
                                          const fastknn_labels_i32 *train_labels,
                                          const fastknn_matrix_f32 *test_data,
                                          const fastknn_config *config,
                                          fastknn_result *result);

fastknn_error fastknn_predict_cuda_mpi(const fastknn_matrix_f32 *train_data,
                                        const fastknn_labels_i32 *train_labels,
                                        const fastknn_matrix_f32 *test_data,
                                        const fastknn_config *config,
                                        fastknn_result *result);

fastknn_error fastknn_predict_cuda_mpi_openmp(const fastknn_matrix_f32 *train_data,
                                                const fastknn_labels_i32 *train_labels,
                                                const fastknn_matrix_f32 *test_data,
                                                const fastknn_config *config,
                                                fastknn_result *result);

fastknn_error fastknn_predict(const fastknn_matrix_f32 *train_data,
                               const fastknn_labels_i32 *train_labels,
                               const fastknn_matrix_f32 *test_data,
                               const fastknn_config *config,
                               fastknn_result *result) {
    if (!train_data || !train_labels || !test_data || !config || !result) {
        return FASTKNN_ERROR_INVALID_ARGUMENT;
    }

    /* Check if backend is available */
    if (!fastknn_backend_available(config->backend)) {
        if (config->verbose > 0) {
            fprintf(stderr, "Backend '%s' is not available\n",
                    fastknn_backend_name(config->backend));
        }
        return FASTKNN_ERROR_BACKEND_NOT_AVAILABLE;
    }

    if (config->verbose > 0) {
        printf("Using backend: %s\n", fastknn_backend_name(config->backend));
        printf("Train size: %d, Test size: %d, k=%d, features=%d\n",
               train_data->rows, test_data->rows, config->k, train_data->cols);
    }

    /* Dispatch to appropriate backend */
    switch (config->backend) {
        case FASTKNN_BACKEND_SERIAL:
            return fastknn_predict_serial(train_data, train_labels, test_data, config, result);

        case FASTKNN_BACKEND_OPENMP:
            return fastknn_predict_openmp(train_data, train_labels, test_data, config, result);

        case FASTKNN_BACKEND_MPI_TRAIN_DECOMP:
            return fastknn_predict_mpi_train(train_data, train_labels, test_data, config, result);

        case FASTKNN_BACKEND_MPI_TEST_DECOMP:
            return fastknn_predict_mpi_test(train_data, train_labels, test_data, config, result);

        case FASTKNN_BACKEND_CUDA:
            return fastknn_predict_cuda(train_data, train_labels, test_data, config, result);

        case FASTKNN_BACKEND_CUDA_TRAIN:
            return fastknn_predict_cuda_train(train_data, train_labels, test_data, config, result);

        case FASTKNN_BACKEND_CUDA_MPI:
            return fastknn_predict_cuda_mpi(train_data, train_labels, test_data, config, result);

        case FASTKNN_BACKEND_CUDA_MPI_OPENMP:
            return fastknn_predict_cuda_mpi_openmp(train_data, train_labels, test_data, config, result);

        default:
            return FASTKNN_ERROR_BACKEND_NOT_AVAILABLE;
    }
}
