#include "fastknn/fastknn.h"

/* Provide stubs for all CUDA-related functions when CUDA is not available */

#ifndef FASTKNN_ENABLE_CUDA

fastknn_error fastknn_predict_cuda(const fastknn_matrix_f32 *train_data,
                                    const fastknn_labels_i32 *train_labels,
                                    const fastknn_matrix_f32 *test_data,
                                    const fastknn_config *config,
                                    fastknn_result *result) {
    (void)train_data; (void)train_labels; (void)test_data; (void)config; (void)result;
    return FASTKNN_ERROR_BACKEND_NOT_AVAILABLE;
}

fastknn_error fastknn_predict_cuda_train(const fastknn_matrix_f32 *train_data,
                                          const fastknn_labels_i32 *train_labels,
                                          const fastknn_matrix_f32 *test_data,
                                          const fastknn_config *config,
                                          fastknn_result *result) {
    (void)train_data; (void)train_labels; (void)test_data; (void)config; (void)result;
    return FASTKNN_ERROR_BACKEND_NOT_AVAILABLE;
}

fastknn_error fastknn_predict_cuda_mpi(const fastknn_matrix_f32 *train_data,
                                        const fastknn_labels_i32 *train_labels,
                                        const fastknn_matrix_f32 *test_data,
                                        const fastknn_config *config,
                                        fastknn_result *result) {
    (void)train_data; (void)train_labels; (void)test_data; (void)config; (void)result;
    return FASTKNN_ERROR_BACKEND_NOT_AVAILABLE;
}

fastknn_error fastknn_predict_cuda_mpi_openmp(const fastknn_matrix_f32 *train_data,
                                                const fastknn_labels_i32 *train_labels,
                                                const fastknn_matrix_f32 *test_data,
                                                const fastknn_config *config,
                                                fastknn_result *result) {
    (void)train_data; (void)train_labels; (void)test_data; (void)config; (void)result;
    return FASTKNN_ERROR_BACKEND_NOT_AVAILABLE;
}

#endif /* !FASTKNN_ENABLE_CUDA */
