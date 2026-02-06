#include "fastknn/fastknn.h"

#ifdef FASTKNN_ENABLE_CUDA

/* Forward declaration */
extern "C" fastknn_error fastknn_predict_cuda(const fastknn_matrix_f32 *train_data,
                                               const fastknn_labels_i32 *train_labels,
                                               const fastknn_matrix_f32 *test_data,
                                               const fastknn_config *config,
                                               fastknn_result *result);

/* CUDA train-optimized variant (stub - delegates to main CUDA for now) */
extern "C" fastknn_error fastknn_predict_cuda_train(const fastknn_matrix_f32 *train_data,
                                                     const fastknn_labels_i32 *train_labels,
                                                     const fastknn_matrix_f32 *test_data,
                                                     const fastknn_config *config,
                                                     fastknn_result *result) {
    /* TODO: Implement train-optimized version */
    return fastknn_predict_cuda(train_data, train_labels, test_data, config, result);
}

#else

extern "C" fastknn_error fastknn_predict_cuda_train(const fastknn_matrix_f32 *train_data,
                                                     const fastknn_labels_i32 *train_labels,
                                                     const fastknn_matrix_f32 *test_data,
                                                     const fastknn_config *config,
                                                     fastknn_result *result) {
    (void)train_data; (void)train_labels; (void)test_data; (void)config; (void)result;
    return FASTKNN_ERROR_BACKEND_NOT_AVAILABLE;
}

#endif

/* CUDA+MPI hybrid stubs */
#if defined(FASTKNN_ENABLE_CUDA) && defined(FASTKNN_ENABLE_MPI)

extern "C" fastknn_error fastknn_predict_cuda_mpi(const fastknn_matrix_f32 *train_data,
                                                   const fastknn_labels_i32 *train_labels,
                                                   const fastknn_matrix_f32 *test_data,
                                                   const fastknn_config *config,
                                                   fastknn_result *result) {
    /* TODO: Implement CUDA+MPI multi-GPU version */
    return fastknn_predict_cuda(train_data, train_labels, test_data, config, result);
}

#else

extern "C" fastknn_error fastknn_predict_cuda_mpi(const fastknn_matrix_f32 *train_data,
                                                   const fastknn_labels_i32 *train_labels,
                                                   const fastknn_matrix_f32 *test_data,
                                                   const fastknn_config *config,
                                                   fastknn_result *result) {
    (void)train_data; (void)train_labels; (void)test_data; (void)config; (void)result;
    return FASTKNN_ERROR_BACKEND_NOT_AVAILABLE;
}

#endif

/* CUDA+MPI+OpenMP hybrid stub */
#if defined(FASTKNN_ENABLE_CUDA) && defined(FASTKNN_ENABLE_MPI) && defined(FASTKNN_ENABLE_OPENMP)

extern "C" fastknn_error fastknn_predict_cuda_mpi_openmp(const fastknn_matrix_f32 *train_data,
                                                          const fastknn_labels_i32 *train_labels,
                                                          const fastknn_matrix_f32 *test_data,
                                                          const fastknn_config *config,
                                                          fastknn_result *result) {
    /* TODO: Implement CUDA+MPI+OpenMP version */
    return fastknn_predict_cuda_mpi(train_data, train_labels, test_data, config, result);
}

#else

extern "C" fastknn_error fastknn_predict_cuda_mpi_openmp(const fastknn_matrix_f32 *train_data,
                                                          const fastknn_labels_i32 *train_labels,
                                                          const fastknn_matrix_f32 *test_data,
                                                          const fastknn_config *config,
                                                          fastknn_result *result) {
    (void)train_data; (void)train_labels; (void)test_data; (void)config; (void)result;
    return FASTKNN_ERROR_BACKEND_NOT_AVAILABLE;
}

#endif
