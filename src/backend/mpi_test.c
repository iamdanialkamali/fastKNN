#include "fastknn/fastknn.h"

#ifdef FASTKNN_ENABLE_MPI
#include <mpi.h>

/* TODO: Implement MPI test decomposition backend */
fastknn_error fastknn_predict_mpi_test(const fastknn_matrix_f32 *train_data,
                                        const fastknn_labels_i32 *train_labels,
                                        const fastknn_matrix_f32 *test_data,
                                        const fastknn_config *config,
                                        fastknn_result *result) {
    /* Stub: fall back to serial for now */
    return fastknn_predict_serial(train_data, train_labels, test_data, config, result);
}

#else

fastknn_error fastknn_predict_mpi_test(const fastknn_matrix_f32 *train_data,
                                        const fastknn_labels_i32 *train_labels,
                                        const fastknn_matrix_f32 *test_data,
                                        const fastknn_config *config,
                                        fastknn_result *result) {
    (void)train_data; (void)train_labels; (void)test_data; (void)config; (void)result;
    return FASTKNN_ERROR_BACKEND_NOT_AVAILABLE;
}

#endif
