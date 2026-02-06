#include "fastknn/fastknn.h"
#include "../core/internal.h"
#include <stdlib.h>

#ifdef FASTKNN_ENABLE_MPI
#include <mpi.h>

/* Scatter training data across MPI ranks */
static void scatter_train_data(const fastknn_matrix_f32 *train_data,
                                const fastknn_labels_i32 *train_labels,
                                fastknn_matrix_f32 *local_train,
                                fastknn_labels_i32 *local_labels,
                                int rank, int size) {
    int num_features = train_data->cols;
    int total_points = train_data->rows;

    /* Calculate sendcounts and displacements */
    int *sendcounts_data = (int*)malloc(size * sizeof(int));
    int *sendcounts_labels = (int*)malloc(size * sizeof(int));
    int *displs_data = (int*)malloc(size * sizeof(int));
    int *displs_labels = (int*)malloc(size * sizeof(int));

    int quotient = total_points / size;
    int remainder = total_points % size;
    int start = 0;

    for (int i = 0; i < size; i++) {
        int count = quotient + (i < remainder ? 1 : 0);
        sendcounts_data[i] = count * num_features;
        sendcounts_labels[i] = count;
        displs_data[i] = start * num_features;
        displs_labels[i] = start;
        start += count;
    }

    int local_rows = sendcounts_labels[rank];
    *local_train = fastknn_matrix_alloc(local_rows, num_features);
    *local_labels = fastknn_labels_alloc(local_rows);

    MPI_Scatterv(train_data->data, sendcounts_data, displs_data, MPI_FLOAT,
                 local_train->data, sendcounts_data[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(train_labels->data, sendcounts_labels, displs_labels, MPI_INT,
                 local_labels->data, sendcounts_labels[rank], MPI_INT, 0, MPI_COMM_WORLD);

    free(sendcounts_data);
    free(sendcounts_labels);
    free(displs_data);
    free(displs_labels);
}

fastknn_error fastknn_predict_mpi_train(const fastknn_matrix_f32 *train_data,
                                         const fastknn_labels_i32 *train_labels,
                                         const fastknn_matrix_f32 *test_data,
                                         const fastknn_config *config,
                                         fastknn_result *result) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int k = config->k;
    const int num_features = train_data->cols;
    const int num_test = test_data->rows;

    /* Scatter training data across ranks */
    fastknn_matrix_f32 local_train;
    fastknn_labels_i32 local_labels;
    scatter_train_data(train_data, train_labels, &local_train, &local_labels, rank, size);

    /* Broadcast test data to all ranks */
    int num_test_global = num_test;
    MPI_Bcast(&num_test_global, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Each rank computes local top-k for each test point */
    fastknn_pair_t *local_top_k = (fastknn_pair_t*)malloc(num_test * k * sizeof(fastknn_pair_t));
    fastknn_pair_t *pairs = (fastknn_pair_t*)malloc(local_train.rows * sizeof(fastknn_pair_t));

    for (int test_idx = 0; test_idx < num_test; test_idx++) {
        const float *test_point = &test_data->data[test_idx * num_features];

        /* Compute distances to local training points */
        for (int train_idx = 0; train_idx < local_train.rows; train_idx++) {
            const float *train_point = &local_train.data[train_idx * num_features];
            float dist = fastknn_euclidean_distance(test_point, train_point, num_features);
            pairs[train_idx].distance = dist;
            pairs[train_idx].label = local_labels.data[train_idx];
        }

        /* Sort and select top k */
        qsort(pairs, local_train.rows, sizeof(fastknn_pair_t), fastknn_pair_compare);
        int top_count = (k < local_train.rows) ? k : local_train.rows;
        for (int i = 0; i < top_count; i++) {
            local_top_k[test_idx * k + i] = pairs[i];
        }
    }

    free(pairs);

    /* Gather all top-k results to rank 0 */
    fastknn_pair_t *global_top_k = NULL;
    if (rank == 0) {
        global_top_k = (fastknn_pair_t*)malloc(num_test * k * size * sizeof(fastknn_pair_t));
    }

    MPI_Gather(local_top_k, num_test * k * 2, MPI_FLOAT,
               global_top_k, num_test * k * 2, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Rank 0 selects final top-k and makes predictions */
    if (rank == 0) {
        result->predictions = fastknn_labels_alloc(num_test);
        result->distances = NULL;
        result->indices = NULL;
        result->owns_distances = 0;
        result->owns_indices = 0;

        fastknn_pair_t *merged_pairs = (fastknn_pair_t*)malloc(k * size * sizeof(fastknn_pair_t));

        for (int test_idx = 0; test_idx < num_test; test_idx++) {
            /* Collect top-k from all ranks for this test point */
            for (int r = 0; r < size; r++) {
                for (int i = 0; i < k; i++) {
                    merged_pairs[r * k + i] = global_top_k[(r * num_test + test_idx) * k + i];
                }
            }

            /* Sort merged results */
            qsort(merged_pairs, k * size, sizeof(fastknn_pair_t), fastknn_pair_compare);

            /* Vote among final top-k */
            int classified_label = fastknn_most_frequent_label(merged_pairs, k, 10);
            result->predictions.data[test_idx] = classified_label;
        }

        free(merged_pairs);
        free(global_top_k);
    } else {
        /* Non-root ranks initialize empty result */
        result->predictions = fastknn_labels_alloc(0);
        result->distances = NULL;
        result->indices = NULL;
        result->owns_distances = 0;
        result->owns_indices = 0;
    }

    free(local_top_k);
    fastknn_matrix_free(&local_train);
    fastknn_labels_free(&local_labels);

    return FASTKNN_SUCCESS;
}

#else

fastknn_error fastknn_predict_mpi_train(const fastknn_matrix_f32 *train_data,
                                         const fastknn_labels_i32 *train_labels,
                                         const fastknn_matrix_f32 *test_data,
                                         const fastknn_config *config,
                                         fastknn_result *result) {
    (void)train_data; (void)train_labels; (void)test_data; (void)config; (void)result;
    return FASTKNN_ERROR_BACKEND_NOT_AVAILABLE;
}

#endif /* FASTKNN_ENABLE_MPI */
