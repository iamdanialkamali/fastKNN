#include "fastknn/fastknn.h"
#include "../core/internal.h"
#include <stdlib.h>
#include <stdio.h>

fastknn_error fastknn_predict_serial(const fastknn_matrix_f32 *train_data,
                                      const fastknn_labels_i32 *train_labels,
                                      const fastknn_matrix_f32 *test_data,
                                      const fastknn_config *config,
                                      fastknn_result *result) {
    if (!train_data || !train_labels || !test_data || !config || !result) {
        return FASTKNN_ERROR_INVALID_ARGUMENT;
    }

    if (train_data->rows != train_labels->count) {
        return FASTKNN_ERROR_INVALID_ARGUMENT;
    }

    if (train_data->cols != test_data->cols) {
        return FASTKNN_ERROR_INVALID_ARGUMENT;
    }

    const int k = config->k;
    const int num_features = train_data->cols;
    const int num_train = train_data->rows;
    const int num_test = test_data->rows;

    /* Allocate predictions */
    result->predictions = fastknn_labels_alloc(num_test);
    if (!result->predictions.data) {
        return FASTKNN_ERROR_ALLOCATION_FAILED;
    }

    result->distances = NULL;
    result->indices = NULL;
    result->owns_distances = 0;
    result->owns_indices = 0;

    /* Allocate temporary distance-label pairs */
    fastknn_pair_t *pairs = (fastknn_pair_t*)malloc(num_train * sizeof(fastknn_pair_t));
    if (!pairs) {
        fastknn_labels_free(&result->predictions);
        return FASTKNN_ERROR_ALLOCATION_FAILED;
    }

    /* Process each test point */
    for (int test_idx = 0; test_idx < num_test; test_idx++) {
        const float *test_point = &test_data->data[test_idx * num_features];

        /* Compute distances to all training points */
        for (int train_idx = 0; train_idx < num_train; train_idx++) {
            const float *train_point = &train_data->data[train_idx * num_features];
            float dist = fastknn_euclidean_distance(test_point, train_point, num_features);
            pairs[train_idx].distance = dist;
            pairs[train_idx].label = train_labels->data[train_idx];
        }

        /* Sort by distance */
        qsort(pairs, num_train, sizeof(fastknn_pair_t), fastknn_pair_compare);

        /* Find most frequent label among k nearest */
        int max_label = 10;  /* Assume max label is 10 for MNIST */
        int classified_label = fastknn_most_frequent_label(pairs, k < num_train ? k : num_train, max_label);

        result->predictions.data[test_idx] = classified_label;
    }

    free(pairs);
    return FASTKNN_SUCCESS;
}
