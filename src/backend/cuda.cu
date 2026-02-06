#include "fastknn/fastknn.h"

#ifdef FASTKNN_ENABLE_CUDA

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        return FASTKNN_ERROR_CUDA_ERROR; \
    } \
} while (0)

/* CUDA kernel to compute distances */
__global__ void compute_distances(const float *train_data, const float *test_data,
                                   float *distances, int num_train, int num_test, int num_features) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_test) {
        for (int i = 0; i < num_train; i++) {
            float sum = 0.0f;
            for (int j = 0; j < num_features; j++) {
                float diff = test_data[tid * num_features + j] - train_data[i * num_features + j];
                sum += diff * diff;
            }
            distances[tid * num_train + i] = sqrtf(sum);
        }
    }
}

/* CUDA kernel to select k neighbors and predict */
__global__ void select_k_neighbors(const float *distances, int *predictions, const int *labels,
                                    int num_train, int k, int num_test) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_test) {
        int top_k_indices[10];  /* Assume k <= 10 */
        int votes[10] = {0};     /* Assume max label is 10 */

        /* Initialize indices */
        for (int i = 0; i < k; i++) {
            top_k_indices[i] = i;
        }

        /* Find k smallest distances */
        for (int i = k; i < num_train; i++) {
            float current_dist = distances[tid * num_train + i];
            int max_idx = 0;
            for (int j = 1; j < k; j++) {
                if (distances[tid * num_train + top_k_indices[j]] >
                    distances[tid * num_train + top_k_indices[max_idx]]) {
                    max_idx = j;
                }
            }
            if (current_dist < distances[tid * num_train + top_k_indices[max_idx]]) {
                top_k_indices[max_idx] = i;
            }
        }

        /* Vote */
        for (int i = 0; i < k; i++) {
            int label = labels[top_k_indices[i]];
            if (label >= 0 && label < 10) {
                votes[label]++;
            }
        }

        /* Find most frequent */
        int classified_label = 0;
        int max_votes = votes[0];
        for (int i = 1; i < 10; i++) {
            if (votes[i] > max_votes) {
                max_votes = votes[i];
                classified_label = i;
            }
        }

        predictions[tid] = classified_label;
    }
}

extern "C" fastknn_error fastknn_predict_cuda(const fastknn_matrix_f32 *train_data,
                                               const fastknn_labels_i32 *train_labels,
                                               const fastknn_matrix_f32 *test_data,
                                               const fastknn_config *config,
                                               fastknn_result *result) {
    const int num_train = train_data->rows;
    const int num_test = test_data->rows;
    const int num_features = train_data->cols;
    const int k = config->k;

    /* Set CUDA device */
    if (config->cuda_device >= 0) {
        CUDA_CHECK(cudaSetDevice(config->cuda_device));
    }

    /* Allocate device memory */
    float *d_train_data, *d_test_data, *d_distances;
    int *d_labels, *d_predictions;

    CUDA_CHECK(cudaMalloc(&d_train_data, num_train * num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_test_data, num_test * num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_distances, num_test * num_train * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, num_train * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_predictions, num_test * sizeof(int)));

    /* Copy data to device */
    CUDA_CHECK(cudaMemcpy(d_train_data, train_data->data,
                           num_train * num_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_test_data, test_data->data,
                           num_test * num_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, train_labels->data,
                           num_train * sizeof(int), cudaMemcpyHostToDevice));

    /* Launch kernels */
    int blockSize = 256;
    int numBlocks = (num_test + blockSize - 1) / blockSize;

    compute_distances<<<numBlocks, blockSize>>>(d_train_data, d_test_data, d_distances,
                                                 num_train, num_test, num_features);
    CUDA_CHECK(cudaDeviceSynchronize());

    select_k_neighbors<<<numBlocks, blockSize>>>(d_distances, d_predictions, d_labels,
                                                   num_train, k, num_test);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy results back */
    result->predictions = fastknn_labels_alloc(num_test);
    CUDA_CHECK(cudaMemcpy(result->predictions.data, d_predictions,
                           num_test * sizeof(int), cudaMemcpyDeviceToHost));

    result->distances = NULL;
    result->indices = NULL;
    result->owns_distances = 0;
    result->owns_indices = 0;

    /* Cleanup */
    cudaFree(d_train_data);
    cudaFree(d_test_data);
    cudaFree(d_distances);
    cudaFree(d_labels);
    cudaFree(d_predictions);

    return FASTKNN_SUCCESS;
}

#else

extern "C" fastknn_error fastknn_predict_cuda(const fastknn_matrix_f32 *train_data,
                                               const fastknn_labels_i32 *train_labels,
                                               const fastknn_matrix_f32 *test_data,
                                               const fastknn_config *config,
                                               fastknn_result *result) {
    (void)train_data; (void)train_labels; (void)test_data; (void)config; (void)result;
    return FASTKNN_ERROR_BACKEND_NOT_AVAILABLE;
}

#endif /* FASTKNN_ENABLE_CUDA */
