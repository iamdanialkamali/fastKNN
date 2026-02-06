#include "fastknn/fastknn.h"
#include "../core/internal.h"

#if defined(FASTKNN_ENABLE_CUDA) && defined(FASTKNN_ENABLE_MPI)

#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
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
__global__ void compute_distances_mpi(const float *train_data, const float *test_data,
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

/* CUDA kernel to select k neighbors */
__global__ void select_k_neighbors_mpi(const float *distances, fastknn_pair_t *top_k,
                                        const int *labels, int num_train, int k, int num_test) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_test) {
        /* Simple selection for top k */
        for (int i = 0; i < k && i < num_train; i++) {
            int min_idx = i;
            for (int j = i + 1; j < num_train; j++) {
                if (distances[tid * num_train + j] < distances[tid * num_train + min_idx]) {
                    min_idx = j;
                }
            }
            /* Store in output */
            top_k[tid * k + i].distance = distances[tid * num_train + min_idx];
            top_k[tid * k + i].label = labels[min_idx];
            /* Swap to maintain sorted order */
            if (min_idx != i) {
                float tmp = distances[tid * num_train + i];
                distances[tid * num_train + i] = distances[tid * num_train + min_idx];
                distances[tid * num_train + min_idx] = tmp;
            }
        }
    }
}

extern "C" fastknn_error fastknn_predict_cuda_mpi(const fastknn_matrix_f32 *train_data,
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
    const int total_train = train_data->rows;

    /* Map rank to GPU (support GPUs 0, 1, 3) */
    int gpu_map[] = {0, 1, 3};
    int gpu_id = gpu_map[rank % 3];
    CUDA_CHECK(cudaSetDevice(gpu_id));

    if (config->verbose > 0 && rank == 0) {
        printf("CUDA+MPI: Using %d GPUs\n", size);
    }

    /* Calculate local training data portion */
    int quotient = total_train / size;
    int remainder = total_train % size;
    int local_train_size = quotient + (rank < remainder ? 1 : 0);
    int train_offset = rank * quotient + (rank < remainder ? rank : remainder);

    /* Allocate device memory */
    float *d_train_data, *d_test_data, *d_distances;
    int *d_labels;
    fastknn_pair_t *d_top_k;

    CUDA_CHECK(cudaMalloc(&d_train_data, local_train_size * num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_test_data, num_test * num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_distances, num_test * local_train_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, local_train_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_top_k, num_test * k * sizeof(fastknn_pair_t)));

    /* Copy local training data to device */
    CUDA_CHECK(cudaMemcpy(d_train_data,
                          &train_data->data[train_offset * num_features],
                          local_train_size * num_features * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_labels,
                          &train_labels->data[train_offset],
                          local_train_size * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_test_data, test_data->data,
                          num_test * num_features * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* Launch kernels */
    int blockSize = 256;
    int numBlocks = (num_test + blockSize - 1) / blockSize;

    compute_distances_mpi<<<numBlocks, blockSize>>>(d_train_data, d_test_data, d_distances,
                                                     local_train_size, num_test, num_features);
    CUDA_CHECK(cudaDeviceSynchronize());

    select_k_neighbors_mpi<<<numBlocks, blockSize>>>(d_distances, d_top_k, d_labels,
                                                       local_train_size, k, num_test);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy local top-k back to host */
    fastknn_pair_t *local_top_k = (fastknn_pair_t*)malloc(num_test * k * sizeof(fastknn_pair_t));
    CUDA_CHECK(cudaMemcpy(local_top_k, d_top_k,
                          num_test * k * sizeof(fastknn_pair_t),
                          cudaMemcpyDeviceToHost));

    /* Gather all top-k results to rank 0 */
    fastknn_pair_t *global_top_k = NULL;
    if (rank == 0) {
        global_top_k = (fastknn_pair_t*)malloc(num_test * k * size * sizeof(fastknn_pair_t));
    }

    MPI_Gather(local_top_k, num_test * k * 2, MPI_FLOAT,
               global_top_k, num_test * k * 2, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Rank 0: Merge and predict */
    if (rank == 0) {
        result->predictions = fastknn_labels_alloc(num_test);
        result->distances = NULL;
        result->indices = NULL;
        result->owns_distances = 0;
        result->owns_indices = 0;

        fastknn_pair_t *merged_pairs = (fastknn_pair_t*)malloc(k * size * sizeof(fastknn_pair_t));

        for (int test_idx = 0; test_idx < num_test; test_idx++) {
            /* Collect top-k from all ranks */
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
        result->predictions = fastknn_labels_alloc(0);
        result->distances = NULL;
        result->indices = NULL;
        result->owns_distances = 0;
        result->owns_indices = 0;
    }

    /* Cleanup */
    free(local_top_k);
    cudaFree(d_train_data);
    cudaFree(d_test_data);
    cudaFree(d_distances);
    cudaFree(d_labels);
    cudaFree(d_top_k);

    return FASTKNN_SUCCESS;
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

#endif /* FASTKNN_ENABLE_CUDA && FASTKNN_ENABLE_MPI */
