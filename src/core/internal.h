#ifndef FASTKNN_INTERNAL_H
#define FASTKNN_INTERNAL_H
#include <stdlib.h>  // calloc, free
#ifdef __cplusplus
extern "C" {
#endif

/* Internal utility functions for backend implementations */

/* Distance functions */
float fastknn_euclidean_distance(const float *a, const float *b, int dim);
float fastknn_euclidean_distance_sq(const float *a, const float *b, int dim);

/* Timing */
void fastknn_get_walltime(double *time);

/* Pair structure for (distance, label) sorting */
typedef struct {
    float distance;
    int label;
} fastknn_pair_t;

/* Comparison function for qsort */
static inline int fastknn_pair_compare(const void *a, const void *b) {
    const fastknn_pair_t *pa = (const fastknn_pair_t*)a;
    const fastknn_pair_t *pb = (const fastknn_pair_t*)b;
    if (pa->distance < pb->distance) return -1;
    if (pa->distance > pb->distance) return 1;
    return 0;
}

/* Find most frequent label in array */
static inline int fastknn_most_frequent_label(const fastknn_pair_t *pairs, int k, int max_label) {
    int *label_count = (int*)calloc(max_label + 1, sizeof(int));
    if (!label_count) return -1;

    for (int i = 0; i < k; i++) {
        if (pairs[i].label >= 0 && pairs[i].label <= max_label) {
            label_count[pairs[i].label]++;
        }
    }

    int classified_label = 0;
    int max_count = label_count[0];
    for (int i = 1; i <= max_label; i++) {
        if (label_count[i] > max_count) {
            max_count = label_count[i];
            classified_label = i;
        }
    }

    free(label_count);
    return classified_label;
}

#ifdef __cplusplus
}
#endif

#endif /* FASTKNN_INTERNAL_H */
