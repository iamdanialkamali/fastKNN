#include "fastknn/fastknn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#define MAX_LINE_SIZE (4096 * 4096)

fastknn_error fastknn_matrix_read_csv(const char *filename,
                                       fastknn_matrix_f32 *matrix,
                                       fastknn_labels_i32 *labels,
                                       int max_rows) {
    if (!filename || !matrix) {
        return FASTKNN_ERROR_INVALID_ARGUMENT;
    }

    FILE *file = fopen(filename, "r");
    if (!file) {
        return FASTKNN_ERROR_IO_ERROR;
    }

    char *buffer = (char*)malloc(MAX_LINE_SIZE);
    if (!buffer) {
        fclose(file);
        return FASTKNN_ERROR_ALLOCATION_FAILED;
    }

    /* First pass: count rows and columns */
    int row_count = 0;
    int col_count = 0;

    while (fgets(buffer, MAX_LINE_SIZE, file)) {
        if (max_rows != -1 && row_count >= max_rows) break;
        row_count++;

        if (row_count == 1) {
            /* Count columns from first row */
            char *tok = strtok(buffer, ",");
            while (tok) {
                col_count++;
                tok = strtok(NULL, ",");
            }
        }
    }

    if (row_count == 0 || col_count == 0) {
        free(buffer);
        fclose(file);
        return FASTKNN_ERROR_IO_ERROR;
    }

    /* Decide if last column is labels */
    int has_labels = (labels != NULL);
    int feature_count = has_labels ? (col_count - 1) : col_count;

    /* Allocate memory */
    *matrix = fastknn_matrix_alloc(row_count, feature_count);
    if (!matrix->data) {
        free(buffer);
        fclose(file);
        return FASTKNN_ERROR_ALLOCATION_FAILED;
    }

    if (has_labels) {
        *labels = fastknn_labels_alloc(row_count);
        if (!labels->data) {
            fastknn_matrix_free(matrix);
            free(buffer);
            fclose(file);
            return FASTKNN_ERROR_ALLOCATION_FAILED;
        }
    }

    /* Second pass: read data */
    rewind(file);
    int row_idx = 0;

    while (fgets(buffer, MAX_LINE_SIZE, file) && row_idx < row_count) {
        int col_idx = 0;
        char *tok = strtok(buffer, ",");

        while (tok && col_idx < col_count) {
            float value = (float)atof(tok);

            if (col_idx < feature_count) {
                /* Feature data */
                matrix->data[row_idx * feature_count + col_idx] = value;
            } else if (has_labels) {
                /* Label (last column) */
                labels->data[row_idx] = (int)value;
            }

            col_idx++;
            tok = strtok(NULL, ",");
        }

        row_idx++;
    }

    free(buffer);
    fclose(file);
    return FASTKNN_SUCCESS;
}
