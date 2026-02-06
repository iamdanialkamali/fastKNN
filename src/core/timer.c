#ifndef FASTKNN_TIME_H
#define FASTKNN_TIME_H

#include <sys/time.h>
#include <stddef.h>   // NULL

/* Get wall time in seconds */
static inline void fastknn_get_walltime(double *time) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *time = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

#endif  // FASTKNN_TIME_H
