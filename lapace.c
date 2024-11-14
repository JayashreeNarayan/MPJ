// Implement a lapace filter on a 3-D array in C with cyclic boundary conditions

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void laplace_filter(double *u, double *u_new, int n) {
    int i, j, k;
    int i_prev, i_next, j_prev, j_next, k_prev, k_next;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int n2 = n * n;
    for (i = 0; i < n; i++) {
        i_prev = (i - 1 + n) % n;
        i_next = (i + 1) % n;
        i0 = i * n2;
        i1 = i_prev * n2;
        i2 = i_next * n2;
        for (j = 0; j < n; j++) {
            j_prev = (j - 1 + n) % n;
            j_next = (j + 1) % n;
            j0 = j * n;
            j1 = j_prev * n;
            j2 = j_next * n;
            for (k = 0; k < n; k++) {
                k_prev = (k - 1 + n) % n;
                k_next = (k + 1) % n;
                u_new[i0 + j0 + k] = (
                    u[i1 + j0 + k] +
                    u[i2 + j0 + k] +
                    u[i0 + j1 + k] +
                    u[i0 + j2 + k] +
                    u[i0 + j0 + k_prev] +
                    u[i0 + j0 + k_next] -
                    u[i0 + j0 + k] * 6.0
                    );
            }
        }
    }
}

double ddot(double *u, double *v, int n) {
    int i;
    double result = 0.0;
    for (i = 0; i < n; i++) {
        result += u[i] * v[i];
    }
    return result;
}

void daxpy(double *v, double *u, double *result, double alpha, int n) {
    int i;
    for (i = 0; i < n; i++) {
        result[i] = u[i] + alpha * v[i];
    }
}

void daxpy2(double *v, double *u, double alpha, int n) {
    int i;
    for (i = 0; i < n; i++) {
        u[i] += alpha * v[i];
    }
}

// void print_array(double *u, int n) {
//     int i, j, k;
//     for (i = 0; i < n; i++) {
//         for (j = 0; j < n; j++) {
//             for (k = 0; k < n; k++) {
//                 printf("%18.8Le ", u[i * n * n + j * n + k]);
//             }
//             printf("\n");
//         }
//         printf("\n");
//     }
// }

// int main(int argc, char *argv[]) {
//     int steps = atoi(argv[1]);
//     double div = atof(argv[2]);

//     int n = 3;
//     long int n3 = n * n * n;
//     long int i, j, k;
//     double *u = (double *)malloc(n3 * sizeof(double));
//     double *u_new = (double *)malloc(n3 * sizeof(double));
//     for (i = 0; i < n; i++) {
//         for (j = 0; j < n; j++) {
//             for (k = 0; k < n; k++) {
//                 u[i * n * n + j * n + k] = i * n * n + j * n + k;
//             }
//         }
//     }

//     // print_array(u, n);

    
//     for (i = 0; i < steps; i++) {
//         laplace_filter(u, u_new, n);
//         for (j = 0; j < n3; j++) {
//             u[j] = u_new[j] / div;
//         }
//     }

//     print_array(u, n);
//     // for (i = 0; i < n; i++) {
//     //     for (j = 0; j < n; j++) {
//     //         for (k = 0; k < n; k++) {
//     //             printf("%18.8e ", u[i * n * n + j * n + k]);
//     //         }
//     //         printf("\n");
//     //     }
//     //     printf("\n");
//     // }
//     free(u);
//     free(u_new);
//     return 0;
// }
