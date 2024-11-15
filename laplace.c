// Implement a lapace filter on a 3-D array in C with cyclic boundary conditions

#include <stdio.h>
#include <stdlib.h>
// #include <string.h>
#include <math.h>
// #include <time.h>

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

double norm(double *u, int n) {
    return sqrt(ddot(u, u, n));
}

int conj_grad(double *b, double *x0, double *x, double tol, int n) {
    long int i;
    long int n3 = n * n * n;
    int iter = 0;

    double *v = (double *)malloc(n3 * sizeof(double));
    double *r = (double *)malloc(n3 * sizeof(double));
    double *p = (double *)malloc(n3 * sizeof(double));
    double *Ap = (double *)malloc(n3 * sizeof(double));
    double alpha, beta, r_dot_v;

    double app;

    for (i = 0; i < n3; i++) {
        x[i] = x0[i];
    }
    laplace_filter(x, r, n);
    daxpy2(b, r, -1.0, n3);

    for (i = 0; i < n3; i++) {
        app = r[i] / 6.0;
        p[i] = app;
        v[i] = -app;
    }

    do {
        iter++;
        laplace_filter(p, Ap, n);
        r_dot_v = ddot(r, v, n3);

        alpha = r_dot_v / ddot(p, Ap, n3);
        daxpy2(p, x, alpha, n3);
        daxpy2(Ap, r, alpha, n3);

        for (i = 0; i < n3; i++) {
            v[i] = -r[i] / 6.0;
        }

        beta = ddot(r, v, n3) / r_dot_v;

        for (i = 0; i < n3; i++) {
            p[i] = beta * p[i] - v[i];
        }        

    } while(norm(r, n3) > tol);

    free(v);
    free(r);
    free(p);
    free(Ap);

    return iter;
}
