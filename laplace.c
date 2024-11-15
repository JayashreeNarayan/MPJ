#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
Apply a 3-D Laplace filter to a 3-D array with cyclic boundary conditions
@param u: the input array
@param u_new: the output array
@param n: the size of the array in each dimension
*/
void laplace_filter(double *u, double *u_new, int n) {
    int i, j, k;
    int i_prev, i_next, j_prev, j_next, k_prev, k_next;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int n2 = n * n;
    #pragma omp parallel for private(i, j, k, i_prev, i_next, j_prev, j_next, k_prev, k_next, i0, i1, i2, j0, j1, j2)
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

/*
Compute the dot product of two vectors
@param u: the first vector
@param v: the second vector
@param n: the size of the vectors
@return the dot product of the two vectors
*/
double ddot(double *u, double *v, int n) {
    int i;
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < n; i++) {
        result += u[i] * v[i];
    }
    return result;
}

/*
Compute the sum of two vectors scaled by a constant (res = u + alpha * v)
and store the result in a third vector
@param v: the first vector
@param u: the second vector
@param result: the vector to store the result
@param alpha: the scaling constant
@param n: the size of the vectors
*/
void daxpy(double *v, double *u, double *result, double alpha, int n) {
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        result[i] = u[i] + alpha * v[i];
    }
}

/*
Compute the sum of two vectors scaled by a constant (u += alpha * v)
and store the result in the second vector
@param v: the first vector
@param u: the second vector
@param alpha: the scaling constant
@param n: the size of the vectors
*/
void daxpy2(double *v, double *u, double alpha, int n) {
    int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        u[i] += alpha * v[i];
    }
}

/*
Compute the Euclidean norm of a vector
@param u: the vector
@param n: the size of the vector
@return the Euclidean norm of the vector
*/
double norm(double *u, int n) {
    return sqrt(ddot(u, u, n));
}

/*
Solve the system of linear equations Ax = b using the conjugate gradient method where A is the Laplace filter
@param b: the right-hand side of the system of equations
@param x0: the initial guess for the solution
@param x: the solution to the system of equations
@param tol: the tolerance for the solution
@param n: the size of the arrays (n_tot = n * n * n)
*/
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

    #pragma omp parallel for
    for (i = 0; i < n3; i++) {
        x[i] = x0[i];
    }
    laplace_filter(x, r, n);
    daxpy2(b, r, -1.0, n3);

    #pragma omp parallel for
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

        #pragma omp parallel for
        for (i = 0; i < n3; i++) {
            v[i] = -r[i] / 6.0;
        }

        beta = ddot(r, v, n3) / r_dot_v;

        #pragma omp parallel for
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
