#include <memory.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#include "../../Utils/BLAS_t.hpp"
#ifdef _USE_OPENBLAS_
#include "cblas.h"
extern "C" {
// #include "lapacke.h"
lapack_int LAPACKE_dgesvd(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, double *a, lapack_int lda, double *s, double *u, lapack_int ldu,
                          double *vt, lapack_int ldvt, double *superb);
lapack_int LAPACKE_dgesdd(int matrix_layout, char jobz, lapack_int m, lapack_int n, double *a, lapack_int lda, double *s, double *u, lapack_int ldu, double *vt,
                          lapack_int ldvt);
lapack_int LAPACKE_dsyev(int matrix_layout, char jobz, char uplo, lapack_int n, double *a, lapack_int lda, double *w);
lapack_int LAPACKE_dsyevd(int matrix_layout, char jobz, char uplo, lapack_int n, double *a, lapack_int lda, double *w);
lapack_int LAPACKE_dtrtrs(int matrix_layout, char uplo, char trans, char diag, lapack_int n, lapack_int nrhs, const double *a, lapack_int lda, double *b,
                          lapack_int ldb);
lapack_int LAPACKE_dgeqrf(int matrix_layout, lapack_int m, lapack_int n, double *a, lapack_int lda, double *tau);
lapack_int LAPACKE_dorgqr(int matrix_layout, lapack_int m, lapack_int n, lapack_int k, double *a, lapack_int lda, const double *tau);
lapack_int LAPACKE_dgeqpf(int matrix_layout, lapack_int m, lapack_int n, double *a, lapack_int lda, lapack_int *jpvt, double *tau);
}
#endif
typedef struct {
    int nrows, ncols;
    double *d;
} mat;

typedef struct {
    int nrows;
    double *d;
} vec;
void vector_set_element(vec *v, int row_num, double val);
void matrix_set_element(mat *M, int row_num, int col_num, double val);
typedef int MKL_INT;

double get_seconds_frac(struct timeval start_timeval, struct timeval end_timeval) {
    long secs_used, micros_used;
    secs_used   = (end_timeval.tv_sec - start_timeval.tv_sec);
    micros_used = ((secs_used * 1000000) + end_timeval.tv_usec) - (start_timeval.tv_usec);
    return (micros_used / 1e6);
}

/* initialize new matrix and set all entries to zero */
mat *matrix_new(int nrows, int ncols) {
    mat *M = (mat *)malloc(sizeof(mat));
    // M->d = (double*)mkl_calloc(nrows*ncols, sizeof(double), 64);
    M->d     = (double *)calloc(nrows * ncols, sizeof(double));
    M->nrows = nrows;
    M->ncols = ncols;
    return M;
}

/* initialize new vector and set all entries to zero */
vec *vector_new(int nrows) {
    vec *v = (vec *)malloc(sizeof(vec));
    // v->d = (double*)mkl_calloc(nrows,sizeof(double), 64);
    v->d     = (double *)calloc(nrows, sizeof(double));
    v->nrows = nrows;
    return v;
}

void matrix_delete(mat *M) {
    // mkl_free(M->d);
    free(M->d);
    free(M);
}

void vector_delete(vec *v) {
    // mkl_free(v->d);
    free(v->d);
    free(v);
}

/* copy contents of mat S to D  */
void matrix_copy(mat *D, mat *S) {
    int i;
// #pragma omp parallel for
#pragma omp parallel shared(D, S) private(i)
    {
#pragma omp for
        for (i = 0; i < ((S->nrows) * (S->ncols)); i++) {
            D->d[i] = S->d[i];
        }
    }
}

/* initialize a random matrix */
void initialize_random_matrix(mat *M) {
    int i, m, n;
    double val;
    m       = M->nrows;
    n       = M->ncols;
    float a = 0.0, sigma = 1.0;
    int N = m * n;
    LARNV(N, M->d);
}

/* initialize new matrix and set all entries to zero  for float*/

void matrix_matrix_mult_row(mat *A, mat *B, mat *C) {
    double alpha, beta;
    alpha = 1.0;
    beta  = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A->nrows, B->ncols, A->ncols, alpha, A->d, A->ncols, B->d, B->ncols, beta, C->d, C->ncols);
}

void matrix_transpose_matrix_mult_row(mat *A, mat *B, mat *C) {
    double alpha, beta;
    alpha = 1.0;
    beta  = 0.0;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, A->ncols, B->ncols, A->nrows, alpha, A->d, A->ncols, B->d, B->ncols, beta, C->d, C->ncols);
}

/* C = A*B, A is a file on hard disk */
void matrix_matrix_mult_disk(FILE *A, mat *B, mat *C, int row, int col, int l) {
    int row_size      = l;
    int read_row_size = row_size;
    double alpha, beta;
    int i, j;  // count
    int m = row, n = col, k = B->ncols;

    alpha = 1.0;
    beta  = 0.0;
    // printf("matrix_matrix_mult_disk is running\n");
    float *M_f = (float *)malloc(read_row_size * n * sizeof(float));
    double *M  = (double *)malloc(read_row_size * n * sizeof(double));

    struct timeval start_timeval_1, end_timeval_1;
    struct timeval start_timeval_2, end_timeval_2;
    double sum = 0;
    double time_1, time_2;
    gettimeofday(&start_timeval_1, NULL);

    for (i = 0; i < m; i += row_size) {
        if (row_size > (m - i))
            read_row_size = m - i;
        gettimeofday(&start_timeval_2, NULL);  // time_2
        fread(M_f, sizeof(float), n * read_row_size, A);
#pragma omp parallel shared(M, M_f, n, read_row_size) private(j)
        {
#pragma omp parallel for
            for (j = 0; j < n * read_row_size; j++) {
                M[j] = M_f[j];  // leixing zhuanhuan
            }
        }
        gettimeofday(&end_timeval_2, NULL);
        sum += get_seconds_frac(start_timeval_2, end_timeval_2);

        /* 1*n , n*k  =  1*k , all m*k */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, read_row_size, B->ncols, n, alpha, M, n, B->d, B->ncols, beta, C->d + i * k, C->ncols);
    }
    gettimeofday(&end_timeval_1, NULL);
    time_1 = get_seconds_frac(start_timeval_1, end_timeval_1);

    time_2 = sum;
    printf("Time for reading data file_(fread-time1): %g second\n", time_2);
    printf("Time for matrix_matrix_mult: %g second\n", time_1);

    free(M_f);
    free(M);
}
/* C = A^T*B ; column major */
/* n*m , m*k  =  n*k , all n*k */
void matrix_transpose_matrix_mult_disk(FILE *A, mat *B, mat *C, int row, int col, int l) {
    int row_size      = l;
    int read_row_size = row_size;

    double alpha, beta;
    int i, j;  // count
    int m = row, n = col, k = B->ncols;
    float *M_f = (float *)malloc(read_row_size * n * sizeof(float));
    double *M  = (double *)malloc(read_row_size * n * sizeof(double));
    ;
    // printf("matrix_transpose_matrix_mult_disk is running\n");
    alpha = 1.0;
    beta  = 1.0;
// innitial C=0
#pragma omp parallel shared(C) private(i)
    {
#pragma omp parallel for
        for (i = 0; i < (C->nrows * C->ncols); i++) {
            C->d[i] = 0.0;
        }
    }

    struct timeval start_timeval_1, end_timeval_1;
    struct timeval start_timeval_2, end_timeval_2;
    double sum = 0;
    double time_1, time_2;
    gettimeofday(&start_timeval_1, NULL);

    for (i = 0; i < m; i += row_size) {
        if (row_size > (m - i))
            read_row_size = m - i;
        gettimeofday(&start_timeval_2, NULL);  // time_2
        fread(M_f, sizeof(float), n * read_row_size, A);
        // cblas_dcopy(k, B->d+i*k, 1, g_row, 1); //g_row = g[i];

#pragma omp parallel shared(M, M_f, n, read_row_size) private(j)
        {
#pragma omp parallel for
            for (j = 0; j < n * read_row_size; j++) {
                M[j] = M_f[j];
            }
        }
        gettimeofday(&end_timeval_2, NULL);
        sum += get_seconds_frac(start_timeval_2, end_timeval_2);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, B->ncols, read_row_size, alpha, M, n, B->d + i * k, B->ncols, beta, C->d, C->ncols);
    }

    gettimeofday(&end_timeval_1, NULL);
    time_1 = get_seconds_frac(start_timeval_1, end_timeval_1);

    time_2 = sum;
    printf("Time for reading data file_(fread-time2): %g second\n", time_2);
    printf("Time for matrix_transpose_matrix_mult: %g second\n", time_1);

    free(M_f);
    free(M);
}

/* get element in column major format */
double matrix_get_element(mat *M, int row_num, int col_num) {  // modify
    return M->d[row_num * (M->ncols) + col_num];
}
double vector_get_element(vec *v, int row_num) { return v->d[row_num]; }

/* extract column of a matrix into a vector */
void matrix_get_col(mat *M, int j, vec *column_vec) {  // modify
    int i;
// unclear
#pragma omp parallel shared(column_vec, M, j) private(i)
    {  // unclear
#pragma omp parallel for
        for (i = 0; i < M->nrows; i++) {
            vector_set_element(column_vec, i, matrix_get_element(M, i, j));
        }
    }
}
/* extract row i of a matrix into a vector */
void matrix_get_row(mat *M, int i, vec *row_vec) {  // modify
    int j;
#pragma omp parallel shared(row_vec, M, i) private(j)
    {
#pragma omp parallel for
        for (j = 0; j < M->ncols; j++) {
            vector_set_element(row_vec, j, matrix_get_element(M, i, j));
        }
    }
}
/* set column of matrix to vector */
void matrix_set_col(mat *M, int j, vec *column_vec) {  // modify
    int i;
#pragma omp parallel shared(column_vec, M, j) private(i)
    {
#pragma omp for
        for (i = 0; i < M->nrows; i++) {
            matrix_set_element(M, i, j, vector_get_element(column_vec, i));
        }
    }
}

/* put vector row_vec as row i of a matrix */
void matrix_set_row(mat *M, int i, vec *row_vec) {  // modify
    int j;
#pragma omp parallel shared(row_vec, M, i) private(j)
    {
#pragma omp parallel for
        for (j = 0; j < M->ncols; j++) {
            matrix_set_element(M, i, j, vector_get_element(row_vec, j));
        }
    }
}

/* set vector element */
void vector_set_element(vec *v, int row_num, double val) {  // modify
    v->d[row_num] = val;
}

/* set element in column major format */
void matrix_set_element(mat *M, int row_num, int col_num, double val) {  // modify
    M->d[row_num * (M->ncols) + col_num] = val;
}

/*********************Lijian***********************/

/* Performs [Q,R] = qr(M,'0') compact QR factorization
M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */
void compact_QR_factorization(mat *M, mat *Q, mat *R) {
    int i, j, m, n, k;
    m = M->nrows;
    n = M->ncols;
    k = min(m, n);

    mat *R_full = matrix_new(m, n);
    matrix_copy(R_full, M);
    // vec *tau = vector_new(n);
    vec *tau = vector_new(k);
    // get R
    // printf("get R..\n");
    // LAPACKE_dgeqrf(CblasColMajor, m, n, R_full->d, n, tau->d);
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, R_full->nrows, R_full->ncols, R_full->d, R_full->ncols, tau->d);

    for (i = 0; i < k; i++) {
        for (j = 0; j < k; j++) {
            if (j >= i) {
                matrix_set_element(R, i, j, matrix_get_element(R_full, i, j));
            }
        }
    }

    // get Q
    matrix_copy(Q, R_full);
    // printf("dorgqr..\n");
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, Q->nrows, Q->ncols, min(Q->ncols, Q->nrows), Q->d, Q->ncols, tau->d);

    // clean up
    matrix_delete(R_full);
    vector_delete(tau);
}

/* orth (Q)*/
void QR_factorization_getQ_inplace(mat *Q) {
    int i, j, m, n, k;
    m        = Q->nrows;
    n        = Q->ncols;
    k        = min(m, n);
    vec *tau = vector_new(k);

    /* do QR */  // sometime core dump, bug of MKL
    // LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, Q->d, n, tau->d);

    /* do QRCP */  // more stable, but more expensive
    printf("Warning: use QRCP to replace QR! (see line 269 of matrix_funs_intel_mkl.c)\n");
    int *jpvt = (int *)malloc(sizeof(int) * n);
    LAPACKE_dgeqpf(LAPACK_ROW_MAJOR, m, n, Q->d, n, jpvt, tau->d);
    free(jpvt);

    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, k, Q->d, n, tau->d);
    vector_delete(tau);
}
/* M(:,inds) = Mc */
void matrix_set_selected_columns(mat *M, int *inds, mat *Mc) {
    int i;
    vec *col_vec;
#pragma omp parallel shared(M, Mc, inds) private(i, col_vec)
    {
#pragma omp parallel for
        for (i = 0; i < (Mc->ncols); i++) {
            col_vec = vector_new(M->nrows);
            matrix_get_col(Mc, i, col_vec);
            matrix_set_col(M, inds[i], col_vec);
            vector_delete(col_vec);
        }
    }
}

/* M(inds,:) = Mr */
void matrix_set_selected_rows(mat *M, int *inds, mat *Mr) {  // modify
    int i;
    vec *row_vec;
#pragma omp parallel shared(M, Mr, inds) private(i, row_vec)
    {
#pragma omp parallel for
        for (i = 0; i < (Mr->nrows); i++) {
            row_vec = vector_new(M->ncols);
            matrix_get_row(Mr, i, row_vec);
            matrix_set_row(M, inds[i], row_vec);
            vector_delete(row_vec);
        }
    }
}

/* Mc = M(:,inds) */
void matrix_get_selected_columns(mat *M, int *inds, mat *Mc) {  // modify
    int i;
    vec *col_vec;
#pragma omp parallel shared(M, Mc, inds) private(i, col_vec)
    {
#pragma omp parallel for
        for (i = 0; i < (Mc->ncols); i++) {
            col_vec = vector_new(M->nrows);
            matrix_get_col(M, inds[i], col_vec);
            matrix_set_col(Mc, i, col_vec);
            vector_delete(col_vec);
        }
    }
}

/* C = A*B & D = A^T*C */
void matrix_union_matrix_mult_disk_mem(FILE *A, mat *B, mat *C, mat *D, int row, int col, int row_size) {
    int read_row_size = row_size;

    double alpha, beta, gama;
    int i, j;  // count
    int m = row, n = col, k = B->ncols;
    float *M_f = (float *)malloc(read_row_size * n * sizeof(float));
    double *M  = (double *)malloc(read_row_size * n * sizeof(double));
    // double *g_row= (double*)malloc(k*sizeof(double)); //C's row vector'
    // printf("matrix_union_matrix_mult_disk_mem is running\n");

    alpha = 1.0;
    beta  = 0.0;
    gama  = 1.0;
    struct timeval start_timeval_1, end_timeval_1;
    struct timeval start_timeval_2, end_timeval_2;
    double sum = 0;
    double time_1, time_2;
    gettimeofday(&start_timeval_1, NULL);  // time_1
    //  #pragma omp parallel shared(D) private(i)
    //     {
    //         #pragma omp parallel for
    //         for(i=0; i < (D->nrows*D->ncols); i++){
    //             D->d[i] = 0.0;
    //         }
    //     }

    for (i = 0; i < m; i += row_size) {
        if (row_size > (m - i))
            read_row_size = m - i;
        gettimeofday(&start_timeval_2, NULL);  // time_2
        fread(M_f, sizeof(float), n * read_row_size, A);
#pragma omp parallel shared(M, M_f, n, read_row_size) private(j)
        {
#pragma omp parallel for
            for (j = 0; j < n * read_row_size; j++) {
                M[j] = M_f[j];
            }
        }
        gettimeofday(&end_timeval_2, NULL);
        sum += get_seconds_frac(start_timeval_2, end_timeval_2);
        // C = A*D
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, read_row_size, B->ncols, n, alpha, M, n, B->d, B->ncols, beta, C->d + i * k, C->ncols);
        // B = A^T*C exchange B & C
        // cblas_dcopy(k, C->d+i*k, 1, g_row, 1); //g_row = g[i];
        // cblas_dger(CblasRowMajor, D->nrows, D->ncols, alpha, M, 1, g_row, 1, D->d, D->ncols); //A := alpha*x*y'+ A,
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, C->ncols, read_row_size, alpha, M, n, C->d + i * k, C->ncols, gama, D->d, D->ncols);
    }

    gettimeofday(&end_timeval_1, NULL);
    time_1 = get_seconds_frac(start_timeval_1, end_timeval_1);

    time_2 = sum;

    printf("Time for reading data file_(fread-time): %g second\n", time_2);
    printf("Time for matrix_union_matrix_mult: %g second\n", time_1);
    // printf("matrix_union_mem is %d KB\n",  getCurrentRSS()/1024);

    free(M_f);
    free(M);
}

// input A B C
// output D E
//  D=A*B E=A^T*C
void matrix_union_matrix_mult_disk_mem_2(FILE *A, mat *B, mat *C, mat *D, mat *E, int row, int col, int row_size) {
    int read_row_size = row_size;

    double alpha, beta, gama;
    int i, j;  // count
    int m = row, n = col, k = B->ncols;
    float *M_f = (float *)malloc(read_row_size * n * sizeof(float));
    double *M  = (double *)malloc(read_row_size * n * sizeof(double));
    // double *g_row= (double*)malloc(k*sizeof(double)); //C's row vector'
    // printf("matrix_union_matrix_mult_disk_mem_2 is running\n");

    // matrix_copy(D,B); //D=B

    // float *a_g_mult=(float*)malloc(n*k*sizeof(float)); // ai * gi , n*k
    alpha = 1.0;
    beta  = 0.0;
    gama  = 1.0;
    struct timeval start_timeval_1, end_timeval_1;
    struct timeval start_timeval_2, end_timeval_2;
    double sum = 0;
    double time_1, time_2;
    gettimeofday(&start_timeval_1, NULL);  // time_1
    //  #pragma omp parallel shared(E) private(i)
    //     {
    //         #pragma omp parallel for
    //         for(i=0; i < (E->nrows*E->ncols); i++){
    //             E->d[i] = 0.0;
    //         }
    //     }

    for (i = 0; i < m; i += row_size) {
        if (row_size > (m - i))
            read_row_size = m - i;
        gettimeofday(&start_timeval_2, NULL);  // time_2
        fread(M_f, sizeof(float), n * read_row_size, A);
#pragma omp parallel shared(M, M_f, n, read_row_size) private(j)
        {
#pragma omp parallel for
            for (j = 0; j < n * read_row_size; j++) {
                M[j] = M_f[j];
            }
        }
        gettimeofday(&end_timeval_2, NULL);
        sum += get_seconds_frac(start_timeval_2, end_timeval_2);
        // C = A*D
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, read_row_size, B->ncols, n, alpha, M, n, B->d, B->ncols, beta, D->d + i * k, D->ncols);
        // B = A^T*C exchange B & C
        // cblas_dcopy(k, C->d+i*k, 1, g_row, 1); //g_row = g[i];
        // cblas_dger(CblasRowMajor, D->nrows, D->ncols, alpha, M, 1, g_row, 1, D->d, D->ncols); //A := alpha*x*y'+ A,
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, C->ncols, read_row_size, alpha, M, n, C->d + (i * C->ncols), C->ncols, gama, E->d, E->ncols);
    }

    gettimeofday(&end_timeval_1, NULL);
    time_1 = get_seconds_frac(start_timeval_1, end_timeval_1);

    time_2 = sum;

    printf("Time for reading data file_(fread-time): %g second\n", time_2);
    printf("Time for matrix_union_matrix_mult2: %g second\n", time_1);

    free(M_f);
    free(M);
}
/*  k*n = k*k k*k n*k  */
void svd_row_cut(mat *A, mat *U, vec *E, mat *V) {
    int m = A->nrows;
    int n = A->ncols;
    int i, j;
    // mat *A_in = matrix_new(m,n);;

    // matrix_copy(A_in, A);
    // printf("dong tai sheng qing\n");
    // double *u = (double*)malloc(m*m*sizeof(double));
    double *vt = (double *)malloc(n * m * sizeof(double));

    // printf("svd is running\n");
    // LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'S', m, n, A->d, n, E->d, U->d, m, vt, n, superb);

    // LAPACKE_dgesdd( int matrix_layout, char jobz, lapack_int m, lapack_int n, double* a, lapack_int lda, double* s, double* u, lapack_int ldu, double* vt,
    // lapack_int ldvt );
    LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', m, n, A->d, n, E->d, U->d, m, vt, n);
    // printf("Complete Lapack svd\n\n");

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            V->d[j * m + i] = vt[i * n + j];
        }
    }

    // printf("svd_row_cut is over\n");

    // matrix_delete(A_in);
    free(vt);
}

/* M = USV^T */
void svd_row(mat *M, mat *U, mat *S, mat *V) {
    int m, n;
    m = M->nrows;
    n = M->ncols;
    // k = min(m,n);
    vec *work = vector_new(2 * max(3 * min(m, n) + max(m, n), 5 * min(m, n)));
    // vec * svals = vector_new(k);
    double *vt = (double *)malloc(n * n * sizeof(double));

    LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', m, n, M->d, n, S->d, U->d, n, vt, n, work->d);

    // initialize_diagonal_matrix(S, svals);
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            V->d[j * n + i] = vt[i * n + j];
        }
    }
    free(vt);
    vector_delete(work);
}

/* D = M(:,inds)' */
void matrix_get_selected_columns_and_transpose(mat *M, int *inds, mat *Mc) {
    int i;
    vec *col_vec;
#pragma omp parallel shared(M, Mc, inds) private(i, col_vec)
    {
#pragma omp parallel for
        for (i = 0; i < (Mc->nrows); i++) {
            col_vec = vector_new(M->nrows);
            matrix_get_col(M, inds[i], col_vec);
            matrix_set_row(Mc, i, col_vec);
            vector_delete(col_vec);
        }
    }
}
void linear_solve_UTxb(mat *A, mat *b) {
    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'T', 'N',  // unclear
                   b->nrows, b->ncols, A->d, A->ncols, b->d, b->ncols);
}

/* C = beta*C + alpha*A(1:Anrows, 1:Ancols)[T]*B(1:Bnrows, 1:Bncols)[T] */
void submatrix_submatrix_mult_with_ab(mat *A, mat *B, mat *C, int Anrows, int Ancols, int Bnrows, int Bncols, int transa, int transb, double alpha,
                                      double beta) {
    int opAnrows, opAncols, opBnrows, opBncols;
    if (transa == CblasTrans) {
        opAnrows = Ancols;
        opAncols = Anrows;
    } else {
        opAnrows = Anrows;
        opAncols = Ancols;
    }

    if (transb == CblasTrans) {
        opBnrows = Bncols;
        opBncols = Bnrows;
    } else {
        opBnrows = Bnrows;
        opBncols = Bncols;
    }

    if (opAncols != opBnrows) {
        printf("error in submatrix_submatrix_mult()");
        exit(0);
    }

    cblas_dgemm(CblasRowMajor, (CBLAS_TRANSPOSE)transa, (CBLAS_TRANSPOSE)transb, opAnrows, opBncols,  // m, n,
                opAncols,                                                                             // k
                alpha, A->d, A->ncols,                                                                // lda // modify
                B->d, B->ncols,                                                                       // ldb
                beta, C->d, C->ncols                                                                  // ldc
    );
}
void submatrix_submatrix_mult(mat *A, mat *B, mat *C, int Anrows, int Ancols, int Bnrows, int Bncols, int transa, int transb) {
    double alpha, beta;
    alpha = 1.0;
    beta  = 0.0;
    submatrix_submatrix_mult_with_ab(A, B, C, Anrows, Ancols, Bnrows, Bncols, transa, transb, alpha, beta);
}

void matrix_get_selected_columns_new(mat *M, int *inds, mat *Mc) {  // modify
    int i;
    vec *col_vec = vector_new(M->nrows);
    for (i = 0; i < (Mc->ncols); i++) {
        matrix_get_col(M, inds[i], col_vec);
        matrix_set_col(Mc, i, col_vec);
    }
    vector_delete(col_vec);
}

void matrix_sub_d(mat *A, mat *B, double d) {
    int i;
    int len = (A->nrows) * (A->ncols);
// #pragma omp parallel for
#pragma omp parallel shared(A, B, d) private(i)
    {
#pragma omp for
        for (i = 0; i < len; i++) {
            A->d[i] = A->d[i] - d * B->d[i];
        }
    }
}

/*[U, S, V] = eigSVD(A)*/
void eigSVD(mat *A, mat *U, mat *S, mat *V) {
    matrix_transpose_matrix_mult_row(A, A, V);
    LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', V->ncols, V->d, V->ncols, S->d);
    mat *V1 = matrix_new(V->ncols, V->ncols);
    matrix_copy(V1, V);
    MKL_INT i, j;
#pragma omp parallel shared(V1, S) private(i, j)
    {
#pragma omp for
        for (i = 0; i < V1->nrows; i++) {
            S->d[i] = sqrt(S->d[i]);
            for (j = 0; j < V1->ncols; j++) {
                V1->d[j * V1->ncols + i] /= S->d[i];
            }
        }
    }
    mat *Uc = matrix_new(U->nrows, U->ncols);
    matrix_matrix_mult_row(A, V1, Uc);
    matrix_copy(U, Uc);
    matrix_delete(Uc);
    matrix_delete(V1);
}

void basic_rSVD(char *filename, int m, int n, int k, int l, int p, mat **U, mat **S, mat **V) {
    int i, j;
    FILE *fid;
    mat *Qt = matrix_new(m, l);
    mat *Q  = matrix_new(n, l);

    initialize_random_matrix(Q);
    fid = fopen(filename, "rb");
    matrix_matrix_mult_disk(fid, Q, Qt, m, n, k);
    fclose(fid);
    QR_factorization_getQ_inplace(Qt);
    for (i = 1; i <= p; i++) {
        fid = fopen(filename, "rb");
        matrix_transpose_matrix_mult_disk(fid, Qt, Q, m, n, k);
        fclose(fid);
        fid = fopen(filename, "rb");
        matrix_matrix_mult_disk(fid, Q, Qt, m, n, k);
        fclose(fid);
        QR_factorization_getQ_inplace(Qt);
    }
    fid = fopen(filename, "rb");
    matrix_transpose_matrix_mult_disk(fid, Qt, Q, m, n, k);
    fclose(fid);

    mat *UU = matrix_new(l, l);
    mat *VV = matrix_new(n, l);
    mat *SS = matrix_new(l, 1);
    svd_row(Q, VV, SS, UU);
    // puts("1");
    mat *UUk = matrix_new(l, k);
    int inds[k];
    *S = matrix_new(k, 1);
    for (i = 0; i < k; i++) {
        inds[i]    = i;
        (*S)->d[i] = SS->d[i];
    }
    *U = matrix_new(m, k);
    matrix_get_selected_columns_new(UU, inds, UUk);
    matrix_matrix_mult_row(Qt, UUk, *U);
    *V = matrix_new(n, k);
    matrix_get_selected_columns_new(VV, inds, *V);
    matrix_delete(Q);
    matrix_delete(Qt);
    matrix_delete(UU);
    matrix_delete(VV);
    matrix_delete(SS);
    matrix_delete(UUk);
}

void matrix_union_matrix_mult_A(float *A, mat *B, mat *C, mat *D, int row, int col, int row_size) {
    int read_row_size = row_size;

    double alpha, beta, gama;
    int i, j;  // count
    int m = row, n = col, k = B->ncols;
    float *M_f = A;  //(float*)malloc(read_row_size*n*sizeof(float));
    double *M  = (double *)malloc(read_row_size * n * sizeof(double));
    // double *g_row= (double*)malloc(k*sizeof(double)); //C's row vector'
    // printf("matrix_union_matrix_mult_disk_mem is running\n");

    alpha = 1.0;
    beta  = 0.0;
    gama  = 1.0;
    struct timeval start_timeval_1, end_timeval_1;
    struct timeval start_timeval_2, end_timeval_2;
    double sum = 0;
    double time_1, time_2;
    gettimeofday(&start_timeval_1, NULL);  // time_1

    for (i = 0; i < m; i += row_size) {
        if (row_size > (m - i))
            read_row_size = m - i;
        gettimeofday(&start_timeval_2, NULL);  // time_2
// fread(M_f, sizeof(float), n*read_row_size, A);
#pragma omp parallel shared(M, M_f, n, read_row_size) private(j)
        {
#pragma omp parallel for
            for (j = 0; j < n * read_row_size; j++) {
                M[j] = M_f[j];
            }
        }
        gettimeofday(&end_timeval_2, NULL);
        sum += get_seconds_frac(start_timeval_2, end_timeval_2);
        // C = A*D
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, read_row_size, B->ncols, n, alpha, M, n, B->d, B->ncols, beta, C->d + i * k, C->ncols);
        // B = A^T*C exchange B & C
        // cblas_dcopy(k, C->d+i*k, 1, g_row, 1); //g_row = g[i];
        // cblas_dger(CblasRowMajor, D->nrows, D->ncols, alpha, M, 1, g_row, 1, D->d, D->ncols); //A := alpha*x*y'+ A,
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, C->ncols, read_row_size, alpha, M, n, C->d + i * k, C->ncols, gama, D->d, D->ncols);
        M_f += n * read_row_size;
    }
    // gettimeofday(&end_timeval_1, NULL);
    // time_1 = get_seconds_frac(start_timeval_1 ,end_timeval_1);
    // time_2 = sum;
    // printf("Time for reading data file_(fread-time): %g second\n",time_2);
    // printf("Time for matrix_union_matrix_mult: %g second\n", time_1);
    // printf("matrix_union_mem is %d KB\n",  getCurrentRSS()/1024);
    free(M);
}
//  PerSVD(filename, 1000, 1000, 50, 50+25, 2, &U1, &S1, &V1);
// void PerSVD(char *filename, int m, int n, int k, int l, int p, double *M,mat **U, mat **S, mat **V)       {
int PerSVD(char *filename, int m, int n, int k, int l, int p, float *A, float *mU, int ldU, float *sigma, float *mVt, int ldV, int flag = 0x0) {
    int loop, i, j;
    mat *Qt = matrix_new(m, l);
    mat *Q  = matrix_new(n, l);
    mat *VV = matrix_new(l, l);
    mat *SS = matrix_new(l, 1);

    initialize_random_matrix(Q);
    eigSVD(Q, Q, SS, VV);

    mat *D1 = matrix_new(l, l);
    mat *D2 = matrix_new(l, l);
    mat *st = matrix_new(l, 1);

    int niter = p + 1;

    double alpha = 0;
    mat *QQ;

    for (loop = 1; loop <= niter; loop++) {
        QQ = matrix_new(n, l);

        if (filename != nullptr) {
            FILE *fid = fopen(filename, "rb");
            matrix_union_matrix_mult_disk_mem(fid, Q, Qt, QQ, m, n, k);
            fclose(fid);
        } else {
            matrix_union_matrix_mult_A(A, Q, Qt, QQ, m, n, k);
        }
        if (loop == niter)
            break;

        matrix_transpose_matrix_mult_row(QQ, QQ, D1);
        matrix_transpose_matrix_mult_row(Qt, Qt, D2);
        double alpha1 = alpha;
        double alpha2 = 0;
        double alpha_tol;
        for (j = 0; j < 100; j++) {
            matrix_copy(VV, D1);
            matrix_sub_d(VV, D2, 2.0 * alpha1);
            LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', VV->ncols, VV->d, VV->ncols, st->d);
            double stl = sqrt(st->d[0] + alpha1 * alpha1);
            if (stl < 1e-10)
                break;
            if (stl < alpha1)
                break;
            alpha2    = alpha1;
            alpha1    = (alpha1 + stl) / 2;
            alpha_tol = (alpha1 - alpha2) / alpha1;
            if (alpha_tol < 1e-2)
                break;
        }
        alpha = alpha1;

        matrix_sub_d(QQ, Q, alpha);
        svd_row(QQ, Q, SS, VV);

        SS->d[0] = SS->d[l - 1];

        if (alpha < SS->d[0])
            alpha = (alpha + SS->d[0]) / 2;

        matrix_delete(QQ);
    }
    matrix_copy(Q, QQ);
    matrix_delete(QQ);
    matrix_delete(D1);
    matrix_delete(D2);
    matrix_delete(st);
    eigSVD(Qt, Qt, SS, VV);
    mat *Sl = matrix_new(l, l);
    for (i = 0; i < l; i++) {
        Sl->d[i * l + i] = 1.0 / SS->d[i];
    }
    mat *SiV = matrix_new(l, l);
    matrix_matrix_mult_row(VV, Sl, SiV);
    mat *B = matrix_new(n, l);
    matrix_matrix_mult_row(Q, SiV, B);
    matrix_delete(SiV);
    eigSVD(B, Q, SS, VV);
    matrix_delete(B);
    matrix_delete(Sl);
    int inds[k];
    int s = l - k;

    mat *S = matrix_new(k, 1);
    for (i = s; i < l; i++) {
        inds[i - s]   = i;
        (S)->d[i - s] = SS->d[i];
        sigma[i - s]  = SS->d[i];
    }
    matrix_delete(S);
    mat *U   = matrix_new(m, k);
    mat *VV2 = matrix_new(k + s, k);
    matrix_get_selected_columns_new(VV, inds, VV2);
    matrix_delete(VV);
    matrix_matrix_mult_row(Qt, VV2, U);
    matrix_delete(Qt);
    for (j = 0; j < m * k; j++) {
        mU[j] = U->d[j];
    }
    mat *V = matrix_new(n, k);
    matrix_get_selected_columns_new(Q, inds, V);
    matrix_delete(Q);
    matrix_delete(SS);
    matrix_delete(VV2);
    float *row  = mVt;
    double *src = V->d;
    for (i = 0; i < k; i++, row += n) {
        for (j = 0; j < n; j++) {
            row[j] = src[j * k + i];
        }
    }
    matrix_delete(U);
    matrix_delete(V);
    return 0x0;
}
