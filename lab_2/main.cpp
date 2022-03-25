#include <iostream>
#include <cmath>
#include <ctime>
#include <omp.h>

#define epsilon 10e-7
#define N 1000

double scalar_mul(const double * vector1, const double * vector2, int size) {
    int i;
    double resultScalar = 0;
#pragma omp parallel for shared(vector1, vector2, size) private (i) default (none) reduction(+:resultScalar)
    for (i = 0; i < size; i++) {
        resultScalar += vector1[i] * vector2[i];
    }
    return resultScalar;
}

void vectors_sub(const double *vector1, const double *vector2, double *resultVector, int size) {
    int i;
#pragma omp parallel for shared(vector1, vector2, resultVector, size) default (none) private (i)
    for (i = 0; i < size; i++) {
        resultVector[i] = vector1[i] - vector2[i];
    }
}

void matrix_mul_vector(const double *matrix, const double *vector1, double *resultVector, int size) {
    int i, j;
#pragma omp parallel for shared(matrix, vector1, resultVector, size) default (none)  private (i, j)
    for (i = 0; i < size; i++) {
        resultVector[i] = 0.0f;
        for (j = 0; j < size; j++) {
            resultVector[i] += matrix[i * size + j] * vector1[j];
        }
    }
}

double finish_count(const double *vector1, const double *vector2, int size) {
    double result;
    int i;
    double lenOfVec_r = 0;
    double lenOfVec_b = 0;
#pragma omp parallel for shared(rVector, size) private (i) default (none) reduction(+:lenOfVec_r)
    for (i = 0; i < size; i++) {
        lenOfVec_r += vector1[i] * vector1[i];
    }
#pragma omp parallel for shared(bVector, size) private (i) default (none) reduction(+:lenOfVec_b)
    for (i = 0; i < size; i++) {
        lenOfVec_b += vector2[i] * vector2[i];
    }
    result = sqrt(lenOfVec_r) / sqrt(lenOfVec_b);
    return result;
}

void conjugate_gradients(double *A, double *b, double *x, double *Az, double *r, double *z, int size) {
    // first iteration
    matrix_mul_vector(A, x, Az, size);
    vectors_sub(b, Az, r, size);
    std::copy(r, r + size, z);
    // next iterations
    double finish = finish_count(r, b, size);
    double  alpha, beta, firstScalar, secondScalar;
    int i;
    while (epsilon <= finish) {
        firstScalar = scalar_mul(r, r, size);
        matrix_mul_vector(A, z, Az, size);
        secondScalar = scalar_mul(Az, z, size);
        alpha = firstScalar / secondScalar;
#pragma omp parallel for shared(x, r, z, Az, alpha, size) private (i) default (none)
        for (i = 0; i < size; i++) {
            x[i] += alpha * z[i];
            r[i] -= alpha * Az[i];
        }
        secondScalar = scalar_mul(r, r, size);
        beta = secondScalar / firstScalar;
#pragma omp parallel for shared(z, r, size, beta) private (i) default (none)
        for (i = 0; i < size; i++) {
            z[i] = r[i] + (beta * z[i]);
        }
    }
}

double* init_matrix(int size){
    int i, j;
    auto res = new double [size * size];
#pragma omp parallel for shared(A, size) default (none)  private (i, j)
    for (i = 0; i < size; i++) {
        for(j = 0; j < size; j++) {
            res[i * size + j] = 1.0f;
            if (i == j){
                res[i * size + j] = 2.0f;
            }
        }
    }
    return res;
}

double* init_vector(int size, double elem) {
    auto res = new double [size];
    int i;
#pragma omp parallel for shared(x, size) default (none)  private (i)
    for(i = 0; i < size; i++) {
        res[i] = elem;
    }
    return res;
}

int main() {
    // init arguments for conjugate gradients method
    auto *A = init_matrix(N);
    auto *b = init_vector(N, N + 1);
    auto *x = init_vector(N, 0.0f);
    auto *Az = new double [N];
    auto *r = new double [N];
    auto *z = new double [N];

    // measuring time work of algorithm
    struct timespec start{}, finish{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    conjugate_gradients(A, b, x, Az, r, z, N);
    clock_gettime(CLOCK_MONOTONIC_RAW, &finish);

    for (int i = 0; i < N; i++) {
        std::cout << "index :" << i + 1 << " res: " << x[i] << std::endl;
    }

    std::cout << "Time: " << ((double) finish.tv_sec - start.tv_sec + 0.000000001 * (double) (finish.tv_nsec - start.tv_nsec)) << '\n';

    delete[] A;
    delete[] b;
    delete[] x;
    delete[] Az;
    delete[] r;
    delete[] z;

    return 0;
}