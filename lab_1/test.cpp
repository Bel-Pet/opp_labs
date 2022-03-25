#include <iostream>
#include <cmath>
#include <mpi.h>

#define epsilon 10e-7
#define N 1000

double* init_matrix(int size, int count, int shift) {
    auto *res = new double [count * size];
    for (int i = 0; i < count; i++) {
        for(int j = 0; j < size; j++) {
            res[i * size + j] = 1.0f;
        }
        res[i * size + shift + i] = 2.0f;
    }
    return res;
}

double* init_vector(int size, double elem) {
    auto *res = new double [size];
    std::fill(res, res + size,  elem);
    return res;
}

double scalar_mul(const double * vector1, const double * vector2, int size) {
    double result = 0;
    for (int i = 0; i < size; i++) {
        result += vector1[i] * vector2[i];
    }
    return result;
}

void matrix_mul_vector(const double *matrix, const double *vector, double *result, int size, int *shift, const int* countRowsAtProc, int ProcRank) {
    for(int i = 0; i < countRowsAtProc[ProcRank]; i++){
        result[i] = 0.0f;
        for (int j = 0; j < size; j++) {
            result[i] += matrix[i * size + j] * vector[j];
        }
    }
    MPI_Allgatherv(result, countRowsAtProc[ProcRank], MPI_DOUBLE, result, countRowsAtProc, shift, MPI_DOUBLE, MPI_COMM_WORLD);
}

double finish_iteration(const double *vector1,const double *vector2, int size) {
    double length1 = 0;
    double length2 = 0;
    for (int i = 0; i < size; i++) {
        length1 += pow(vector1[i], 2);
        length2 += pow(vector2[i], 2);
    }
    return sqrt(length1) / sqrt(length2);
}

void conjugate_gradients(double *A, double *b, double *x, double *Az, double *r, double *z, int size, int *shift, int *countRowsAtProc, int ProcRank) {
    // first iteration
    matrix_mul_vector(A, x, Az, size, shift, countRowsAtProc, ProcRank);
    for (int i = 0; i < size; i++) {
        r[i] = b[i] - Az[i];
    }
    std::copy(r, r + size, z);

    // next iterations
    double alpha, beta, firstScalar, secondScalar;
    while (epsilon <= finish_iteration(r, b, size)) {
        firstScalar = scalar_mul(r, r, size);
        matrix_mul_vector(A, z, Az, size, shift, countRowsAtProc, ProcRank);
        secondScalar = scalar_mul(Az, z, size);
        alpha = firstScalar / secondScalar;
        for (int i = 0; i < size; i++) {
            x[i] += alpha * z[i];
            r[i] -= alpha * Az[i];
        }
        secondScalar = scalar_mul(r, r, size);
        beta = secondScalar / firstScalar;
        for (int i = 0; i < size; i++) {
            z[i] = r[i] + (beta * z[i]);
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int ProcNum, ProcRank;
    MPI_Comm_size ( MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank ( MPI_COMM_WORLD, &ProcRank);

    // shift elements for each process
    auto *shift = new int [ProcNum];
    // num of elements to read by each process
    auto *countRowsAtProc = new int [ProcNum];
    for (int i = 0; i < ProcNum; i++){
        if (i < N % ProcNum) {
            countRowsAtProc[i] = N / ProcNum + 1;
            shift[i] = (i) * (countRowsAtProc[i]);
        } else {
            countRowsAtProc[i] = N / ProcNum;
            shift[i] = (N % ProcNum * (countRowsAtProc[i] + 1)) + ((i - N % ProcNum) * countRowsAtProc[i]);
        }
    }

    /* init arguments for conjugate gradients method*/
    auto *A = init_matrix(N, countRowsAtProc[ProcRank], shift[ProcRank]);
    auto *b = init_vector(N, N + 1);
    auto *x = init_vector(N, 0.0f);
    auto *Az = new double [N];
    auto *r = new double [N];
    auto *z = new double [N];

    // measuring time work of algorithm on every process
    double startTime = MPI_Wtime();
    conjugate_gradients(A, b, x, Az, r, z, N, shift, countRowsAtProc, ProcRank);
    double endTime = MPI_Wtime();

    for (int i = 0; i < countRowsAtProc[ProcRank]; i++) {
        std::cout << "index :" << i + 1 << " res: " << x[i] << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Time: " << endTime - startTime << " sec" << std::endl;

    delete[] A;
    delete[] b;
    delete[] x;
    delete[] countRowsAtProc;
    delete[] shift;
    delete[] Az;
    delete[] r;
    delete[] z;

    MPI_Finalize();
    return 0;
}
