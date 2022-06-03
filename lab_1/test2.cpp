#include <iostream>
#include <mpi.h>

#define epsilon 10e-5
#define N 1000
#define TEG 123

double scalar_mul(const double *vector1, const double *vector2, int count) {
    double len = 0.0f;
    for (int i = 0; i < count; i++) {
        len += vector1[i] * vector2[i];
    }
    double res = 0.0f;
    MPI_Allreduce(&len, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return res;
}

void matrix_mul_vector_divided(const double *A, const double *b, double *result, const int *shift, const int *count, int size) {
    int p_num, p_rank;
    MPI_Comm_size (MPI_COMM_WORLD, &p_num);
    MPI_Comm_rank (MPI_COMM_WORLD, &p_rank);
    int cur_proc_rank;
    auto *p_vector = new double [size / p_num];
    std::copy(p_vector, p_vector + size / p_num, b);
    for (int i = 0; i < p_num; ++i) {
        cur_proc_rank = (p_rank + p_num + i - 1) % p_num;
        MPI_Sendrecv_replace(p_vector, size / p_num + 1, MPI_DOUBLE, (p_rank + 1) % p_num, TEG, (p_rank + p_num - 1) % p_num, TEG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int k = 0; k < count[p_rank]; ++k){
            result[i] = 0.0f;
            for (int l = 0; l < count[cur_proc_rank]; ++l) {
                result[k] += A[k * size + l + shift[cur_proc_rank]] * p_vector[l];
            }
        }
    }
    delete[] p_vector;
}

void CG_divided(const double *A, const double *b, double *result, const int *shift, const int *count, int size) {
    int p_rank;
    MPI_Comm_rank (MPI_COMM_WORLD, &p_rank);
    auto *Az = new double[count[p_rank]];
    auto *r = new double[count[p_rank]];
    auto *z = new double[count[p_rank]];
    // first iteration
    matrix_mul_vector_divided(A, result, Az, shift, count, size);
    for (int i = 0; i < count[p_rank]; i++) {
        r[i] = b[i] - Az[i];
        z[i] = b[i] - Az[i];
    }
    // next iterations
    double alpha, beta, firstScalar, secondScalar;
    while (epsilon * epsilon <= scalar_mul(r, r, count[p_rank]) / scalar_mul(b, b, count[p_rank])) {
        MPI_Barrier(MPI_COMM_WORLD);
        firstScalar = scalar_mul(r, r, count[p_rank]);
        matrix_mul_vector_divided(A, z, Az, shift, count, size);
        secondScalar = scalar_mul(Az, z, count[p_rank]);
        alpha = firstScalar / secondScalar;
        for (int i = 0; i < count[p_rank]; i++) {
            result[i] += alpha * z[i];
            r[i] -= alpha * Az[i];
        }
        secondScalar = scalar_mul(r, r, count[p_rank]);
        beta = secondScalar / firstScalar;
        for (int i = 0; i < count[p_rank]; i++) {
            z[i] = r[i] + (beta * z[i]);
        }
    }
    delete[] Az;
    delete[] r;
    delete[] z;
}

int* init_count(int proc_num) {
    auto *res = new int [proc_num];
    for (int i = 0; i < proc_num; i++){
        if (i < N % proc_num) {
            res[i] = N / proc_num + 1;
        } else {
            res[i] = N / proc_num;
        }
    }
    return res;
}

int* init_shift(const int *count, int proc_num) {
    auto *res = new int [proc_num];
    for (int i = 0; i < proc_num; i++){
        if (i < N % proc_num) {
            res[i] = (i) * (count[i]);
        } else {
            res[i] = (N % proc_num * (count[i] + 1)) + ((i - N % proc_num) * count[i]);
        }
    }
    return res;
}

double* init_matrix(int count, int shift) {
    auto *res = new double [count * N];
    for (int i = 0; i < count; i++) {
        for(int j = 0; j < N; j++) {
            res[i * N + j] = 1.0f;
        }
        res[i * N + shift + i] = 2.0f;
    }
    return res;
}

double* init_vector(int count, double elem) {
    auto *res = new double [count];
    std::fill(res, res + count,  elem);
    return res;
}

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int proc_num, proc_rank;
    MPI_Comm_size (MPI_COMM_WORLD, &proc_num);
    MPI_Comm_rank (MPI_COMM_WORLD, &proc_rank);

    // num of elements to read by each process
    auto *count = init_count(proc_num);
    // shift elements for each process
    auto *shift = init_shift(count, proc_num);
    /* init arguments for conjugate gradients method*/
    auto *A = init_matrix(count[proc_rank], shift[proc_rank]);
    auto *b = init_vector(count[proc_rank], N + 1);
    auto *x = init_vector(count[proc_rank], 0.0f);

    // measuring time work of algorithm on every process
    double startTime = MPI_Wtime();
    CG_divided(A, b, x, shift, count, N);
    double endTime = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < proc_num; ++i) {
        if (i == proc_rank) {
            for (int j = 0; j < count[proc_rank]; j++) {
                std::cout << "index :" << j + 1 << " res: " << x[j] << std::endl;
            }
            std::cout << "Time: " << endTime - startTime << " sec" << std::endl;
        }
    }

    delete[] A;
    delete[] b;
    delete[] x;
    delete[] count;
    delete[] shift;

    MPI_Finalize();
    return 0;
}
