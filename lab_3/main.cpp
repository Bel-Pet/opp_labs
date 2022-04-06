#include <iostream>
#include <mpi.h>

#define MATRIX_MIN 0
#define MATRIX_MAX 1
#define N1 8
#define N2 4
#define N3 4

double* init_matrix(int column_len, int str_len){
    auto *matrix = new double[str_len * column_len];
    for (int i = 0; i < column_len; ++i) {
        for (int j = 0; j < str_len; ++j) {
            matrix[i * str_len + j] = (double) rand() / (double)RAND_MAX;
            if (i == j) matrix[i * str_len + j] = 1;
        }
    }
    return matrix;
}

void print_matrix(const double * matrix, int column_len, int str_len){
    for (size_t i = 0; i < column_len; ++i) {
        for (size_t j = 0; j < str_len; ++j) {
            printf("%f ", matrix[i * str_len + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void scatter(double *A, double *segmentA, double *B, double *segmentB, int *coords, int segmentRows, int segmentCols, MPI_Comm rowComm, MPI_Comm colComm){
    if (coords[0] == 0) {
        MPI_Scatter(A, segmentRows * N2, MPI_DOUBLE, segmentA, segmentRows * N2, MPI_DOUBLE, 0, colComm);
    }
    if (coords[1] == 0) {
        MPI_Datatype sendSegment;
        MPI_Datatype sendSegmentDouble;

        MPI_Type_vector(N2, segmentCols, N3, MPI_DOUBLE, &sendSegment);
        MPI_Type_commit(&sendSegment);

        MPI_Type_create_resized(sendSegment, 0, segmentCols * sizeof(double), &sendSegmentDouble);
        MPI_Type_commit(&sendSegmentDouble);

        MPI_Scatter(B, 1, sendSegmentDouble, segmentB, N2 * segmentCols, MPI_DOUBLE, 0, rowComm);

        MPI_Type_free(&sendSegment);
        MPI_Type_free(&sendSegmentDouble);
    }

    MPI_Bcast(segmentA, segmentRows * N2, MPI_DOUBLE, 0, rowComm);
    MPI_Bcast(segmentB, N2 * segmentCols, MPI_DOUBLE, 0, colComm);
}

void gather(double * C, double * segmentC, const int * dims, int * coords, int segmentRows, int segmentCols, int ProcNum, MPI_Comm gridComm, MPI_Comm rowComm, MPI_Comm colComm){
    MPI_Datatype recvSegment;
    MPI_Datatype recvSegmentDouble;
    MPI_Type_vector(segmentRows, segmentCols, N3, MPI_DOUBLE, &recvSegment);
    MPI_Type_commit(&recvSegment);
    MPI_Type_create_resized(recvSegment, 0, segmentCols * sizeof(double), &recvSegmentDouble);
    MPI_Type_commit(&recvSegmentDouble);

    int recvCounts[ProcNum];
    std::fill(recvCounts, recvCounts + ProcNum, 1);
    int displs[ProcNum];
    for (int procRank = 0; procRank < ProcNum; ++procRank) {
        MPI_Cart_coords(gridComm, procRank, 2, coords);
        displs[procRank] = dims[0] * segmentRows * coords[1] + coords[0];
    }

    MPI_Gatherv(segmentC, segmentRows * segmentCols, MPI_DOUBLE, C, recvCounts, displs, recvSegmentDouble, 0, gridComm);

    MPI_Type_free(&recvSegment);
    MPI_Type_free(&recvSegmentDouble);
    MPI_Comm_free(&gridComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&rowComm);
}

void mainWork(double * A, double  * B, double * C, int ProcNum, int ProcRank){
    MPI_Comm gridComm, rowComm, colComm;

    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2];
    MPI_Dims_create(ProcNum, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &gridComm);
    MPI_Cart_coords(gridComm, ProcRank, 2, coords);
    MPI_Comm_split(gridComm, coords[1], coords[0], &rowComm);
    MPI_Comm_split(gridComm, coords[0], coords[1], &colComm);
    int segmentRows = N1 / dims[1];
    int segmentCols = N3 / dims[0];
    auto *segmentA = new double[segmentRows * N2];
    auto *segmentB = new double[N2 * segmentCols];
    auto *segmentC = new double[segmentRows * segmentCols];

    std::fill(segmentC, segmentC + segmentRows * segmentCols, 0);

    scatter(A, segmentA, B, segmentB, coords, segmentRows, segmentCols, rowComm, colComm);

    for (int i = 0; i < segmentRows; ++i) {
        for (int k = 0; k < N2; ++k) {
            for (int j = 0; j < segmentCols; ++j) {
                segmentC[i * segmentCols + j] += segmentA[i * N2 + k] * segmentB[k * segmentCols + j];
            }
        }
    }

    gather(C, segmentC, dims, coords, segmentRows, segmentCols, ProcNum, gridComm, rowComm, colComm);

    delete[] segmentA;
    delete[] segmentB;
    delete[] segmentC;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int ProcNum, ProcRank;
    MPI_Comm_size (MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank (MPI_COMM_WORLD, &ProcRank);

    double *A, *B, *C;
    if (ProcRank == 0) {
        A = init_matrix(N1, N2);
        B = init_matrix(N2, N3);
        C = init_matrix(N1, N3);
    }

    mainWork(A, B, C, ProcNum, ProcRank);

    if (ProcRank == 0) {
        print_matrix(A, N1, N2);
        print_matrix(B, N2, N3);
        print_matrix(C, N1, N3);
    }

    delete[] A;
    delete[] B;
    delete[] C;

    MPI_Finalize();
    return 0;
}
