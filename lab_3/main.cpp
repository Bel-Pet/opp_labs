#include <iostream>
#include <mpi.h>

#define DIMENSION 2
#define MAIN_PROC 0
#define ZERO_BRANCH 0

#define N1 8
#define N2 8
#define N3 8

#define P1 2
#define P2 2

#define MIN 0
#define MAX 1

double* initMatrix(int column, int row) {
    auto *matrix = new double[row * column];
    for (size_t i = 0; i < column; ++i) {
        for (size_t j = 0; j < row; ++j) {
            matrix[i * row + j] = 1.0f;
            if (i == j) matrix[i * row + j] = 2.0f;
            //matrix[i * secondBoard + j] = (MAX - MIN) * ((double) rand() / (double)RAND_MAX) + MIN;
        }
    }
    return matrix;
}

void printMatrix(const double *matrix, int column, int row) {
    for (size_t i = 0; i < column; ++i) {
        for (size_t j = 0; j < row; ++j) {
            std::cout << matrix[i * row + j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
/* return 2D grid communicator for matrix multiplication
 * dims - array number of processes in each block*/
MPI_Comm initGrid2D(const int *dims) {
    int periods[DIMENSION] = {0, 0};
    int reorder = false;
    MPI_Comm result;
    MPI_Cart_create(MPI_COMM_WORLD, DIMENSION, dims, periods, reorder, &result);

    return result;
}
/* return communicator for columns or rows
 * varying_coords - array of dimension belonging to communicator
 * grid - communicator 2D grid*/
MPI_Comm initSplitGrid(int color, int key, MPI_Comm grid) {
    MPI_Comm result;
    //MPI_Cart_sub(grid, varyingCoords, &result);
    MPI_Comm_split(grid, color, key, &result);

    return result;
}
/* return segment matrix of segments matrices multiplication
 * A, B - multiply segments matrices
 * row, column - number of rows and columns result segment
 * shift - number of columns in A and rows in B*/
double* mulSegments(const double *A, const double *B, int row, int column, int shift) {
    auto *result = new double[row * column];
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            result[i * column + j] = 0.0f;
            for (int k = 0; k < shift; ++k) {
                result[i * column + j] += A[i * shift + k] * B[k * column + j];
            }
        }
    }
    return result;
}
/* return assembled matrix form segments on main processes, other processes return nullptr
 * segment - segment of result matrix
 * dims - array number of processes in each block
 * rank - process rank
 * grid - communicator 2D grid*/
double* gatherMatrix(double *segment, const int *dims, int rank, MPI_Comm grid) {
    double *result = nullptr;
    int count[dims[0] * dims[1]], shift[dims[0] * dims[1]];
    int row = N1 / dims[1];
    int column = N3 / dims[0];
    if (rank == MAIN_PROC) {
        result = new double[N1 * N3];
        for (int i = 0; i < dims[0]; ++i) {
            for (int j = 0; j < dims[1]; ++j) {
                count[i * dims[1] + j] = 1;
                shift[i * dims[1] + j] = i * dims[1] * row + j;
            }
        }
    }
    MPI_Datatype segmentType;
    MPI_Type_vector(row, column, N3, MPI_DOUBLE, &segmentType);
    MPI_Type_create_resized(segmentType, 0, column * sizeof(double), &segmentType);
    MPI_Type_commit(&segmentType);

    MPI_Gatherv(segment, row * column, MPI_DOUBLE, result, count, shift, segmentType, MAIN_PROC, grid);

    MPI_Type_free(&segmentType);

    return result;
}
/* return segment of matrix to current process
 * matrix - original matrix
 * pos - coordinate number
 * coord - process coordinates
 * block_size - number of base type elements in each block
 * comm1, comm2 - segment recipient group communicators*/
double* initSegment(double *matrix, int pos, const int *coords, int blockSize, MPI_Comm comm1, MPI_Comm comm2){
    auto *segmentA = new double[blockSize * N2];
    if (pos == 0 && coords[pos] == ZERO_BRANCH) {
        MPI_Scatter(matrix, blockSize * N2, MPI_DOUBLE, segmentA, blockSize * N2, MPI_DOUBLE, MAIN_PROC, comm2);
    }
    if (pos == 1 && coords[pos] == ZERO_BRANCH) {
        MPI_Datatype segmentType;
        MPI_Type_vector(N2, blockSize, N3, MPI_DOUBLE, &segmentType);
        MPI_Type_create_resized(segmentType, 0, blockSize * sizeof(double), &segmentType);
        MPI_Type_commit(&segmentType);

        MPI_Scatter(matrix, 1, segmentType, segmentA, N2 * blockSize, MPI_DOUBLE, MAIN_PROC, comm2);

        MPI_Type_free(&segmentType);
    }
    MPI_Bcast(segmentA, blockSize * N2, MPI_DOUBLE, MAIN_PROC, comm1);
    return segmentA;
}
/* realizable algorithm, return desired matrix main process, other processes return nullptr
 * A, B - multiply matrices
 * size - number of processes
 * rank - current process number
 * dims - array number of processes in each block*/
double* matricesMulGrid2D(double *A, double *B, int size, int rank, const int *dims){
    if (dims[0] * dims[1] != size) {
        std::cout << "Bad dims" << std::endl;
        return nullptr;
    }
    MPI_Comm gridComm = initGrid2D(dims);
    int coords[DIMENSION];
    MPI_Cart_coords(gridComm, rank, DIMENSION, coords);
    MPI_Comm rowComm = initSplitGrid(coords[1], coords[0], gridComm);
    MPI_Comm colComm = initSplitGrid(coords[0], coords[1], gridComm);
    if (N1 % dims[1] != 0 || N3 % dims[0] != 0) {
        std::cout << "Bad data" << std::endl;
        return nullptr;
    }
    int segmentRows = N1 / dims[1];
    int segmentCols = N3 / dims[0];

    auto *segmentA = initSegment(A, 0, coords, segmentRows, rowComm, colComm);
    auto *segmentB = initSegment(B, 1, coords, segmentCols, colComm, rowComm);
    auto *segmentC = mulSegments(segmentA, segmentB, segmentRows, segmentCols, N2);

    double *result = gatherMatrix(segmentC, dims, rank, gridComm);

    delete[] segmentA;
    delete[] segmentB;
    delete[] segmentC;

    return result;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int ProcNum, ProcRank;
    MPI_Comm_size (MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank (MPI_COMM_WORLD, &ProcRank);

    int dims[2] = {P1, P2};
    //MPI_Dims_create(ProcNum, DIMENSION, dims);
    double *A = nullptr;
    double *B = nullptr;
    if (ProcRank == 0) {
        A = initMatrix(N1, N2);
        B = initMatrix(N2, N3);
    }
    double time = -MPI_Wtime();
    double *C = matricesMulGrid2D(A, B, ProcNum, ProcRank, dims);
    time += MPI_Wtime();

    if (ProcRank == 0) {
        printMatrix(A, N1, N2);
        printMatrix(B, N2, N3);
        printMatrix(C, N1, N3);
        std::cout << "Time: " << time << "sec" << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;

    MPI_Finalize();
    return 0;
}
