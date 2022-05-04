#include <iostream>
#include<cmath>
#include <algorithm>
#include <mpi.h>

#define EPSILON 1e-8
#define PARAMETER 1e5

#define NX 300
#define NY 300
#define NZ 300

#define DX 2.0f
#define DY 2.0f
#define DZ 2.0f

constexpr double hx = DX / (NX - 1);
constexpr double hy = DY / (NY - 1);
constexpr double hz = DZ / (NZ - 1);
constexpr double h2X = hx * hx;
constexpr double h2Y = hy * hy;
constexpr double h2Z = hz * hz;
constexpr double factor = 1 / (2 / h2X + 2 / h2Y + 2 / h2Z + PARAMETER);

double phi(double x, double y, double z) {
    return x * x + y * y + z * z;
}

double tail(double x, double y, double z) {
    return 6 - phi(x, y, z) * PARAMETER;
}
/* send and receive size item other processes
 * pos - starting position of data sending
 * size - number of elements
 * rank - send and receive process rank
 * F - buffer data sending
 * tag - received tag
 * buf - buffer data received
 * sReg, rReg - send and receive request*/
void sendBorders(int pos, int size, int rank, double *F, int tag, double *buf, MPI_Request *sReq, MPI_Request *rReq) {
    MPI_Isend(&(F[pos]), size, MPI_DOUBLE, rank, tag, MPI_COMM_WORLD, &sReq[tag]);
    MPI_Irecv(buf, size, MPI_DOUBLE, rank, 1 - tag, MPI_COMM_WORLD, &rReq[1 - tag]);
}
/* block process until data received
 * sReg, rReg - send and receive request*/
void receiveBorders(MPI_Request sReq, MPI_Request rReq) {
    MPI_Wait(&rReq, MPI_STATUS_IGNORE);
    MPI_Wait(&sReq, MPI_STATUS_IGNORE);
}

void initBounds(int X, int Y, int Z, double **F, const int *offsets, int rank) {
    int startLine = offsets[rank];
    for (int i = 0; i < X; i++, startLine++) {
        for (int j = 0; j < Y; j++) {
            for (int k = 0; k < Z; k++) {
                if ((startLine != 0) && (j != 0) && (k != 0) && (startLine != NX-1) && (j != Y-1) && (k != Z-1)) {
                    F[0][i * Y * Z + j * Z + k] = 0;
                    F[1][i * Y * Z + j * Z + k] = 0;
                } else {
                    F[0][i * Y * Z + j * Z + k] = phi(startLine * hx, j * hy, k * hz);
                    F[1][i * Y * Z + j * Z + k] = F[0][i * Y * Z + j * Z + k];
                }
            }
        }
    }
}
/* return difference from original function
 * X, Y, Z - number of element form coords
 * F - calculated function
 * shift - shift form current process in F*/
double findDifference(int X, int Y, int Z, double *F, int shift) {
    double maxCount = 0;
    double max, difference;
    for (int i = 1; i < X - 2; i++) {
        for (int j = 1; j < Y; j++) {
            for (int k = 1; k < Z; k++) {
                difference = fabs(F[i * Y * Z + j * Z + k] - phi((i + shift) * hx, j * hy, k * hz));
                maxCount = std::max(maxCount, difference);
            }
        }
    }
    MPI_Allreduce(&maxCount, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return max;
}
/*void asyncSend(int pos1, int pos2, int size, int tag, double *F, double **buffer, int rank, int numOfProc, MPI_Request *sendRequest, MPI_Request *receiveRequest) {
    if (rank > 0) { //если не нулевой
        MPI_Isend(&(F[pos1]), size, MPI_DOUBLE, rank - 1, tag, MPI_COMM_WORLD, &sendRequest[0]);
        MPI_Irecv(buffer[0], size, MPI_DOUBLE, rank - 1, 1 - tag, MPI_COMM_WORLD, &receiveRequest[1]);
    }
    if (rank < numOfProc - 1) { //если не последний
        MPI_Isend(&(F[pos2]), size, MPI_DOUBLE,rank + 1,1 - tag, MPI_COMM_WORLD, &sendRequest[1]);
        MPI_Irecv(buffer[1], size, MPI_DOUBLE, rank + 1, tag, MPI_COMM_WORLD, &receiveRequest[0]);
    }
}
void receive(int rank, int numOfProc, MPI_Request *sendRequest, MPI_Request *receiveRequest) {
    if (rank > 0) { // если не нулевой
        MPI_Wait(&receiveRequest[1], MPI_STATUS_IGNORE);
        MPI_Wait(&sendRequest[0], MPI_STATUS_IGNORE);
    }
    if (rank < numOfProc - 1) { //если не последний
        MPI_Wait(&receiveRequest[0], MPI_STATUS_IGNORE);
        MPI_Wait(&sendRequest[1], MPI_STATUS_IGNORE);
    }
}
double condition(int Y, int Z, double f, int cur, double ** F, int x, int y, int k){
    double diff = fabs(F[cur][x * Y * Z + y * Z + k] - F[1 - cur][x * Y * Z + y * Z + k]);
    f = std::max(f, diff);
    if (diff > EPSILON) {
        max = diff;
    }
    return f;
}
double borderCalculate(int X, int Y, int Z, int cur, int shift, int rank, int size, double max, double **F, double **buffer) {
    double p, termX, termY, termZ;
    for (int y = 1; y < Y-1; ++y) {
        for (int z = 1; z < Z-1; ++z) {
            if (rank != 0) {
                int x = 0;
                termX = (F[1 - cur][(x + 1) * Y * Z + y * Z + z] + buffer[0][y * Z + z]) / h2X;
                termY = (F[1 - cur][x * Y * Z + (y + 1) * Z + z] + F[1 - cur][x * Y * Z + (y - 1) * Z + z]) / h2Y;
                termZ = (F[1 - cur][x * Y * Z + y * Z + (z + 1)] + F[1 - cur][x * Y * Z + y * Z + (z - 1)]) / h2Z;
                p = tail((x + shift) * hx, y * hy, z * hz);
                F[cur][x * Y * Z + y * Z + z] = factor * (termX + termY + termZ - p);

                double diff = fabs(F[cur][x * Y * Z + y * Z + z] - F[1 - cur][x * Y * Z + y * Z + z]);
                max = std::max(max, diff);
            }
            if (rank != size - 1) {
                int x = X - 1;
                termX = (buffer[1][y * Z + z] + F[1 - cur][(x - 1) * Y * Z + y * Z + z]) / h2X;
                termY = (F[1 - cur][x * Y * Z + (y + 1) * Z + z] + F[1 - cur][x * Y * Z + (y - 1) * Z + z]) / h2Y;
                termZ = (F[1 - cur][x * Y * Z + y * Z + (z + 1)] + F[1 - cur][x * Y * Z + y * Z + (z - 1)]) / h2Z;
                p = tail((x + shift) * hx, y * hy, z * hz);
                F[cur][x * Y * Z + y * Z + z] = factor * (termX + termY + termZ - p);

                double diff = fabs(F[cur][x * Y * Z + y * Z + z] - F[1 - cur][x * Y * Z + y * Z + z]);
                max = std::max(max, diff);
            }
        }
    }
    return max;
}*/
/* calculate subregion in finding function and return max differences from iteration functions
 * X, Y, Z - number of element form coords
 * shift - shift form current process in F
 * cur - number nuxt iteration in F
 * max - max differences from iteration functions
 * F - calculated functions*/
double calculateMid(int X, int Y, int Z, int shift, int cur, double max, double **F) {
    double p, termX, termY, termZ;
    for (int x = 1; x < X - 1; ++x) {
        for (int y = 1; y < Y - 1; ++y) {
            for (int z = 1; z < Z - 1; ++z) {
                termX = (F[1 - cur][(x + 1) * Y * Z + y * Z + z] + F[1 - cur][(x - 1) * Y * Z + y * Z + z]) / h2X;
                termY = (F[1 - cur][x * Y * Z + (y + 1) * Z + z] + F[1 - cur][x * Y * Z + (y - 1) * Z + z]) / h2Y;
                termZ = (F[1 - cur][x * Y * Z + y * Z + (z + 1)] + F[1 - cur][x * Y * Z + y * Z + (z - 1)]) / h2Z;
                p = tail((x + shift) * hx, y * hy, z * hz);
                F[cur][x * Y * Z + y * Z + z] = factor * (termX + termY + termZ - p);

                double diff = fabs(F[cur][x * Y * Z + y * Z + z] - F[1 - cur][x * Y * Z + y * Z + z]);
                max = std::max(max, diff);
            }
        }
    }
    return max;
}
/* calculate subregion in finding function and return max differences from iteration functions
 * Y, Z - number of element form coords
 * shift - shift form current process in F
 * cur - number nuxt iteration in F
 * max - max differences from iteration functions
 * F - calculated functions
 * buffer - buffer received left border from another process*/
double calculateLeftBorders(int Y, int Z, int cur, int shift, double max, double **F, const double *buffer) {
    double p, termX, termY, termZ;
    for (int y = 1; y < Y-1; ++y) {
        for (int z = 1; z < Z-1; ++z) {
            int x = 0;
            termX = (F[1 - cur][(x + 1) * Y * Z + y * Z + z] + buffer[y * Z + z]) / h2X;
            termY = (F[1 - cur][x * Y * Z + (y + 1) * Z + z] + F[1 - cur][x * Y * Z + (y - 1) * Z + z]) / h2Y;
            termZ = (F[1 - cur][x * Y * Z + y * Z + (z + 1)] + F[1 - cur][x * Y * Z + y * Z + (z - 1)]) / h2Z;
            p = tail((x + shift) * hx, y * hy, z * hz);
            F[cur][x * Y * Z + y * Z + z] = factor * (termX + termY + termZ - p);

            double diff = fabs(F[cur][x * Y * Z + y * Z + z] - F[1 - cur][x * Y * Z + y * Z + z]);
            max = std::max(max, diff);
        }
    }
    return max;
}
/* calculate subregion in finding function and return max differences from iteration functions
 * X, Y, Z - number of element form coords
 * shift - shift form current process in F
 * cur - number nuxt iteration in F
 * max - max differences from iteration functions
 * F - calculated functions
 * buffer - buffer received left border from another process*/
double calculateRightBorders(int X, int Y, int Z, int cur, int shift, double max, double **F, const double *buffer) {
    double p, termX, termY, termZ;
    for (int y = 1; y < Y-1; ++y) {
        for (int z = 1; z < Z-1; ++z) {
            int x = X - 1;
            termX = (buffer[y * Z + z] + F[1 - cur][(x - 1) * Y * Z + y * Z + z]) / h2X;
            termY = (F[1 - cur][x * Y * Z + (y + 1) * Z + z] + F[1 - cur][x * Y * Z + (y - 1) * Z + z]) / h2Y;
            termZ = (F[1 - cur][x * Y * Z + y * Z + (z + 1)] + F[1 - cur][x * Y * Z + y * Z + (z - 1)]) / h2Z;
            p = tail((x + shift) * hx, y * hy, z * hz);
            F[cur][x * Y * Z + y * Z + z] = factor * (termX + termY + termZ - p);

            double diff = fabs(F[cur][x * Y * Z + y * Z + z] - F[1 - cur][x * Y * Z + y * Z + z]);
            max = std::max(max, diff);
        }
    }
    return max;
}
/* realizable algorithm, return difference from original function
 * X, Y, Z - number of element form coords
 * shift - shift form current process in F
 * size - number of processes
 * rank - current process number
 * F - calculated functions*/
double jacodi(int X, int Y, int Z, int shift, int size, int rank, double **F) {
    double * borders[2];
    borders[0] = new double[Z * Y];
    borders[1] = new double[Z * Y];
    MPI_Request sReq[2], rReq[2];
    int cur = 0;
    double max, maxCount;
    do {
        maxCount = 0;
        cur = 1 - cur;
        // Send borders
        if (rank != 0) {
            sendBorders(0, Z * Y, rank - 1, F[1 - cur], 0, borders[0], sReq, rReq);
        }
        if (rank != size - 1) {
            sendBorders((X - 1) * Y * Z, Z * Y, rank + 1, F[1 - cur], 1, borders[1], sReq, rReq);
        }
        // Calculate subregion
        maxCount = calculateMid(X, Y, Z, shift, cur, maxCount, F);
        // Waiting to receive and calculate borders
        if (rank != 0) {
            receiveBorders(sReq[0], rReq[1]);
            maxCount = calculateLeftBorders(Y, Z, cur, shift, maxCount, F, borders[0]);
        }
        if (rank != size - 1) {
            receiveBorders(sReq[1], rReq[0]);
            maxCount = calculateRightBorders(X, Y, Z, cur, shift, maxCount, F, borders[1]);
        }
        // Synchronization of termination criteria between processes
        MPI_Allreduce(&maxCount, &max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }while (max >= EPSILON);
    // Calculate difference from original function
    double result = findDifference(X, Y, Z, F[cur], shift);
    delete[] borders[0];
    delete[] borders[1];

    return result;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int pSize, pRank;
    MPI_Comm_size(MPI_COMM_WORLD, &pSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &pRank);

    int shiftForIProc = 0;
    int linesPerProc[pSize], shifts[pSize];
    for (int i = 0; i < pSize; ++i) {
        shifts[i] = shiftForIProc;
        linesPerProc[i] = NX / pSize;
        if (i > pSize - (NX % pSize)) {
            linesPerProc[i]++;
        }
        shiftForIProc += linesPerProc[i];
    }
    double * F[2];
    F[0] = new double[linesPerProc[pRank] * NY * NZ];
    F[1] = new double[linesPerProc[pRank] * NY * NZ];
    initBounds(linesPerProc[pRank], NY, NZ, F, shifts, pRank);

    double time = -MPI_Wtime();
    double result = jacodi(linesPerProc[pRank], NY, NZ, shifts[pRank], pSize, pRank, F);
    time += MPI_Wtime();

    if (pRank == 0) {
        std::cout << "Max difference: " << result << std::endl;
        std::cout << "Time: " << time << std::endl;
    }

    delete[] F[0];
    delete[] F[1];
    MPI_Finalize();
    return 0;
}
