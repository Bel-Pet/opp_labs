#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#include <mpi.h>

/// constant parameters
#define OPERATION_SUCCESS 0
#define NUMBER_OF_THREADS 2
#define FIRST_THREAD 0
#define SECOND_THREAD 1
#define TEG_RANK 0
#define TEG_TASK 1
#define STOP_THREAD -1

/// dynamic parameters
#define NUMBER_OF_LISTS 4
#define ALL_NUMBER_OF_TASKS 100
#define BORDER_OF_TASKS_TO_SHARE 2
#define U_SECONDS 300000

typedef struct ArgsTag {
    int *tasks;
    int numberOfLeftTasks;
    int numberOfExecutedTasks;
    pthread_mutex_t mutex;
} Args;

/// filling list of tasks with weights
void fillTasks(Args *arg, int numberOfTasks, int rank, int size, int iterCounter) {
    for (int i = 0; i < numberOfTasks; i++) {
        arg->tasks[i] = abs(rank - (iterCounter % size));
    }
}

/// task list execution
int executeTasks(Args *arg) {
    int result = 0;
    int i = 0;
    while (arg->numberOfLeftTasks > 0) {
        for (int j = 0; j < arg->tasks[i]; j++) {
            result++;
            usleep(U_SECONDS);
        }

        arg->numberOfExecutedTasks++;
        i++;

        pthread_mutex_lock(&arg->mutex);
        arg->numberOfLeftTasks--;
        pthread_mutex_unlock(&arg->mutex);
    }

    return result;
}

/// distribution of list of tasks by processes
int distributeTasks(int size, int rank) {
    int res = ALL_NUMBER_OF_TASKS / size;
    if (ALL_NUMBER_OF_TASKS % size > 0 && ALL_NUMBER_OF_TASKS % size - 1 > rank) res++;

    return res;
}

/// task lists execution and receiving part of task list from other process
int executeTaskList(Args *arg, int numberOfTasks, int size, int rank) {
    MPI_Status status;
    int numberOfAdditionalTasks;
    int result = 0;

    for (int i = 0; i < NUMBER_OF_LISTS; i++) {
        fillTasks(arg, numberOfTasks, rank, size, i);

        arg->numberOfLeftTasks = numberOfTasks;
        arg->numberOfExecutedTasks = 0;
        numberOfAdditionalTasks = 0;

        result += executeTasks(arg);

        for (int j = 0; j < size; j++) {
            if (j == rank) continue;

            MPI_Send(&rank, 1, MPI_INT, j, TEG_RANK, MPI_COMM_WORLD);
            MPI_Recv(&numberOfAdditionalTasks, 1, MPI_INT, j, TEG_TASK, MPI_COMM_WORLD, &status);

            if (numberOfAdditionalTasks > 0) {
                MPI_Recv(arg->tasks, numberOfAdditionalTasks, MPI_INT, j, TEG_TASK, MPI_COMM_WORLD, &status);
                arg->numberOfLeftTasks = numberOfAdditionalTasks;
                result += executeTasks(arg);
                printf("%d executed add tasks - %d\n", rank, numberOfAdditionalTasks);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        printf("%d executed tasks - %d\n", rank, arg->numberOfExecutedTasks);
    }

    return result;
}

/// thread executing tasks
void* executor(void* args) {
    Args *arg = (Args*) args;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int numberOfTasks = distributeTasks(size, rank);

    arg->tasks = (int*)malloc(sizeof(int) * numberOfTasks);
    if (arg->tasks == NULL) {
        perror("malloc");
        pthread_exit((void *) EXIT_FAILURE);
    }

    int result = executeTaskList(arg, numberOfTasks, size, rank);

    int stopThreadFlag = STOP_THREAD;
    MPI_Send(&stopThreadFlag, 1, MPI_INT, rank, TEG_RANK, MPI_COMM_WORLD);

    free(arg->tasks);

    int globalResult;
    MPI_Allreduce(&result, &globalResult, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    printf("local: %d\n", result);
    if (rank == 0) printf("Result: %d\n", globalResult);

    pthread_exit( EXIT_SUCCESS);
}

/// thread sending tasks
void* receiver(void* args) {
    Args *arg = (Args*) args;

    int numberOfSharedTasks;
    int otherRank;
    MPI_Status status;

    while (true) {
        MPI_Recv(&otherRank, 1, MPI_INT, MPI_ANY_SOURCE, TEG_RANK, MPI_COMM_WORLD, &status);

        if (otherRank == STOP_THREAD) break;

        if (arg->numberOfLeftTasks > BORDER_OF_TASKS_TO_SHARE) {
            pthread_mutex_lock(&arg->mutex);
            numberOfSharedTasks = arg->numberOfLeftTasks / 2;
            arg->numberOfLeftTasks -= numberOfSharedTasks;
            pthread_mutex_unlock(&arg->mutex);

            MPI_Send(&numberOfSharedTasks, 1, MPI_INT, status.MPI_SOURCE, TEG_TASK, MPI_COMM_WORLD);
            MPI_Send(&arg->tasks[numberOfSharedTasks - 1], numberOfSharedTasks, MPI_INT, status.MPI_SOURCE, TEG_TASK, MPI_COMM_WORLD);
        }
        else {
            numberOfSharedTasks = 0;
            MPI_Send(&numberOfSharedTasks, 1, MPI_INT, status.MPI_SOURCE, TEG_TASK, MPI_COMM_WORLD);
        }
    }

    pthread_exit(NULL);
}

int createAndStartThreads() {
    Args someArgs;

    if (pthread_mutex_init(&someArgs.mutex, NULL) != OPERATION_SUCCESS) {
        perror("pthread_mutex_init");
        abort();
    }

    pthread_t threads[NUMBER_OF_THREADS];
    pthread_attr_t attrs;

    if (pthread_attr_init(&attrs) != OPERATION_SUCCESS) {
        perror("pthread_attr_init");
        abort();
    }
    if (pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE) != OPERATION_SUCCESS) {
        perror("pthread_attr_setdetachstate");
        abort();
    }
    if (pthread_create(&threads[FIRST_THREAD], &attrs, receiver, (void*) &someArgs) != OPERATION_SUCCESS) {
        perror("pthread_create");
        abort();
    }
    if (pthread_create(&threads[SECOND_THREAD], &attrs, executor, (void*) &someArgs) != OPERATION_SUCCESS) {
        perror("pthread_create");
        abort();
    }

    pthread_attr_destroy(&attrs);

    int status;
    if (pthread_join(threads[FIRST_THREAD], NULL) != OPERATION_SUCCESS) {
        perror("pthread_join");
        abort();
    }
    if (pthread_join(threads[SECOND_THREAD], (void**)&status) != OPERATION_SUCCESS) {
        perror("pthread_join");
        abort();
    }
    if (status != EXIT_SUCCESS) abort();

    if (pthread_mutex_destroy(&someArgs.mutex) != OPERATION_SUCCESS) {
        perror("pthread_mutex_destroy");
        abort();
    }
}

int main(int argc, char** argv) {
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double time = -MPI_Wtime();

    int result = createAndStartThreads();

    time += MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Time: %f sec", time);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
