#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#include <mpi.h>

/// constant parameters
#define NUMBER_OF_THREADS 2
#define FIRST_THREAD 0
#define SECOND_THREAD 1
#define TEG_RANK 0
#define TEG_TASK 1
#define STOP_THREAD -1

/// dynamic parameters
#define NUMBER_OF_LISTS 4
#define ALL_NUMBER_OF_TASKS 128
#define BORDER_OF_TASKS_TO_SHARE 2
#define U_SECONDS 300000

struct Task {
    int *tasks;
    int numberOfLeftTasks;
    int numberOfExecutedTasks;
};

struct Task taskList;
pthread_mutex_t mutex;

void fillTasks(int numberOfTasks, int rank, int size, int iterCounter) {
    for (int i = 0; i < numberOfTasks; i++) {
        taskList.tasks[i] = abs(rank - (iterCounter % size));
    }
}

void executeTasks() {
    int i = 0;
    while (taskList.numberOfLeftTasks > 0) {
        for (int j = 0; j < taskList.tasks[i]; j++) {
            usleep(U_SECONDS);
        }

        taskList.numberOfExecutedTasks++;
        i++;

        pthread_mutex_lock(&mutex);

        taskList.numberOfLeftTasks--;

        pthread_mutex_unlock(&mutex);
    }
}

void* receiver(void* args) {
    int numberOfSharedTasks;
    int otherRank;
    MPI_Status status;

    while (true) {
        MPI_Recv(&otherRank, 1, MPI_INT, MPI_ANY_SOURCE, TEG_RANK, MPI_COMM_WORLD, &status);

        if (otherRank == STOP_THREAD) pthread_exit(NULL);

        if (taskList.numberOfLeftTasks > BORDER_OF_TASKS_TO_SHARE) {
            pthread_mutex_lock(&mutex);

            numberOfSharedTasks = taskList.numberOfLeftTasks / 2;
            taskList.numberOfLeftTasks -= numberOfSharedTasks;

            pthread_mutex_unlock(&mutex);

            MPI_Send(&numberOfSharedTasks, 1, MPI_INT, status.MPI_SOURCE, TEG_TASK, MPI_COMM_WORLD);
            MPI_Send(&taskList.tasks[numberOfSharedTasks - 1], numberOfSharedTasks, MPI_INT, status.MPI_SOURCE, TEG_TASK, MPI_COMM_WORLD);
        }
        else {
            numberOfSharedTasks = 0;
            MPI_Send(&numberOfSharedTasks, 1, MPI_INT, status.MPI_SOURCE, TEG_TASK, MPI_COMM_WORLD);
        }
    }

    pthread_exit(NULL);
}

int distributeTasks(int size, int rank) {
    int res = ALL_NUMBER_OF_TASKS / size;
    if (ALL_NUMBER_OF_TASKS % size > 0 && ALL_NUMBER_OF_TASKS % size - 1 > rank) res++;

    return res;
}

void executeTaskList(int size, int rank, int numberOfTasks) {
    MPI_Status status;
    int numberOfAdditionalTasks;

    for (int i = 0; i < NUMBER_OF_LISTS; i++) {
        fillTasks(numberOfTasks, rank, size, i);

        taskList.numberOfLeftTasks = numberOfTasks;
        taskList.numberOfExecutedTasks = 0;
        numberOfAdditionalTasks = 0;

        executeTasks();

        for (int j = 0; j < size; j++) {
            if (j == rank) continue;

            MPI_Send(&rank, 1, MPI_INT, j, TEG_RANK, MPI_COMM_WORLD);
            MPI_Recv(&numberOfAdditionalTasks, 1, MPI_INT, j, TEG_TASK, MPI_COMM_WORLD, &status);

            if (numberOfAdditionalTasks > 0) {
                MPI_Recv(taskList.tasks, numberOfAdditionalTasks, MPI_INT, j, TEG_TASK, MPI_COMM_WORLD, &status);
                taskList.numberOfLeftTasks = numberOfAdditionalTasks;
                executeTasks();
                printf("%d executed add tasks - %d\n", rank, numberOfAdditionalTasks);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        printf("%d executed tasks - %d\n", rank, taskList.numberOfExecutedTasks);
    }
}

void* executor(void* args) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int numberOfTasks = distributeTasks(size, rank);

    taskList.tasks = (int*)malloc(sizeof(int) * numberOfTasks);

    executeTaskList(size, rank, numberOfTasks);

    int stopThreadFlag = STOP_THREAD;
    MPI_Send(&stopThreadFlag, 1, MPI_INT, rank, TEG_RANK, MPI_COMM_WORLD);

    free(taskList.tasks);
    pthread_exit(NULL);
}

void createAndStartThreads() {
    pthread_mutex_init(&mutex, NULL);

    pthread_t threads[NUMBER_OF_THREADS];
    pthread_attr_t attrs;

    pthread_attr_init(&attrs);
    /**
     * PTHREAD_CREATE_JOINABLE - we want to use thread one time and before his work join him and clean memory
     * PTHREAD_CREATE_DETACHED - we want to use thread several times and reuse his memory
     */
    pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE);

    ///create and start two threads
    pthread_create(&threads[FIRST_THREAD], &attrs, receiver, NULL);
    pthread_create(&threads[SECOND_THREAD], &attrs, executor, NULL);

    /// delete memory for attribute
    pthread_attr_destroy(&attrs);

    ///wait threads when they end those job and to delete memory this threads
    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);

    pthread_mutex_destroy(&mutex);
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

    createAndStartThreads();

    time += MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("%f sec\n", time);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
