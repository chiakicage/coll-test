# Collective Test

Implement two collective operations: `Reduce` and `Broadcast` in a single node

```c++
// Single Process Version
int intraReduce(int *const sendbuff[], int* recvbuff, size_t count, int root,
                ncclComm_t *comms, cudaStream_t *streams);
int intraBroadcast(int* sendbuff, int *const recvbuff[], size_t count, int root,
                   ncclComm_t *comms, cudaStream_t *streams);
// One Device per Process Version
int intraReduce(const int* sendbuff, int* recvbuff, size_t count, int root,
                ncclComm_t comm, cudaStream_t stream);
int intraBroadcast(const int* sendbuff, int* recvbuff, size_t count, int root,
                   ncclComm_t comm, cudaStream_t stream);
```
## Build

To build the tests, just type `make`.

If CUDA is not installed in /usr/local/cuda, you may specify CUDA_HOME. Similarly, if NCCL is not installed in /usr, you may specify NCCL_HOME.

By now there is a MPI version, if MPI is not installed in /usr/local/mpi, you may specify MPI_HOME.

```bash
$ make CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl MPI_HOME=/path/to/mpi
```

## Run

Run on a single node with 4 GPUs using 1 process.

```bash
$ ./build/intra
```

Run on a single node with 4 GPUs using 4 processes.

```bash
$ mpirun -np 4 ./build/intra_mpi
```



