# Collective Test

Implement two collective operations: `Reduce` and `Broadcast` in a single node

```bash
int intraReduce(const int* sendbuff, int* recvbuff, size_t count, int root, ncclComm_t comm, cudaStream_t stream);
int intraBroadcast(const int* sendbuff, int* recvbuff, size_t count, int root, ncclComm_t comm, cudaStream_t stream);
```
## Build

To build the tests, just type `make`.

If CUDA is not installed in /usr/local/cuda, you may specify CUDA_HOME. Similarly, if NCCL is not installed in /usr, you may specify NCCL_HOME.

By now the tests rely on MPI, and if MPI is not installed in /usr/local/mpi, you may specify MPI_HOME.

```bash
$ make CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl MPI_HOME=/path/to/mpi
```

## Run

Run on a single node with 4 GPUs

```bash
$ mpirun -np 4 ./build/intra
```

