#include <mpi.h>
#include <nccl.h>

#include <iostream>

const int BUFFER_SIZE = 10;

int intraReduce(const int* sendbuff, int* recvbuff, size_t count, int root,
                ncclComm_t comm, cudaStream_t stream) {
  ncclReduce(sendbuff, recvbuff, count, ncclInt, ncclSum, root, comm, stream);
  cudaStreamSynchronize(stream);
  return 0;
}

int intraBroadcast(const int* sendbuff, int* recvbuff, size_t count, int root,
                   ncclComm_t comm, cudaStream_t stream) {
  ncclBroadcast(sendbuff, recvbuff, count, ncclInt, root, comm, stream);
  cudaStreamSynchronize(stream);
  return 0;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int count;
  cudaGetDeviceCount(&count);
  if (size != count) {
    std::cout << "size != nDev" << std::endl;
    return 0;
  }

  ncclUniqueId id;
  ncclComm_t comm;
	int *data, *output, *buffer;
  cudaStream_t s;

  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  std::cout << "rank: " << rank << std::endl;

  cudaSetDevice(rank);
  cudaStreamCreate(&s);
  cudaMallocManaged(&buffer, BUFFER_SIZE * sizeof(int));
  cudaMemset(buffer, 0, BUFFER_SIZE * sizeof(int));

  if (rank == 0) {
    cudaMallocManaged(&data, BUFFER_SIZE * sizeof(int));
    cudaMemset(data, 0, BUFFER_SIZE * sizeof(int));
    cudaMallocManaged(&output, BUFFER_SIZE * sizeof(int));
    cudaMemset(output, 0, BUFFER_SIZE * sizeof(int));

    for (int i = 0; i < BUFFER_SIZE; i++) {
      data[i] = i;
    }
  }

  ncclCommInitRank(&comm, size, id, rank);

  intraBroadcast(data, buffer, BUFFER_SIZE, 0, comm, s);

  intraReduce(buffer, output, BUFFER_SIZE, 0, comm, s);


  cudaStreamSynchronize(s);

  if (rank == 0) {
    for (int i = 0; i < BUFFER_SIZE; i++) {
      std::cout << output[i] << std::endl;
    }
  }

  cudaStreamDestroy(s);

  ncclCommDestroy(comm);

  MPI_Finalize();

  return 0;
}
