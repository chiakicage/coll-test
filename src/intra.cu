#include <mpi.h>
#include <nccl.h>

#include <iostream>

const int BUFFER_SIZE = 10;
const int nDev = 4;

int intraReduce(int *const sendbuff[], int* recvbuff, size_t count, int root,
                ncclComm_t *comms, cudaStream_t *streams) {
  ncclGroupStart();
  for (int i = 0; i < nDev; i++) {
    ncclReduce(sendbuff[i], recvbuff, count, ncclInt, ncclSum, root, comms[i], streams[i]);
  }
  for (int i = 0; i < nDev; i++) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
  }
  ncclGroupEnd();
  return 0;
}

int intraBroadcast(int* sendbuff, int *const recvbuff[], size_t count, int root,
                   ncclComm_t *comms, cudaStream_t *streams) {
  ncclGroupStart();
  for (int i = 0; i < nDev; i++) {
    ncclBroadcast(sendbuff, recvbuff[i], count, ncclInt, root, comms[i], streams[i]);
  }
  for (int i = 0; i < nDev; i++) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
  }
  ncclGroupEnd();
  return 0;
}

int main(int argc, char* argv[]) {
  


  ncclComm_t comms[nDev];
  int *data, *output, *buffer[nDev];
  cudaStream_t s[nDev];

  
  for (int i = 0; i < nDev; i++) {
    cudaSetDevice(i);
    cudaStreamCreate(&s[i]);
    cudaMallocManaged(&buffer[i], BUFFER_SIZE * sizeof(int));
    cudaMemset(buffer[i], 0, BUFFER_SIZE * sizeof(int));
  }
  cudaSetDevice(0);
  cudaMallocManaged(&data, BUFFER_SIZE * sizeof(int));
  cudaMemset(data, 0, BUFFER_SIZE * sizeof(int));
  cudaMallocManaged(&output, BUFFER_SIZE * sizeof(int));
  cudaMemset(output, 0, BUFFER_SIZE * sizeof(int));

  for (int i = 0; i < BUFFER_SIZE; i++) {
    data[i] = i;
  }
 

  ncclCommInitAll(comms, nDev, NULL);


  intraBroadcast(data, buffer, BUFFER_SIZE, 0, comms, s);

  intraReduce(buffer, output, BUFFER_SIZE, 0, comms, s);

  for (int i = 0; i < BUFFER_SIZE; i++) {
    std::cout << output[i] << " ";
  }
  std::cout << std::endl;

  cudaSetDevice(0);
  cudaFree(data);
  cudaFree(output);
  for (int i = 0; i < nDev; i++) {
    cudaSetDevice(i);
    cudaStreamDestroy(s[i]);
    cudaFree(buffer[i]);
    ncclCommDestroy(comms[i]);
  }


  return 0;
}
