#include <cuda.h>
#include <cuda_runtime.h>
#ifndef __CUDACC__
#define __CUDACC__
#include <device_functions.h>
#endif
#include <device_launch_parameters.h>

#include <iostream>

const int BUFFER_SIZE = 10;

__global__ void cudaReduce(int *const recvbuffs[], int *recvbuff, size_t count,
                           int root, int ndev) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    int sum = 0;
    for (int i = 0; i < ndev; i++) {
      sum += recvbuffs[i][index];
    }
    recvbuff[index] = sum;
  }
  __syncthreads();
}

int intraReduce(int *const sendbuff[], int *recvbuff, size_t count, int root,
                int *devs, int ndev) {
  cudaStream_t streams[ndev];
  for (int i = 0; i < ndev; i++) {
    cudaSetDevice(devs[i]);
    cudaStreamCreate(&streams[i]);
  }
  int **recvbuffs;
  cudaMallocManaged(&recvbuffs, ndev * sizeof(int *));
  cudaSetDevice(devs[root]);
  for (int i = 0; i < ndev; i++) {
    cudaMallocManaged(&recvbuffs[i], count * sizeof(int));
  }
  for (int i = 0; i < ndev; i++) {
    if (i != root) {
      cudaMemcpyPeerAsync(recvbuffs[i], root, sendbuff[i], devs[i],
                          count * sizeof(int), streams[i]);
    } else {
      cudaMemcpyAsync(recvbuffs[i], sendbuff[i], count * sizeof(int),
                      cudaMemcpyDeviceToDevice, streams[i]);
    }
  }
  for (int i = 0; i < ndev; i++) {
    cudaSetDevice(devs[i]);
    cudaStreamSynchronize(streams[i]);
  }
  cudaDeviceSynchronize();
  // for (int i = 0; i < ndev; i++) {
  //   for (int j = 0; j < count; j++) {
  //     std::cout << recvbuffs[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;
  cudaSetDevice(devs[root]);
  int blockSize = 256;
  int numBlocks = (count + blockSize - 1) / blockSize;
  cudaReduce<<<numBlocks, blockSize>>>(recvbuffs, recvbuff, count, root, ndev);
  cudaDeviceSynchronize();
  for (int i = 0; i < ndev; i++) {
    if (i != root) {
      cudaFree(recvbuffs[i]);
    }
  }
  for (int i = 0; i < ndev; i++) {
    cudaSetDevice(devs[i]);
    cudaStreamDestroy(streams[i]);
  }
  return 0;
}

int intraBroadcast(int *sendbuff, int *const recvbuff[], size_t count, int root,
                   int *devs, int ndev) {
  cudaStream_t streams[ndev];
  for (int i = 0; i < ndev; i++) {
    cudaSetDevice(devs[i]);
    cudaStreamCreate(&streams[i]);
  }
  for (int i = 0; i < ndev; i++) {
    cudaMemcpyPeerAsync(recvbuff[i], devs[i], sendbuff, root,
                        count * sizeof(int), streams[i]);
  }
  for (int i = 0; i < ndev; i++) {
    cudaSetDevice(devs[i]);
    cudaStreamSynchronize(streams[i]);
  }
  cudaDeviceSynchronize();
  for (int i = 0; i < ndev; i++) {
    cudaSetDevice(devs[i]);
    cudaStreamDestroy(streams[i]);
  }
  return 0;
}

int main(int argc, char *argv[]) {
  int ndev;
  cudaGetDeviceCount(&ndev);

  int *data, *output, *buffer[ndev], devs[ndev];
  for (int i = 0; i < ndev; i++) {
    devs[i] = i;
  }

  for (int i = 0; i < ndev; i++) {
    cudaSetDevice(i);
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

  intraBroadcast(data, buffer, BUFFER_SIZE, 0, devs, ndev);

  // for (int i = 0; i < ndev; i++) {
  //   for (int j = 0; j < BUFFER_SIZE; j++) {
  //     std::cout << buffer[i][j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  intraReduce(buffer, output, BUFFER_SIZE, 0, devs, ndev);

  for (int i = 0; i < BUFFER_SIZE; i++) {
    std::cout << output[i] << " ";
  }
  std::cout << std::endl;

  cudaSetDevice(0);
  cudaFree(data);
  cudaFree(output);
  for (int i = 0; i < ndev; i++) {
    cudaSetDevice(i);
    cudaFree(buffer[i]);
  }

  return 0;
}
