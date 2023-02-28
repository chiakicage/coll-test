#include <nccl.h>
#include <iostream>
#include <mpi.h>

const int BUFFER_SIZE = 10;
const int nDev = 4;


int intraReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t* comms) {
	cudaStream_t streams[nDev];
	for (int i = 0; i < nDev; i++) {
		cudaSetDevice(i);
		cudaStreamCreate(&streams[i]);
	}

	ncclGroupStart();
	ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
	ncclGroupEnd();
	return 0;
}

int intraBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
	ncclGroupStart();
	ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
	ncclGroupEnd();
	return 0;
}

int main(int argc, char* argv[]) {
	int nDev;
	cudaGetDeviceCount(&nDev);

	

	ncclComm_t comms[nDev];
	ncclCommInitAll(comms, nDev, NULL);

	int* sendbuf[nDev];
	int* recvbuf[nDev];

	for (int i = 0; i < nDev; i++) {
		cudaSetDevice(i);
		cudaMallocManaged(&sendbuf[i], BUFFER_SIZE * sizeof(int));
		// cudaMemset(sendbuf[i], 1, BUFFER_SIZE * sizeof(int));
		for (int j = 0; j < BUFFER_SIZE; j++) {
			sendbuf[i][j] = j;
		}
		cudaMallocManaged(&recvbuf[i], BUFFER_SIZE * sizeof(int));
		cudaMemset(recvbuf[i], 0, BUFFER_SIZE * sizeof(int));
	}

	// ncclCommFinalize(comm);
	// ncclCommDestroy(comm);
	for (int i = 0; i < nDev; i++) {
		cudaSetDevice(i);
		cudaStreamSynchronize(streams[i]);
	}

	int sum = 0;
	cudaSetDevice(1);
	for (int i = 0; i < BUFFER_SIZE; i++) {
		// sum += recvbuf[0][i];
		std::cout << recvbuf[1][i] << std::endl;
	}
	std::cout << "sum: " << sum << std::endl;
	std::cout << "done" << std::endl;

	for (int i = 0; i < nDev; i++) {
		cudaSetDevice(i);
		cudaFree(sendbuf[i]);
		cudaFree(recvbuf[i]);
	}

	for (int i = 0; i < nDev; i++) {
		ncclCommDestroy(comms[i]);
	}

	return 0;
}
