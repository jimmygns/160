#include <gputk.h>

#define BLOCK_SIZE 16

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this lab
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  float Cvalue = 0;

//     // i is the index of the block
//   if (row >= numCRows || col >= numCColumns) {
//     return;
//   }
//   for (int i = 0; i < ceil((1.0*numAColumns)/BLOCK_SIZE); i++) {
//     if (row < numARows && i*BLOCK_SIZE + threadIdx.x < numAColumns) {
//       As[threadIdx.y][threadIdx.x] = A[row*numAColumns + i*BLOCK_SIZE + threadIdx.x];
//     } else {
//       As[threadIdx.y][threadIdx.x] = 0;
//     }
//     if (col < numBColumns && i*BLOCK_SIZE + threadIdx.y < numBRows) {
//       Bs[threadIdx.y][threadIdx.x] = B[(i*BLOCK_SIZE+threadIdx.y)*numBColumns + col];
//     } else {
//       Bs[threadIdx.y][threadIdx.x] = 0;
//     }
//     __syncthreads();
//     for (int j = 0; j < BLOCK_SIZE; j++) {
//       Cvalue += As[threadIdx.y][j] * Bs[j][threadIdx.x];
//     }
//     __syncthreads();
//   }
//   C[row*numCColumns + col] = Cvalue;
// }


  for (int i = 0; i < ceil((1.0*numAColumns)/BLOCK_SIZE); i++) {
    if (row < numARows && i * BLOCK_SIZE + threadIdx.x < numAColumns) {
      As[threadIdx.y][threadIdx.x] = A[row * numAColumns + i * BLOCK_SIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0;
    }
    if (i * BLOCK_SIZE + threadIdx.y < numBRows && col < numBColumns) {
      Bs[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * numBColumns + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();
    for (int j = 0; j < BLOCK_SIZE; j++) {
      Cvalue += As[threadIdx.y][j] * Bs[j][threadIdx.x];
    }
    __syncthreads();
  }
  if (row >= numCRows || col >= numCColumns) {
    return;
  }

  C[row * numCColumns + col] = Cvalue;
}


int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(float));

  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),
             cudaMemcpyHostToDevice);

  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil((1.0*numCColumns)/BLOCK_SIZE), ceil((1.0*numCRows)/BLOCK_SIZE), 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float),
             cudaMemcpyDeviceToHost);

  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
