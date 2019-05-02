#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 1024

unsigned int getmax(unsigned int *, unsigned int);
unsigned int getmaxcu(unsigned int *, unsigned int);

void printDeviceInfo()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; i++)
    {
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
} // End of the print device info function




__global__ void getmaxCUDA1(unsigned int * num_d, int new_size, unsigned int * block_result) {
  __shared__ unsigned int local_num[THREADS_PER_BLOCK];
  int tid = threadIdx.x;
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  //Grab the array from the global memory to shared memory.(In one block)
   local_num[tid] = 0;
   if(thread_id < new_size) {
    local_num[tid] = num_d[thread_id];
   }
   __syncthreads();

  for(int i = (blockDim.x)/2; i >= 1 ; i = i / 2) {
    int stride = i;
    if(tid < stride && (tid + stride) < blockDim.x) {
      if(local_num[tid] < local_num[tid+stride]) {
        local_num[tid] = local_num[tid+stride];
      }
    }

    __syncthreads();
  }

  //Get the block max.
  if(tid == 0) {
    block_result[blockIdx.x] = local_num[0];
  }

}



int main(int argc, char *argv[])
{
    unsigned int size = 0;  // The size of the array
    unsigned int i;  // loop index
    unsigned int * numbers; //pointer to the array

    if(argc !=2)
    {
       printf("usage: maxseq num\n");
       printf("num = size of the array\n");
       exit(1);
    }

    size = atol(argv[1]);

    numbers = (unsigned int *)malloc(size * sizeof(unsigned int));
    if( !numbers )
    {
       printf("Unable to allocate mem for an array of size %u\n", size);
       exit(1);
    }

    printDeviceInfo();


   srand(time(NULL)); // setting a seed for the random number generator
    // Fill-up the array with random numbers from 0 to size-1
    for( i = 0; i < size; i++){
       numbers[i] = rand()  % size;
//       printf("number[%d] is %d\n",i, numbers[i]);
    }
    printf(" The maximum number in the array is: %u\n",
           getmaxcu(numbers, size));
    printf(" The maximum number by sequential  in the array is: %u\n",
           getmax(numbers, size));
    free(numbers);
    exit(0);
}


/*
   input: pointer to an array of long int
          number of elements in the array
   output: the maximum number of the array
*/
unsigned int getmaxcu(unsigned int num[], unsigned int size)
{
  unsigned int max = num[0];
 // printf("max[5] is %d max [6] is %d max[1] is %d\n",num[5], num[6], num[1]);

  unsigned int* num_d;
  unsigned int* block_result;
  unsigned int* result;
  unsigned int block_num = size / THREADS_PER_BLOCK;
  if(size % THREADS_PER_BLOCK != 0) {
    block_num = block_num + 1;
  }
  unsigned int block_count = block_num;
  int sizen = size*sizeof(unsigned int);
  int sizeb = block_num*sizeof(unsigned int);
  int new_size = size;

  //1.Transfer num[] to device memory.
  cudaMalloc((void**)&num_d, sizen);
  cudaMemcpy(num_d, num, sizen, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&block_result, sizeb);

  //2.Kernel invocation code.
  while(block_num >= 1) {
    if(block_num == 1) {
      getmaxCUDA1<<<block_num, THREADS_PER_BLOCK>>>(num_d, new_size, block_result);
      block_num--;
    } else {
      getmaxCUDA1<<<block_num, THREADS_PER_BLOCK>>>(num_d, new_size, block_result);
      new_size = block_num;
      if(new_size % THREADS_PER_BLOCK != 0) {
        block_num = new_size / THREADS_PER_BLOCK + 1;
      } else {block_num = new_size / THREADS_PER_BLOCK;}
    }
    num_d = block_result;
  }
  result = (unsigned int*) malloc(block_count * sizeof(unsigned int));
  cudaMemcpy(result, block_result, sizeb, cudaMemcpyDeviceToHost);


  max = result[0];
  //3. Free device memory for num[].
  cudaFree(num_d);
  cudaFree(block_result);


  return( max );

}

unsigned int getmax(unsigned int num[], unsigned int size)
{
  unsigned int i;
  unsigned int max = num[0];

  for(i = 1; i < size; i++)
	if(num[i] > max)
	   max = num[i];

  return( max );

}
