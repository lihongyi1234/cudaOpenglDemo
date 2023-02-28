#pragma once
#include "cuda_opengl_demo.h"
#include <vector_types.h>

//clamp x to range [a, b]
__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

__device__ int clamp(int x, int a, int b) { return max(a, min(b, x)); }

__device__ int rgbToInt(float r, float g, float b) {
	r = clamp(r, 0.0f, 255.0f);
	g = clamp(g, 0.0f, 255.0f);
	b = clamp(b, 0.0f, 255.0f);
	return (int(b) << 16) | (int(g) << 8) | int(r);
}

__global__ void cudaProcess(unsigned int* g_data, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;
	extern __shared__ uchar4 sdata[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int _x = blockIdx.x * bw + tx;
	int _y = blockIdx.y * bh + ty;

	uchar4 c4 = make_uchar4((_x & 0x20) ? 100 : 0, 0, (_y & 0x20) ? 100 : 0, 0);
	g_data[y * width + x] = rgbToInt(c4.z, c4.y, c4.x);
}

void launch_cudaProcess(unsigned int* g_data, int width, int height)
{
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	cudaProcess << <numBlocks, threadsPerBlock >> > (g_data, width, height);

}
