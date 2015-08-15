#include "DivideBlock3D.h"
#include "Parameter.h"
#include "ReadSaveImage.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
* 功能：从图像中提取块
* 输入：im_D 待提取块的图像
* 输出：blocks_D 提取的块，一行为一个块
* 输入：row 输入图像行数
* 输入：col 输入图像列数
*/
__global__ void DivideBlock3D_raw(float *im_D, float *blocks_D, int row, int col)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标
	int index = y_id * col + x_id;

	__shared__ float sData[BLOCKSIZE + 4][BLOCKSIZE + 4];

	for (int k = 0; k < batch; k++)
	{
		float *dataPtr = &im_D[k * row * col];

		/* 左上角 16 * 16 */
		if (x_id < col && y_id < row)
			sData[threadIdx.y][threadIdx.x] = dataPtr[index];

		if (blockDim.x != gridDim.x && blockDim.y != gridDim.y)
		{
			/* 右下角 4 * 4 */
			if (threadIdx.y >= 12 && threadIdx.x >= 12)
				sData[threadIdx.y + 4][threadIdx.x + 4] = dataPtr[index + 4 * col + 4];

			/* 右上角 16 * 4 */
			if (threadIdx.x >= 12)
				sData[threadIdx.y][threadIdx.x + 4] = dataPtr[index + 4];

			/* 左下角 4 * 16 */
			if (threadIdx.y >= 12)
				sData[threadIdx.y + 4][threadIdx.x] = dataPtr[index + 4 * col];
		}

		__syncthreads();

		if (x_id < col - 4 && y_id < row - 4)
		{
			int indexOffset = (y_id * (col - blockR + 1) + x_id) * blockSizes + k * blockR * blockR;

			for (int i = 0; i < blockR; i++)
			{
				int indexRow = i * blockR;
				for (int j = 0; j < blockR; j++)
				{
					blocks_D[indexOffset + indexRow + j] = sData[threadIdx.y + i][threadIdx.x + j];
				}
			}
		}

		__syncthreads();
	}
}


/*
* 功能：从图像中提取块
* 输入：im_D 待提取块的图像
* 输出：blocks_D 提取的块，一行为一个块
* 输入：row 输入图像行数
* 输入：col 输入图像列数
*/
__global__ void DivideBlock3D(float *im_D, float *blocks_D, int row, int col)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标
	int index = y_id * col + x_id;

	__shared__ float sData[BLOCKSIZE + (blockR - 1)][BLOCKSIZE + (blockR - 1)];


	for (int k = 0; k < batch; k++)
	{
		float *dataPtr = &im_D[k * row * col];

		/* 左上角 16 * 16 */
		if (x_id < col && y_id < row) 
			sData[threadIdx.y][threadIdx.x] = dataPtr[index];

		if (blockDim.x != gridDim.x && blockDim.y != gridDim.y)
		{
			/* 右下角 4 * 4 */
			if (threadIdx.y >= BLOCKSIZE - (blockR - 1) && threadIdx.x >= BLOCKSIZE - (blockR - 1))
				sData[threadIdx.y + (blockR - 1)][threadIdx.x + (blockR - 1)] = dataPtr[index + (blockR - 1) * col + (blockR - 1)];

			/* 右上角 16 * 4 */
			if (threadIdx.x >= BLOCKSIZE - (blockR - 1))
				sData[threadIdx.y][threadIdx.x + (blockR - 1)] = dataPtr[index + (blockR - 1)];

			/* 左下角 4 * 16 */
			if (threadIdx.y >= BLOCKSIZE - (blockR - 1))
				sData[threadIdx.y + (blockR - 1)][threadIdx.x] = dataPtr[index + (blockR - 1) * col];
		}

		__syncthreads();

		if (x_id < col - blockR + 1 && y_id < row - blockR + 1)
		{
#if majorRow
			int indexOffset = (y_id * (col - blockR + 1) + x_id) * blockSizes + k * blockR * blockR; // 取块按行优先，(0,1)开始的是第2块
#else
			int indexOffset = (x_id * (row - blockR + 1) + y_id) * blockSizes + k * blockR * blockR; // 取块按列优先，(1,0)开始的是第2块
#endif

			for (int i = 0; i < blockR; i++)
			{
				int indexRow = i * blockR;
				for (int j = 0; j < blockR; j++)
				{
					blocks_D[indexOffset + indexRow + j] = sData[threadIdx.y + i][threadIdx.x + j];
				}
			}
		}

		__syncthreads();
	}
}

