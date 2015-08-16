#include "CalculateMeanVar.h"
#include "Parameter.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
/*
* 功能：计算每个图像块的均值和方差
* 输入：blocks_D 提取的块，一行为一个块
* 输出：blocksMean_D 每个块的均值
* 输出：blocksVar_D 每个块的方差
* 输入：rowNum 全局块的行数
* 输入：colNum 全局块的列数
*/
__global__ void BM_Calculate_Mean_Var(float *blocks_D, float *blocksMean_D, float *blocksVar_D, int rowNum, int colNum)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标 

	if (x_id < rowNum * colNum)
	{
		float *currBlk = &blocks_D[x_id * blockSizes];

		/* 计算均值 */
		float blocksMean = 0.0f;
		for (int i = 0; i < blockSizes; i++)
		{
			blocksMean += currBlk[i];
		}
		blocksMean = blocksMean / blockSizes;

		blocksMean_D[x_id] = blocksMean;

		/* 计算方差 */
		float blocksVar = 0.0f;
		for (int i = 0; i < blockSizes; i++)
		{
			blocksVar += (currBlk[i] - blocksMean) * (currBlk[i] - blocksMean);
		}
		blocksVar = blocksVar / blockSizes;

		blocksVar_D[x_id] = blocksVar;
	}
}