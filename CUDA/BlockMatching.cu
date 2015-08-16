#include "BlockMatching.h"
#include "Parameter.h"
#include <iostream>

__constant__ int searchOffset[2][8] = { { -1, -1, -1, 0, 0, 1, 1, 1 }, { -1, 0, 1, -1, 1, -1, 0, 1 } }; // 搜索窗的偏移

/**
* 功能：计算两个向量的欧拉距离
* 输入：objects 样本数据
* 输出：clusters 聚类中心数据
* 输入：vecLength 向量长度
*/
__device__ inline static float EuclidDistance(float *center, float *search, int vecLength)
{
	float dist = 0.0f;

	for (int i = 0; i < vecLength; i++)
	{
		float onePoint = center[i] - search[i];
		dist += onePoint * onePoint;
	}
	
	return(dist);
}

/**
* 功能：查找当前块是否为相似块
* 输入：posIdx_D 当前中心块对应的相似块的位置索引
* 输出：weiIdx_D 当前中心块对应的相似块的权重索引
* 输入：wei 新的权重
* 输入：pos 新的权重对应的位置
*/
__device__ void FindSimilarBlocks(int *posIdx_D, float *weiIdx_D, float wei, int pos)
{
	int index = similarBlkNum - 1;

	while (index >= 0 && abs(weiIdx_D[index]) < 1e-6)
	{
		index--;
	}

	if (index == similarBlkNum - 1 && weiIdx_D[index] >= wei)
	{
		index--;
	}

	while (index >= 0 && weiIdx_D[index] > wei)
	{
		weiIdx_D[index + 1] = weiIdx_D[index];
		posIdx_D[index + 1] = posIdx_D[index];

		index--;
	}

	if (similarBlkNum - 1 != index)
	{
		weiIdx_D[index + 1] = wei;
		posIdx_D[index + 1] = pos;
	}
}

/**
*输出：posIdx_D 相似块的位置
* 输出：weiIdx_D 相似块的权重
* 输入：rowNum 中心块的行数
* 输入：colNum 中心块的列数
*/
__global__ void BM_Init_WeightAndPos(int *posIdx_D, float *weiIdx_D, int rowNum, int colNum)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标

	if (y_id < rowNum && x_id < colNum)
	{
		int offInCentralBlk = y_id * colNum + x_id; // 按行优先

		float *weiIdx = &weiIdx_D[offInCentralBlk * similarBlkNum];
		int *posIdx = &posIdx_D[offInCentralBlk * similarBlkNum];
		for (int i = 0; i < similarBlkNum; i++)
		{
			weiIdx[i] = 2e30;
			posIdx[i] = offInCentralBlk;
		}
	}
}

/*
* 功能：计算搜索窗的坐标范围
* 输入：blocks_D 提取的块，一行为一个块
* 输入：leftUpRow_D 每个中心块的行起始坐标
* 输入：leftUpCol_D 每个中心块的列起始坐标
* 输入：rmin_D 搜索窗的行最小坐标
* 输入：rmax_D 搜索窗的行最大坐标
* 输入：cmin_D 搜索窗的列最小坐标
* 输入：cmax_D 搜索窗的列最大坐标
* 输入：blocksMean_D 每个块的均值
* 输入：blocksVar_D 每个块的方差
* 输出：posIdx_D 相似块的位置
* 输出：weiIdx_D 相似块的权重
* 输入：rowNum 中心块的行数
* 输入：colNum 中心块的列数
*/
__global__ void BlockMatching_R(float *blocks_D, int *leftUpRow_D, int *leftUpCol_D, int *rmin_D, int *rmax_D, int *cmin_D, int *cmax_D, float *blocksMean_D, float *blocksVar_D, int *posIdx_D, float *weiIdx_D, int rowNum, int colNum)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标

	if (y_id < rowNum && x_id < colNum)
	{
		int offInAllBlk = leftUpRow_D[y_id] * (imCol - blockR + 1) + leftUpCol_D[x_id]; // 按行优先
		int offInCentralBlk = y_id * colNum + x_id; // 按行优先

		float *ptrCenter = &blocks_D[offInAllBlk * blockSizes];

		/* 遍历窗中的块 */
		for (int i = rmin_D[y_id]; i <= rmax_D[y_id]; i++)
		{
			for (int j = cmin_D[x_id]; j <= cmax_D[x_id]; j++)
			{
				int searchIdx = i * (imCol - blockR + 1) + j; // 按行优先
				float *ptrSearchIdx = &blocks_D[searchIdx * blockSizes];

				if (BM_muMax > (blocksMean_D[offInAllBlk] / blocksMean_D[searchIdx]) > BM_muMin && BM_deltaMax > (blocksVar_D[offInAllBlk] / blocksVar_D[searchIdx]) > BM_deltaMin)
				{
					float dist = EuclidDistance(ptrCenter, ptrSearchIdx, blockSizes);
					FindSimilarBlocks(&posIdx_D[offInCentralBlk * similarBlkNum], &weiIdx_D[offInCentralBlk * similarBlkNum], dist / float(blockSizes), searchIdx);
				}
			}
		}
		//if (x_id == 0 && y_id == 252)
		//{
		//	for (int i = 0; i < similarBlkNum; i++)
		//	{
		//		printf("x_id = %d, y_id = %d, posIdx_D[%d] = %d, weiIdx_D[%d] = %f\n", x_id, y_id, i, posIdx_D[offInCentralBlk * similarBlkNum + i], i, weiIdx_D[offInCentralBlk * similarBlkNum + i]);
		//	}
		//}
	}
}

/*
* 功能：计算搜索窗的坐标范围
* 输入：blocks_D 提取的块，一行为一个块
* 输入：leftUpRow_D 每个中心块的行起始坐标
* 输入：leftUpCol_D 每个中心块的列起始坐标
* 输入：blocksMean_D 每个块的均值
* 输入：blocksVar_D 每个块的方差
* 输出：posIdx_D 相似块的位置
* 输出：weiIdx_D 相似块的权重
* 输入：rowNum 中心块的行数
* 输入：colNum 中心块的列数
*/
__global__ void BlockMatching_S(float *blocks_D, int *leftUpRow_D, int *leftUpCol_D, float *blocksMean_D, float *blocksVar_D, int *posIdx_D, float *weiIdx_D, int rowNum, int colNum)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标

	if (y_id < rowNum && x_id < colNum)
	{
		int offInAllBlk = leftUpRow_D[y_id] * (imCol - blockR + 1) + leftUpCol_D[x_id]; // 按行优先
		int offInCentralBlk = y_id * colNum + x_id; // 按行优先

		float *ptrCenter = &blocks_D[offInAllBlk * blockSizes];

		/* 遍历Jump Flooding 中的块 */
		int searchIdx = offInAllBlk;
		for (int step = 1; step <= imRow / 2; step *= 2)
		{
			for (int i = 0; i < 8; i++)
			{
				int currRow = leftUpRow_D[y_id] + searchOffset[0][i] * step;
				int currCol = leftUpCol_D[x_id] + searchOffset[1][i] * step;
				if (currRow >= 0 && currRow < imRow - blockR && currCol >=0 && currCol < imCol - blockR)
				{
					searchIdx = currRow * (imCol - blockR + 1) + currCol; // 按行优先
					float *ptrSearchIdx = &blocks_D[searchIdx * blockSizes];

					if (BM_muMax >(blocksMean_D[offInAllBlk] / blocksMean_D[searchIdx]) > BM_muMin && BM_deltaMax >(blocksVar_D[offInAllBlk] / blocksVar_D[searchIdx]) > BM_deltaMin)
					{
						float dist = EuclidDistance(ptrCenter, ptrSearchIdx, blockSizes);
						FindSimilarBlocks(&posIdx_D[offInCentralBlk * similarBlkNum], &weiIdx_D[offInCentralBlk * similarBlkNum], dist / float(blockSizes), searchIdx);
					}
				}
			}
		}
		FindSimilarBlocks(&posIdx_D[offInCentralBlk * similarBlkNum], &weiIdx_D[offInCentralBlk * similarBlkNum], 0.0f, offInAllBlk);
	}
}

/*
* 功能：计算搜索窗的坐标范围
* 输入：blocks_D 提取的块，一行为一个块
* 输入：leftUpRow_D 每个中心块的行起始坐标
* 输入：leftUpCol_D 每个中心块的列起始坐标
* 输入：rmin_D 搜索窗的行最小坐标
* 输入：rmax_D 搜索窗的行最大坐标
* 输入：cmin_D 搜索窗的列最小坐标
* 输入：cmax_D 搜索窗的列最大坐标
* 输入：blocksMean_D 每个块的均值
* 输入：blocksVar_D 每个块的方差
* 输出：posIdx_D 相似块的位置
* 输出：weiIdx_D 相似块的权重
* 输入：rowNum 中心块的行数
* 输入：colNum 中心块的列数
*/
__global__ void BlockMatching_RS(float *blocks_D, int *leftUpRow_D, int *leftUpCol_D, int *rmin_D, int *rmax_D, int *cmin_D, int *cmax_D, float *blocksMean_D, float *blocksVar_D, int *posIdx_D, float *weiIdx_D, int rowNum, int colNum)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标
	int y_id = blockDim.y * blockIdx.y + threadIdx.y; // 行坐标

	if (y_id < rowNum && x_id < colNum)
	{
		int offInAllBlk = leftUpRow_D[y_id] * (imCol - blockR + 1) + leftUpCol_D[x_id]; // 按行优先
		int offInCentralBlk = y_id * colNum + x_id; // 按行优先

		float *ptrCenter = &blocks_D[offInAllBlk * blockSizes];

		/* 遍历窗中的块 */
		int searchIdx = offInAllBlk;
		for (int i = rmin_D[y_id]; i <= rmax_D[y_id]; i++)
		{
			for (int j = cmin_D[x_id]; j <= cmax_D[x_id]; j++)
			{
				searchIdx = i * (imCol - blockR + 1) + j; // 按行优先
				float *ptrSearchIdx = &blocks_D[searchIdx * blockSizes];
				if (BM_muMax > (blocksMean_D[offInAllBlk] / blocksMean_D[searchIdx]) > BM_muMin && BM_deltaMax > (blocksVar_D[offInAllBlk] / blocksVar_D[searchIdx]) > BM_deltaMin)
				{
					float dist = EuclidDistance(ptrCenter, ptrSearchIdx, blockSizes);
					FindSimilarBlocks(&posIdx_D[offInCentralBlk * similarBlkNum], &weiIdx_D[offInCentralBlk * similarBlkNum], dist / float(blockSizes), searchIdx);
				}
			}
		}

		/* 遍历Jump Flooding 中的块 */
		for (int step = winRadius + 1; step <= imRow / 2; step *= 2)
		{
			for (int i = 0; i < 8; i++)
			{
				int currRow = leftUpRow_D[y_id] + searchOffset[0][i] * step;
				int currCol = leftUpCol_D[x_id] + searchOffset[1][i] * step;
				if (currRow >= 0 && currRow < imRow - blockR && currCol >=0 && currCol < imCol - blockR)
				{
					searchIdx = currRow * (imCol - blockR + 1) + currCol; // 按行优先
					float *ptrSearchIdx = &blocks_D[searchIdx * blockSizes];

					if (BM_muMax >(blocksMean_D[offInAllBlk] / blocksMean_D[searchIdx]) > BM_muMin && BM_deltaMax >(blocksVar_D[offInAllBlk] / blocksVar_D[searchIdx]) > BM_deltaMin)
					{
						float dist = EuclidDistance(ptrCenter, ptrSearchIdx, blockSizes);
						FindSimilarBlocks(&posIdx_D[offInCentralBlk * similarBlkNum], &weiIdx_D[offInCentralBlk * similarBlkNum], dist / float(blockSizes), searchIdx);
					}
				}
			}
		}
	}
}

/*
* 功能：计算相似块的权重
* 输出：weiIdx_D 相似块的权重
* 输入：rowNum 中心块的行数
* 输入：colNum 中心块的列数
*/
__global__ void BM_Calculate_Weight(float *weiIdx_D, int rowNum, int colNum)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x; // 列坐标 

	if (x_id < rowNum * colNum)
	{
		__shared__ float sData[BLOCKSIZE * BLOCKSIZE][similarBlkNum];

		float *currBlk = &weiIdx_D[x_id * similarBlkNum];
		float sum = 1e-15;
		
		if (x_id == 0 * colNum + 252)
		{
			for (int i = 0; i < similarBlkNum; i++)
			{
				printf("x_id = %d, sum = %f, weiIdx_D[%d] = %f\n", x_id, sum, i, weiIdx_D[x_id * similarBlkNum + i]);
			}
			printf("\n");
		}

		/* 高斯加权 并 读入共享内存*/
		for (int i = 0; i < similarBlkNum; i++)
		{
			sData[threadIdx.x][i] = exp(-currBlk[i] / BM_hp);
		}

		__syncthreads();

		/* 求和 */
		for (int i = 0; i < similarBlkNum; i++)
		{
			sum += sData[threadIdx.x][i];
		}
		__syncthreads();

		/* 量化 */
		for (int i = 0; i < similarBlkNum; i++)
		{
			currBlk[i] = sData[threadIdx.x][i] / sum;
		}

		if (x_id == 0 * colNum + 252)
		{
			for (int i = 0; i < similarBlkNum; i++)
			{
				printf("x_id = %d, sum = %f, weiIdx_D[%d] = %f\n", x_id, sum, i, weiIdx_D[x_id * similarBlkNum + i]);
			}
		}
	}
}