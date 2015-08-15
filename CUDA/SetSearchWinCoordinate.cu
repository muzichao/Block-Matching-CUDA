#include "SetSearchWinCoordinate.h"
#include "Parameter.h"

/*
* 功能：计算中心块起始坐标范围
* 输入：row 图像行数
* 输入：col 图像列数
* 输入：iBlock 块的大小
* 输入：leftUpRow 块的起始行坐标
* 输出：leftUpCol 块的起始列坐标
* 输入：im_H 图像，存储顺序：行-列-页
*/
void SetParameter(int row, int col, int step, int iBlock, std::vector<int> &leftUpRow, std::vector<int> &leftUpCol)
{
	int N = row - iBlock; // 块的起始行坐标范围
	int M = col - iBlock; // 块的起始列坐标范围

	for (int i = 0; i < N; i += step)
	{
		leftUpRow.push_back(i);
	}

	if (leftUpRow.back() != N)
	{
		leftUpRow.push_back(N);
	}

	for (int i = 0; i < M; i += step)
	{
		leftUpCol.push_back(i);
	}

	if (leftUpCol.back() != M)
	{
		leftUpCol.push_back(M);
	}
}

/*
* 功能：计算搜索窗的坐标范围
* 输入：leftUpRow_D 每个中心块的行起始坐标
* 输入：leftUpCol_D 每个中心块的列起始坐标
* 输出：rmin_D 搜索窗的行最小坐标
* 输出：rmax_D 搜索窗的行最大坐标
* 输出：cmin_D 搜索窗的列最小坐标
* 输出：cmax_D 搜索窗的列最大坐标
* 输入：rowNum 中心块的行数
* 输入：colNum 中心块的列数
* 输入：rowMax 搜索窗行数的最大坐标，从 0 开始
* 输入：colMax 搜索窗列数的最大坐标，从 0 开始
*/
__global__ void SetSearchWinCoordinate(int *leftUpRow_D, int *leftUpCol_D, int *rmin_D, int *rmax_D, int *cmin_D, int *cmax_D, int rowNum, int colNum, int rowMax, int colMax)
{
	int x_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (0 == blockIdx.x) /* 搜索窗的行最小坐标 */
	{
		for (int i = threadIdx.x; i < rowNum; i += blockDim.x)
		{
			int currData = leftUpRow_D[i] - winRadius;
			rmin_D[i] = currData >= 0 ? currData : 0;
			//printf("rmin_D[%d] = %d\n", i, rmin_D[i]);
		}
	}
	else if (1 == blockIdx.x) /* 搜索窗的行最大坐标 */
	{
		for (int i = threadIdx.x; i < rowNum; i += blockDim.x)
		{
			int currData = leftUpRow_D[i] + winRadius;
			rmax_D[i] = currData < rowMax ? currData : rowMax;
			//printf("rmax_D[%d] = %d\n", i, rmax_D[i]);
		}
	}
	else if (2 == blockIdx.x) /* 搜索窗的列最小坐标 */
	{
		for (int i = threadIdx.x; i < colNum; i += blockDim.x)
		{
			int currData = leftUpCol_D[i] - winRadius;
			cmin_D[i] = currData >= 0 ? currData : 0;
			//printf("cmin_D[%d] = %d\n", i, cmin_D[i]);
		}
	}
	else if (3 == blockIdx.x) /* 搜索窗的列最大坐标 */
	{
		for (int i = threadIdx.x; i < colNum; i += blockDim.x)
		{
			int currData = leftUpCol_D[i] + winRadius;
			cmax_D[i] = currData < colMax ? currData : colMax;
			//printf("cmax_D[%d] = %d\n", i, cmax_D[i]);
		}
	}

}