#include "Parameter.h"
#include "ReadSaveImage.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\sequence.h>

#include "SetSearchWinCoordinate.h"
#include "DivideBlock3D.h"
#include "BlockMatching.h"

#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::string;
using std::vector;


void CUDAmain(float *im_H)
{
	/* 初始化参数 */
	vector<int> leftUpRow; // 每个中心块的行起始坐标
	vector<int> leftUpCol; // 每个中心块的列起始坐标

	SetParameter(imRow, imCol, winStep, blockR, leftUpRow, leftUpCol);

	int N1 = leftUpRow.size(); // 多少行中心块
	int M1 = leftUpCol.size(); // 多少列中心块

	int N = imRow - blockR + 1; // 多少行全局块
	int M = imCol - blockR + 1; // 多少列全局块


	/* 开辟主机端内存 */
	float *blocks_H = (float*)malloc(allBlockNum * blockSizes * sizeof(float));

	/* 开辟设备端内存 */
	float *im_D;
	cudaMalloc((void**)&im_D, imRow * imCol * batch * sizeof(float));

	float *blocks_D;
	cudaMalloc((void**)&blocks_D, allBlockNum * blockSizes * sizeof(float));
	cudaMemset(blocks_D, 0, allBlockNum * blockSizes * sizeof(float));
	
	int *leftUpRow_D, *leftUpCol_D;
	cudaMalloc((void**)&leftUpRow_D, N1 * sizeof(int));
	cudaMalloc((void**)&leftUpCol_D, M1 * sizeof(int));

	int *rmin_D, *rmax_D, *cmin_D, *cmax_D;
	cudaMalloc((void**)&rmin_D, N1 * sizeof(int));
	cudaMalloc((void**)&rmax_D, N1 * sizeof(int));
	cudaMalloc((void**)&cmin_D, M1 * sizeof(int));
	cudaMalloc((void**)&cmax_D, M1 * sizeof(int));

	int *I_D;
	cudaMalloc((void**)&I_D, N * M * sizeof(int));

	int *posIdx_D;
	cudaMalloc((void**)&posIdx_D, N1 * M1 * similarBlkNum * sizeof(int));
	cudaMemset(posIdx_D, 0, N1 * M1 * similarBlkNum * sizeof(int));

	float *weiIdx_D;
	cudaMalloc((void**)&weiIdx_D, N1 * M1 * similarBlkNum * sizeof(float));
	cudaMemset(weiIdx_D, 0, N1 * M1 * similarBlkNum * sizeof(float));


	/* 拷贝数据 */
	cudaMemcpy(im_D, im_H, imRow * imCol * batch * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(leftUpRow_D, &leftUpRow[0], N1 *sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(leftUpCol_D, &leftUpCol[0], M1 *sizeof(int), cudaMemcpyHostToDevice);

	/* 定义线程格和线程块 */
	dim3 dimBlock1D(BLOCKSIZE * BLOCKSIZE);
	dim3 dimBlock2D(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid1D_4(4);
	dim3 dimGrid2D_N_M((M + dimBlock2D.x - 1) / dimBlock2D.x, (N + dimBlock2D.y - 1) / dimBlock2D.y);
	dim3 dimGrid2D_N1_M1((M1 + dimBlock2D.x - 1) / dimBlock2D.x, (N1 + dimBlock2D.y - 1) / dimBlock2D.y);
	dim3 dimGrid1D_N1M1((N1 * M1 + dimBlock1D.x - 1) / dimBlock1D.x);


	/* 记录时间 */
	cudaEvent_t start_GPU, end_GPU;
	float elaspsedTime;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&end_GPU);
	cudaEventRecord(start_GPU, 0);

	SetSearchWinCoordinate<<<dimGrid1D_4, dimBlock1D>>>(leftUpRow_D, leftUpCol_D, rmin_D, rmax_D, cmin_D, cmax_D, N1, M1, (N - 1), (M - 1));

	/* 隐藏元素指针到 device_vector 指针 */
	thrust::device_ptr<int> I_D_ptr(I_D);

	/* 产生 0 - (N * M) 的索引，用于按键值排序及合并 */
	thrust::sequence(I_D_ptr, I_D_ptr + (N * M));

	/* 计时结束 */
	cudaEventRecord(end_GPU, 0);
	cudaEventSynchronize(end_GPU);
	cudaEventElapsedTime(&elaspsedTime, start_GPU, end_GPU);

	/* 打印信息 */
	std::cout << "初始化的时间为：" << elaspsedTime << "ms." << std::endl;

	/*****************************************************************************************************************************************************/

	/* 记录时间 */
	elaspsedTime = 0.0f;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&end_GPU);
	cudaEventRecord(start_GPU, 0);

	DivideBlock3D<<<dimGrid2D_N_M, dimBlock2D>>>(im_D, blocks_D, imRow, imCol);

	/* 计时结束 */
	cudaEventRecord(end_GPU, 0);
	cudaEventSynchronize(end_GPU);
	cudaEventElapsedTime(&elaspsedTime, start_GPU, end_GPU);

	/* 打印信息 */
	std::cout << "DivideBlock3D 的时间为：" << elaspsedTime << "ms." << std::endl;

	/*****************************************************************************************************************************************************/
	/* 记录时间 */
	elaspsedTime = 0.0f;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&end_GPU);
	cudaEventRecord(start_GPU, 0);

	//BlockMatching_R<<<dimGrid2D_N1_M1, dimBlock2D>>>(blocks_D, leftUpRow_D, leftUpCol_D, rmin_D, rmax_D, cmin_D, cmax_D, posIdx_D, weiIdx_D, N1, M1);
	BlockMatching_S<<<dimGrid2D_N1_M1, dimBlock2D>>>(blocks_D, leftUpRow_D, leftUpCol_D, posIdx_D, weiIdx_D, N1, M1);
	//BlockMatching_RS<<<dimGrid2D_N1_M1, dimBlock2D>>>(blocks_D, leftUpRow_D, leftUpCol_D, rmin_D, rmax_D, cmin_D, cmax_D, posIdx_D, weiIdx_D, N1, M1);

	/* 计时结束 */
	cudaEventRecord(end_GPU, 0);
	cudaEventSynchronize(end_GPU);
	cudaEventElapsedTime(&elaspsedTime, start_GPU, end_GPU);

	/* 打印信息 */
	std::cout << "BlockMatching 的时间为：" << elaspsedTime << "ms." << std::endl;

	/*****************************************************************************************************************************************************/
	/* 记录时间 */
	elaspsedTime = 0.0f;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&end_GPU);
	cudaEventRecord(start_GPU, 0);

	BM_Calculate_Weight<<<dimGrid1D_N1M1, dimBlock1D>>>(weiIdx_D, N1, M1);

	/* 计时结束 */
	cudaEventRecord(end_GPU, 0);
	cudaEventSynchronize(end_GPU);
	cudaEventElapsedTime(&elaspsedTime, start_GPU, end_GPU);

	/* 打印信息 */
	std::cout << "更新权重的时间为：" << elaspsedTime << "ms." << std::endl;

	/*****************************************************************************************************************************************************/


	/* 释放设备端内存*/
	cudaFree(im_D);
	cudaFree(blocks_D);
	cudaFree(leftUpRow_D);
	cudaFree(leftUpCol_D);
	cudaFree(rmin_D);
	cudaFree(rmax_D);
	cudaFree(cmin_D);
	cudaFree(cmax_D);
	cudaFree(weiIdx_D);
	cudaFree(posIdx_D);
	cudaFree(I_D);
	

	/* 释放主机端内存 */
	free(blocks_H);
}

int main()
{
	string strIm = "D:\\Document\\vidpic\\CUDA\\BlockMatch\\im.txt";
	float *im_H = (float*)malloc(imRow * imCol * batch * sizeof(float));
	ReadFile(im_H, strIm, imRow * imCol * batch);

	CUDAmain(im_H);

	free(im_H);

	cudaDeviceReset();
	return 0;
}