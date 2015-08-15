#ifndef BLOCKMATCHING_H
#define BLOCKMATCHING_H


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
* 功能：计算搜索窗的坐标范围 (窗)
* 输入：blocks_D 提取的块，一行为一个块
* 输入：leftUpRow_D 每个中心块的行起始坐标
* 输入：leftUpCol_D 每个中心块的列起始坐标
* 输入：rmin_D 搜索窗的行最小坐标
* 输入：rmax_D 搜索窗的行最大坐标
* 输入：cmin_D 搜索窗的列最小坐标
* 输入：cmax_D 搜索窗的列最大坐标
* 输出：posIdx_D 相似块的位置
* 输出：weiIdx_D 相似块的权重
* 输入：rowNum 中心块的行数
* 输入：colNum 中心块的列数
*/
__global__ void BlockMatching_R(float *blocks_D, int *leftUpRow_D, int *leftUpCol_D, int *rmin_D, int *rmax_D, int *cmin_D, int *cmax_D, int *posIdx_D, float *weiIdx_D, int rowNum, int colNum);

/*
* 功能：计算搜索窗的坐标范围 (Jump Flooding)
* 输入：blocks_D 提取的块，一行为一个块
* 输入：leftUpRow_D 每个中心块的行起始坐标
* 输入：leftUpCol_D 每个中心块的列起始坐标
* 输出：posIdx_D 相似块的位置
* 输出：weiIdx_D 相似块的权重
* 输入：rowNum 中心块的行数
* 输入：colNum 中心块的列数
*/
__global__ void BlockMatching_S(float *blocks_D, int *leftUpRow_D, int *leftUpCol_D, int *posIdx_D, float *weiIdx_D, int rowNum, int colNum);

/*
* 功能：计算搜索窗的坐标范围 (窗 + Jump Flooding)
* 输入：blocks_D 提取的块，一行为一个块
* 输入：leftUpRow_D 每个中心块的行起始坐标
* 输入：leftUpCol_D 每个中心块的列起始坐标
* 输入：rmin_D 搜索窗的行最小坐标
* 输入：rmax_D 搜索窗的行最大坐标
* 输入：cmin_D 搜索窗的列最小坐标
* 输入：cmax_D 搜索窗的列最大坐标
* 输出：posIdx_D 相似块的位置
* 输出：weiIdx_D 相似块的权重
* 输入：rowNum 中心块的行数
* 输入：colNum 中心块的列数
*/
__global__ void BlockMatching_RS(float *blocks_D, int *leftUpRow_D, int *leftUpCol_D, int *rmin_D, int *rmax_D, int *cmin_D, int *cmax_D, int *posIdx_D, float *weiIdx_D, int rowNum, int colNum);
/*
* 功能：计算相似块的权重
* 输出：weiIdx_D 相似块的权重
* 输入：rowNum 中心块的行数
* 输入：colNum 中心块的列数
*/
__global__ void BM_Calculate_Weight(float *weiIdx_D, int rowNum, int colNum);
#endif // BLOCKMATCHING_H