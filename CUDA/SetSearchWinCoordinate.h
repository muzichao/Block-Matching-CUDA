#ifndef SETSEARCHWINCOORDINATE_H
#define SETSEARCHWINCOORDINATE_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>

/*
* 功能：计算中心块起始坐标范围
* 输入：row 图像行数
* 输入：col 图像列数
* 输入：iBlock 块的大小
* 输入：leftUpRow 块的起始行坐标
* 输出：leftUpCol 块的起始列坐标
* 输入：im_H 图像，存储顺序：行-列-页
*/
void SetParameter(int row, int col, int step, int iBlock, std::vector<int> &leftUpRow, std::vector<int> &leftUpCol);

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
__global__ void SetSearchWinCoordinate(int *leftUpRow_D, int *leftUpCol_D, int *rmin_D, int *rmax_D, int *cmin_D, int *cmax_D, int rowNum, int colNum, int rowMax, int colMax);

#endif // SETSEARCHWINCOORDINATE_H