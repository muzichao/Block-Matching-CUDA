#ifndef CALCULATEMEANVAR_H
#define CALCULATEMEANVAR_H

/*
* 功能：计算每个图像块的均值和方差
* 输入：blocks_D 提取的块，一行为一个块
* 输出：blocksMean_D 每个块的均值
* 输出：blocksVar_D 每个块的方差
* 输入：rowNum 全局块的行数
* 输入：colNum 全局块的列数
*/
__global__ void BM_Calculate_Mean_Var(float *blocks_D, float *blocksMean_D, float *blocksVar_D, int rowNum, int colNum);

#endif // DIVIDEBLOCK3D_H