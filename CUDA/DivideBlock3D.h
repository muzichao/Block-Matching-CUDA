#ifndef DIVIDEBLOCK3D_H
#define DIVIDEBLOCK3D_H

/*
* 功能：从图像中提取块
* 输入：im_D 待提取块的图像
* 输出：blocks_D 提取的块，一行为一个块
* 输入：row 输入图像行数
* 输入：col 输入图像列数
*/
__global__ void DivideBlock3D(float *im_D, float *blocks_D, int row, int col);

#endif // DIVIDEBLOCK3D_H