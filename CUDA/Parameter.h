#ifndef PARAMETER_H
#define PARAMETER_H

#define majorRow 1 // 取块的时候行优先，0 表示列优先

#define imRow 512 // 输入图像的行数
#define imCol 512 // 输入图像的列数
#define batch 3 // 输入图像的第三维

#define winRadius 25 // 搜索窗半径
#define blockR 5 // 窗的大小
#define blockSizes (blockR * blockR * batch)
#define similarBlkNum 10 // 像素块的个数
#define winStep 2 // 窗的步长

#define blockRow (imRow - blockR) // 块的左上角的行最大值
#define blockCol (imCol - blockR) // 块的左上角的列最大值

#define allBlockNum ((blockRow + 1) * (blockCol + 1)) // 所有块的个数

#define BM_muMin 0.95f // 块的相似度均值参数
#define BM_muMax 1.05f // 块的相似度均值参数
#define BM_deltaMin 0.5f // 块的相似度方差参数
#define BM_deltaMax 1.5f // 块的相似度方差参数

#define BM_hp 80.0f // 计算相似块权重时候的高斯参数
#define BLOCKSIZE 16 // 线程块的大小
#define BLOCKSIZE_32 32 // 线程块的大小

#endif //PARAMETER_H