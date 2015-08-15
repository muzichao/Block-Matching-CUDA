#ifndef READSAVEIMAGE_H
#define READSAVEIMAGE_H

#include <iostream>
/**
* 功能：把数据保存到文本文件中
* 输入：fileData 输入数据的头指针
* 输入：fileName 保存的文本文件的文件名
* 输入：dataNum 保存的数据个数
*/
void SaveFile(float *fileData, std::string fileName, int dataNum);

/**
* 功能：从txt文件中读取数据
* 输出：fileData 输出数据的头指针
* 输入：fileName 读取的文本文件的文件名
* 输入：dataNum 读取的数据个数
*/
void ReadFile(float *fileData, std::string fileName, int dataNum);
#endif // !READSAVEIMAGE_H
