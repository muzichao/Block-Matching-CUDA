
%将数据保存为文本文件，方便vs调用

clc
clear all
close all

load im.mat

saveMatToText(im, 'im.txt');

