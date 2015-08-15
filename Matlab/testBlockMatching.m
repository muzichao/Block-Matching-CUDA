% 取块时行优先

clc;
clear;
close all

%%
par.s1   = 25; % 搜索窗半径
par.nblk = 10; % 相似块的个数
par.win  = 5; % 块的大小
par.step = 2; % 中心窗的步长

im = single(imread('LenaRGB.bmp'));

%%

tic
[posArr, weiArr]   =  blockMaching(im, par);
toc

