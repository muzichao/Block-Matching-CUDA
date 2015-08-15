function  [posIdx, weiIdx]   =  blockMaching(im, par)
% 功能 ： 块匹配，计算与中心块相似的块以及权重
% im ： 输入图像
% par ： 参数
% posIdx ： 与中心块相似的块的位置索引
% weiIdx ： 与中心块相似的块的权重索引
% 2015 06 13 李超

[height, width, ch] = size(im);
searchRadius  =  par.s1; % 搜索窗半径
similarBlkNum =  par.nblk; % 相似块的个数
win           =  par.win; % 块的大小
blockSize     =  win^2; % 每个块的元素个数
step          =  par.step; % 中心窗的步长
hp            =  80; % 高斯参数

%% 参数
N          =  height - win + 1; % 块的起始行坐标范围
M          =  width - win + 1; % 块的起始列坐标范围
blockNum   =  N * M; % 块的个数
leftUpRow  =  1:step:N; % 每个块的行起始坐标
leftUpRow  =  [leftUpRow leftUpRow(end)+1:N]; % 添加最后一个块，因为可能最后不一定正好取完
leftUpCol  =  1:step:M; % 每个块的列起始坐标
leftUpCol  =  [leftUpCol leftUpCol(end)+1:M]; % 添加最后一个块，因为可能最后不一定正好取完

%% 分块
X = divideBlock3D(im, win); % 提取了所有的块，也就是说步长为1个像素
X = X';

%% 查找相似块并计算权重
I      = reshape(1:blockNum, N, M); % 所有块的索引
I = I'; % 行优先
N1     = length(leftUpRow); % 多少行中心块
M1     = length(leftUpCol); % 多少列中心块
posIdx = zeros(similarBlkNum, N1*M1 ); % 每个中心块的相似块索引
weiIdx = zeros(similarBlkNum, N1*M1 ); % 每个中心块对应的相似块的权重
rmin   = bsxfun(@max, bsxfun(@minus, leftUpRow, searchRadius), 1); % 搜索窗的行最小坐标
rmax   = bsxfun(@min, bsxfun(@plus, leftUpRow, searchRadius), N); % 搜索窗的行最大坐标
cmin   = bsxfun(@max, bsxfun(@minus, leftUpCol, searchRadius), 1); % 搜索窗的列最小坐标
cmax   = bsxfun(@min, bsxfun(@plus, leftUpCol, searchRadius), M); % 搜索窗的列最大坐标

for  i  =  1 : N1
    for  j  =  1 : M1        
        offInAllBlk   = (leftUpRow(i) - 1) * M + leftUpCol(j); % 当前中心块在所有块中的索引(行优先)
        offInCentralBlk  = (i-1) * M1 + j; % 当前中心块在所有中心块中的索引(行优先)
        
        idx   = I(rmin(i):rmax(i), cmin(j):cmax(j));  %在由[rmin:rmax, cmin:cmax]确定的搜索窗中搜索相似块
        idx = idx';
        idx   = idx(:);
             
        dis   = sum(bsxfun(@minus, X(idx, :), X(offInAllBlk, :)).^2, 2); % 搜索窗内的块与中心块的距离
        dis   = dis ./ (blockSize*ch); % 归一化
        similarBlkInd = maxN(dis, similarBlkNum); % 找到距离最小的几个块的索引
        posIdx(:,offInCentralBlk)  =  idx(similarBlkInd); % 保存相似块索引

        wei   =  exp( -dis(similarBlkInd) ./ hp ); % 高斯
        weiIdx(:,offInCentralBlk)  =  wei ./ (sum(wei(:))+eps); % 归一化
    end
end

function index = maxN(data, N)
% 功能 ：找到 data 中最大的 N 个数，并返回索引
% 2015 06 13 李超

[~, index] = sort(data);
index = index(1:N);