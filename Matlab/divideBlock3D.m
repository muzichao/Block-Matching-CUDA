function block = divideBlock3D(im, win)
% 功能 ：提取了所有的块，也就是说步长为1个像素
% im : 输入图像
% win : 块的大小，一个维度
% 说明 :
% 2015 06 12 李超

[height, width, ch] = size(im); % 图像的行、列和页数
blockSize = win * win * ch; % 块的元素个数
N         =  height - win + 1; % 块的起始行坐标范围
M         =  width - win + 1; % 块的起始列坐标范围
blockNum  =  N * M; % 块的个数
block     =  zeros(blockSize, blockNum, 'single');

% 依次提取块
k = 0;
for m = 1:ch
    for i  = 1:win
        for j  = 1:win
            k    =  k+1;
            blk  =  im(i : height-win+i, j : width-win+j, m);
            blk = blk'; % 更改取块的方向，变成行优先取
            blk  =  blk(:);
            block(k,:) =  blk';
        end
    end
end