function saveMatToText(data, saveFileName)
% 将数据保存为文本文件

fid=fopen(saveFileName, 'wt');

% 一行一行写入，空格隔开，最后换行
for k = 1:size(data, 3)
    for i = 1:size(data, 1)
        fprintf(fid, '%f ', data(i, 1:end-1, k));
        fprintf(fid, '%f\n', data(i, end, k));
    end
end

fclose(fid);