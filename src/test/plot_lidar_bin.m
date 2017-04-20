% fileID = fopen('kitti_005_0000000000.bin');
% A = fread(fileID,[4,Inf],'float')';
% c=[A(1:10:end,3),A(1:10:end,3),A(1:10:end,3)];
% scatter3(A(1:10:end,1),A(1:10:end,2),A(1:10:end,3),1,c,'filled');


fileID = fopen('1490991816963079454.bin');
A = fread(fileID,[4,Inf],'float')';
den=1;
c=[A(1:den:end,4),A(1:den:end,4),A(1:den:end,4)];
% figure('Color',[1 1 1]);
figure
scatter3(A(1:den:end,1),A(1:den:end,2),A(1:den:end,3),3,1-c.^0.05,'filled');
