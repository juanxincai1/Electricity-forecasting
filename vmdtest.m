tic
clc
clear all
fs=1;%采样频率，即时间序列两个数据之间的时间间隔，这里间隔1h采样
Ts=1/fs;%采样周期

X = xlsread('北半球光伏数据.xlsx','C2:E296');

save origin_data X

L=length(X);%采样点数,即有多少个数据
t=(0:L-1)*Ts;%时间序列
STA=0; %采样起始位置，这里第0h开始采样

%--------- some sample parameters forVMD：对于VMD样品参数进行设置---------------
alpha = 2500;       % moderate bandwidth constraint：适度的带宽约束/惩罚因子
tau = 0;          % noise-tolerance (no strict fidelity enforcement)：噪声容限（没有严格的保真度执行）
K = 5;              % modes：分解的模态数
DC = 0;             % no DC part imposed：无直流部分
init = 1;           % initialize omegas uniformly  ：omegas的均匀初始化
tol = 1e-7         
%--------------- Run actual VMD code:数据进行vmd分解---------------------------
[u, u_hat, omega] = VMD(X(:,end), alpha, tau, K, DC, init, tol);

save vmd_data u

figure(1);
imfn=u;
n=size(imfn,1); %size(X,1),返回矩阵X的行数；size(X,2),返回矩阵X的列数；N=size(X,2)，就是把矩阵X的列数赋值给N
subplot(n+1,1,1);  % m代表行，n代表列，p代表的这个图形画在第几行、第几列。例如subplot(2,2,[1,2])
plot(t,X(:,end)); 
ylabel('原始信号','fontsize',12,'fontname','宋体');
title('VMD分解');

for n1=1:n
    subplot(n+1,1,n1+1);
    plot(t,u(n1,:));%输出IMF分量，a(:,n)则表示矩阵a的第n列元素，u(n1,:)表示矩阵u的n1行元素
    ylabel(['IMF' int2str(n1)]);%int2str(i)是将数值i四舍五入后转变成字符，y轴命名
end
 xlabel('时间\itt/hour','fontsize',12,'fontname','宋体');


