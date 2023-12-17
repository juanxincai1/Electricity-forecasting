%适应度函数
%mse作为适应度值
function [fitness,net] = fun(x,numFeatures,numResponses,X) 

numHiddenUnits = round(x(1));%LSTM网路包含的隐藏单元数目
maxEpochs = round(x(2));%最大训练周期
InitialLearnRate = x(3);%初始学习率

num_samples = length(X);       % 样本个数 
kim = 5;                      % 延时步长（kim个历史数据作为自变量）
zim =  1;                      % 跨zim个时间点进行预测
or_dim = size(X,2);

%  重构数据集
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X(i: i + kim - 1,:), 1, kim*or_dim), X(i + kim + zim - 1,:)];
end


% 训练集和测试集划分
outdim = 1;                                  % 最后一列为输出
num_size = 0.7;                              % 训练集占数据集比例
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';

%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
[t_train, ps_output] = mapminmax(T_train, 0, 1);

%设置网络
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%指定训练选项，采用cpu训练， 这里用cpu是为了保证能直接运行，如果需要gpu训练，改成gpu就行了，且保证cuda有安装
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'ExecutionEnvironment' ,'cpu',...
    'InitialLearnRate',InitialLearnRate,...
    'GradientThreshold',1, ...
    'LearnRateDropPeriod',100, ...
    'LearnRateDropFactor',0.2, ...%指定初始学习率 0.005，在 125 轮训练后通过乘以因子 0.2 来降低学习率
    'L2Regularization',0.01, ...
    'Verbose',0);
%'Plots','training-progress'
%训练LSTM
net = trainNetwork(p_train,t_train,layers,options);


%训练集测试
numTimeStepsTrain = size(p_train,2);
for i = 1:numTimeStepsTrain
    [net,predictTrain_fit(:,i)] = predictAndUpdateState(net,p_train(:,i),'ExecutionEnvironment','cpu');
    %参数更新太多，不能用predict，用predictAndUpdateState
end

%如果不归一化比较
fitness = sqrt(mse(t_train-predictTrain_fit));
disp('训练结束....')
end