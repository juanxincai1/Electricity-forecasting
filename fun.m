%��Ӧ�Ⱥ���
%mse��Ϊ��Ӧ��ֵ
function [fitness,net] = fun(x,numFeatures,numResponses,X) 

numHiddenUnits = round(x(1));%LSTM��·���������ص�Ԫ��Ŀ
maxEpochs = round(x(2));%���ѵ������
InitialLearnRate = x(3);%��ʼѧϰ��

num_samples = length(X);       % �������� 
kim = 5;                      % ��ʱ������kim����ʷ������Ϊ�Ա�����
zim =  1;                      % ��zim��ʱ������Ԥ��
or_dim = size(X,2);

%  �ع����ݼ�
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X(i: i + kim - 1,:), 1, kim*or_dim), X(i + kim + zim - 1,:)];
end


% ѵ�����Ͳ��Լ�����
outdim = 1;                                  % ���һ��Ϊ���
num_size = 0.7;                              % ѵ����ռ���ݼ�����
num_train_s = round(num_size * num_samples); % ѵ������������
f_ = size(res, 2) - outdim;                  % ��������ά��


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';

%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
[t_train, ps_output] = mapminmax(T_train, 0, 1);

%��������
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%ָ��ѵ��ѡ�����cpuѵ���� ������cpu��Ϊ�˱�֤��ֱ�����У������Ҫgpuѵ�����ĳ�gpu�����ˣ��ұ�֤cuda�а�װ
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'ExecutionEnvironment' ,'cpu',...
    'InitialLearnRate',InitialLearnRate,...
    'GradientThreshold',1, ...
    'LearnRateDropPeriod',100, ...
    'LearnRateDropFactor',0.2, ...%ָ����ʼѧϰ�� 0.005���� 125 ��ѵ����ͨ���������� 0.2 ������ѧϰ��
    'L2Regularization',0.01, ...
    'Verbose',0);
%'Plots','training-progress'
%ѵ��LSTM
net = trainNetwork(p_train,t_train,layers,options);


%ѵ��������
numTimeStepsTrain = size(p_train,2);
for i = 1:numTimeStepsTrain
    [net,predictTrain_fit(:,i)] = predictAndUpdateState(net,p_train(:,i),'ExecutionEnvironment','cpu');
    %��������̫�࣬������predict����predictAndUpdateState
end

%�������һ���Ƚ�
fitness = sqrt(mse(t_train-predictTrain_fit));
disp('ѵ������....')
end