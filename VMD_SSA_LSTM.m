clc;
clear 
close all

%% LSTMԤ��
tic
load origin_data.mat
load vmd_data.mat

disp('��������������������������������������������������������������������������������������������')
disp('��һ��LSTMԤ��')
disp('��������������������������������������������������������������������������������������������')

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
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  ��ʽת��
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    % �ع��������24���룬1���������1571�У�ͨ�������ʽת����
    % ��i=1ʱ���ѵ�1�е�24���������Ž�һ��1*1��cell���
    % �Դ����ƹ����1571��cell����Ϊ{i, 1}������1571��cell�ų�һ��
    % ��1��cell�ڲ���24������24*1�������е�
    % cell�����ݸ�ʽ��double
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

%  ����LSTM���磬
layers = [ ...
    sequenceInputLayer(f_)              % �����
    lstmLayer(70)                      
    reluLayer                           
    fullyConnectedLayer(outdim)         % �ع��
    regressionLayer];

%  ��������
options = trainingOptions('adam', ...                 % �Ż��㷨Adam
    'MaxEpochs', 70, ...                            % ���ѵ������
    'GradientThreshold', 1, ...                       % �ݶ���ֵ
    'InitialLearnRate', 0.01, ...         % ��ʼѧϰ��
    'LearnRateSchedule', 'piecewise', ...             % ѧϰ�ʵ���
    'LearnRateDropPeriod', 60, ...                   % ѵ��850�κ�ʼ����ѧϰ��
    'LearnRateDropFactor',0.2, ...                    % ѧϰ�ʵ�������
    'L2Regularization', 0.01, ...         % ���򻯲���
    'ExecutionEnvironment', 'cpu',...                 % ѵ������
    'Verbose', 0, ...                                 % �ر��Ż�����
    'Plots', 'training-progress');                    % ��������

%  ѵ��
net = trainNetwork(vp_train, vt_train, layers, options);
%analyzeNetwork(net);% �鿴����ṹ
%  Ԥ��
t_sim1 = predict(net, vp_train); 
t_sim2 = predict(net, vp_test); 

%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_train1 = T_train;
T_test2 = T_test;

%  ���ݸ�ʽת��
T_sim1 = cell2mat(T_sim1);% cell2mat��cellԪ������ת��Ϊ��ͨ����
T_sim2 = cell2mat(T_sim2);

% ָ�����
disp('ѵ�������ָ��')
[mae1,rmse1,mape1,error1]=calc_error(T_train1,T_sim1');
fprintf('\n')

disp('���Լ����ָ��')
[mae2,rmse2,mape2,error2]=calc_error(T_test2,T_sim2');
fprintf('\n')
toc


tic
disp('��������������������������������������������������������������������������������������������')
disp('VMD-LSTMԤ��')
disp('��������������������������������������������������������������������������������������������')

imf=u;
c=size(imf,1);
%% ��ÿ��������ģ
for d=1:c
disp(['��',num2str(d),'��������ģ'])

X_imf=[X(:,1:end-1) imf(d,:)'];
num_samples = length(X_imf);  % �������� 

%  �ع����ݼ�
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X_imf(i: i + kim - 1,:), 1, kim*or_dim), X_imf(i + kim + zim - 1,:)];
end


% ѵ�����Ͳ��Լ�����
outdim = 1;                                  % ���һ��Ϊ���
num_size = 0.7;                              % ѵ����ռ���ݼ�����
num_train_s = round(num_size * num_samples); % ѵ������������
f_ = size(res, 2) - outdim;                  % ��������ά��


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';


P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';


%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  ��ʽת��
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

%  ����LSTM���磬
layers = [ ...
    sequenceInputLayer(f_)              % �����
    lstmLayer(70)                      % LSTM��
    reluLayer                           % Relu�����
    fullyConnectedLayer(outdim)         % �ع��
    regressionLayer];

%  ��������
options = trainingOptions('adam', ...                 % �Ż��㷨Adam
    'MaxEpochs', 70, ...                            % ���ѵ������
    'GradientThreshold', 1, ...                       % �ݶ���ֵ
    'InitialLearnRate', 0.01, ...         % ��ʼѧϰ��
    'LearnRateSchedule', 'piecewise', ...             % ѧϰ�ʵ���
    'LearnRateDropPeriod', 60, ...                   % ѵ��850�κ�ʼ����ѧϰ��
    'LearnRateDropFactor',0.2, ...                    % ѧϰ�ʵ�������
    'L2Regularization', 0.01, ...         % ���򻯲���
    'ExecutionEnvironment', 'cpu',...                 % ѵ������
    'Verbose', 0, ...                                 % �ر��Ż�����
    'Plots', 'training-progress');                    % ��������

%  ѵ��
net = trainNetwork(vp_train, vt_train, layers, options);
%  Ԥ��
t_sim5 = predict(net, vp_train); 
t_sim6 = predict(net, vp_test); 

%  ���ݷ���һ��
T_sim5_imf = mapminmax('reverse', t_sim5, ps_output);
T_sim6_imf = mapminmax('reverse', t_sim6, ps_output);

%  ���ݸ�ʽת��
T_sim5(d,:) = cell2mat(T_sim5_imf);% cell2mat��cellԪ������ת��Ϊ��ͨ����
T_sim6(d,:) = cell2mat(T_sim6_imf);
T_train5(d,:)= T_train;
T_test6(d,:)= T_test;
end

% ������Ԥ��Ľ�����
T_sim5=sum(T_sim5);
T_sim6=sum(T_sim6);
T_train5=sum(T_train5);
T_test6=sum(T_test6);

% ָ�����
disp('ѵ�������ָ��')
[mae5,rmse5,mape5,error5]=calc_error(T_train5,T_sim5);
fprintf('\n')

disp('���Լ����ָ��')
[mae6,rmse6,mape6,error6]=calc_error(T_test6,T_sim6);
fprintf('\n')
toc

%% VMD-SSA-LSTMԤ��
tic
disp('��������������������������������������������������������������������������������������������')
disp('VMD-SSA-LSTMԤ��')
disp('��������������������������������������������������������������������������������������������')

% SSA��������
pop=3; % ��Ⱥ����
Max_iter=5; % ����������
dim=3; % �Ż�LSTM��3������
lb = [50,50,0.001];%�±߽�
ub = [300,300,0.01];%�ϱ߽�
numFeatures=f_;
numResponses=outdim;
fobj = @(x) fun(x,numFeatures,numResponses,X) ;
[Best_pos,Best_score,curve,BestNet]=SSA(pop,Max_iter,lb,ub,dim,fobj);

% ���ƽ�������
figure
plot(curve,'r-','linewidth',3)
xlabel('��������')
ylabel('���������RMSE')
legend('�����Ӧ��')
title('SSA-LSTM�Ľ�����������')

disp('')
disp(['�������ص�Ԫ��ĿΪ   ',num2str(round(Best_pos(1)))]);
disp(['�������ѵ������Ϊ   ',num2str(round(Best_pos(2)))]);
disp(['���ų�ʼѧϰ��Ϊ   ',num2str((Best_pos(3)))]);

%% ��ÿ��������ģ
for d=1:c
disp(['��',num2str(d),'��������ģ'])

X_imf=[X(:,1:end-1) imf(d,:)'];

%  �ع����ݼ�
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X_imf(i: i + kim - 1,:), 1, kim*or_dim), X_imf(i + kim + zim - 1,:)];
end


% ѵ�����Ͳ��Լ�����
outdim = 1;                                  % ���һ��Ϊ���
num_size = 0.7;                              % ѵ����ռ���ݼ�����
num_train_s = round(num_size * num_samples); % ѵ������������
f_ = size(res, 2) - outdim;                  % ��������ά��


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  ��ʽת��
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

% ��Ѳ�����LSTMԤ��
layers = [ ...
    sequenceInputLayer(f_)              % �����
    lstmLayer(round(Best_pos(1)))      % LSTM��
    reluLayer                           % Relu�����
    fullyConnectedLayer(outdim)         % �ع��
    regressionLayer];


options = trainingOptions('adam', ...                 % �Ż��㷨Adam
    'MaxEpochs', round(Best_pos(2)), ...                            % ���ѵ������
    'GradientThreshold', 1, ...                       % �ݶ���ֵ
    'InitialLearnRate', Best_pos(3), ...         % ��ʼѧϰ��
    'LearnRateSchedule', 'piecewise', ...             % ѧϰ�ʵ���
    'LearnRateDropPeriod', round(Best_pos(2)*0.9), ...                   % ѵ��850�κ�ʼ����ѧϰ��
    'LearnRateDropFactor',0.2, ...                    % ѧϰ�ʵ�������
    'L2Regularization', 0.001, ...          % ���򻯲���
    'ExecutionEnvironment', 'cpu',...                 % ѵ������
    'Verbose', 0, ...                                 % �ر��Ż�����
    'Plots', 'training-progress');                    % ��������

%  ѵ��
net = trainNetwork(vp_train, vt_train, layers, options);
%  Ԥ��
t_sim7 = predict(net, vp_train); 
t_sim8 = predict(net, vp_test); 

%  ���ݷ���һ��
T_sim7_imf = mapminmax('reverse', t_sim7, ps_output);
T_sim8_imf = mapminmax('reverse', t_sim8, ps_output);

%  ���ݸ�ʽת��
T_sim7(d,:) = cell2mat(T_sim7_imf);% cell2mat��cellԪ������ת��Ϊ��ͨ����
T_sim8(d,:) = cell2mat(T_sim8_imf);
T_train7(d,:)= T_train;
T_test8(d,:)= T_test;
end

% ������Ԥ��Ľ�����
T_sim7=sum(T_sim7);
T_sim8=sum(T_sim8);
T_train7=sum(T_train7);
T_test8=sum(T_test8);

% ָ�����
disp('ѵ�������ָ��')
[mae7,rmse7,mape7,error7]=calc_error(T_train7,T_sim7);
fprintf('\n')

disp('���Լ����ָ��')
[mae8,rmse8,mape8,error8]=calc_error(T_test8,T_sim8);
fprintf('\n')
toc

%% ����ģ�Ͳ��Լ������ͼ�Ա�

figure
plot(T_test2,'k','linewidth',3);
hold on;
plot(T_sim2,'y','linewidth',3);
hold on;
plot(T_sim6,'g','linewidth',3);
hold on;
plot(T_sim8,'r','linewidth',3);
legend('Target','LSTM','VMD-LSTM','VMD-SSA-LSTM');
title('����ģ��Ԥ�����Ա�ͼ');
xlabel('Sample Index');
ylabel('Values');
grid on;

figure
plot(error2,'k','linewidth',3);
hold on
plot(error6,'g','linewidth',3);
hold on
plot(error8,'r','linewidth',3);
legend('LSTM','VMD-LSTM','VMD-SSA-LSTM');
title('����ģ��Ԥ�����Ա�ͼ');
grid on;