clear
clc
close all

% addpath to the libsvm toolbox
addpath('../libsvm-3.21/matlab');

% addpath to the data
dirData = '../libsvm-3.21'; 
addpath(dirData);


% read the data set
[heart_scale_label, heart_scale_inst] = libsvmread(fullfile(dirData,'heart_scale'));
[N D] = size(heart_scale_inst);

% Determine the train and test index
trainIndex = zeros(N,1); trainIndex(1:200) = 1;
testIndex = zeros(N,1); testIndex(201:N) = 1;
trainData = heart_scale_inst(trainIndex==1,:);
trainLabel = heart_scale_label(trainIndex==1,:);
testData = heart_scale_inst(testIndex==1,:);
testLabel = heart_scale_label(testIndex==1,:);
C = 200;
[N,K] = size(trainData);
fprintf('N = %i, K = %i\n', N, K);
if ~exist('iw')
    iw = randn(K, 1);
end
w = iw;
for a = 1:1000   
    gammaD = ones(200,1);
    sumxd = zeros(K,K);
    sumyd = zeros(K,1);

    for m=1:200
        gammaD(m) = abs(1-trainLabel(m)*(w'*trainData(m,:)'));
        sumxd = sumxd + ((trainData(m,:)' * trainData(m,:)) / gammaD(m));
        sumyd = sumyd + trainLabel(m) * (1 + (1/gammaD(m)) * trainData(m,:)');
    end 
    delta = eye(K)*C;
    sigma =  zeros(K,K);
    sigma = inv(delta + sumxd); 
    mu = sigma * sumyd;
    w = mu ;
end
accu = 0;
for m=1:70
    temp = (w' * testData(m,:)');
    if temp >= 0
        temp = 1;
    end
    if temp < 0
        temp = -1;
    end
    if testLabel(m) == temp
        accu = accu + 1;
    end
end
fprintf('Accu = %0.4f,\n', (accu/70)*100);