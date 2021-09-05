clear all;
close all;
clc;


xhr=importdata('HR_data.txt',',');
xbr=importdata('BR_data.txt',',');

data=zeros(600,2,20);
data(:,1,:)=xhr;
data(:,2,:)=xbr;

data=num2cell(data,[2,3]);
for i=1:length(data)
    data{i}=squeeze(data{i});
    
end
ages=importdata('age_data.txt',',');
bins=[0,72,156];
% bins=[0,48,96,156];
% bins=[0,36,72,108,156];
y=discretize(ages,bins,'categorical');

%%
[m,n] = size(data) ;
P = 0.70 ;
idx = randperm(m)  ;
Xtrain = data(idx(1:round(P*m)),:) ; 
Xtest = data(idx(round(P*m)+1:end),:) ;
ytrain=y(idx(1:round(P*m)),:);
ytest=y(idx(round(P*m)+1:end),:);

%%
inputSize = 2;
numHiddenUnits = 50;
numClasses = length(bins)-1;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

maxEpochs = 100;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'SequenceLength','longest', ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(Xtrain,ytrain,layers,options);

%%
ypred = classify(net,Xtest);

acc = sum(ypred == ytest)./numel(ytest);

M=confusionmat(ytest,ypred)