[XTrain,~,YTrain] = digitTrain4DArrayData;
[XTest,~,YTest] = digitTest4DArrayData;

storeX = arrayDatastore(XTrain,'IterationDimension',4, ...
 'OutputType','cell');
storeY = arrayDatastore(YTrain,'OutputType','cell');

combined = combine(storeX,storeY);


storeXtest = arrayDatastore(XTest,'IterationDimension',4, ...
 'OutputType','cell');
storeYtest = arrayDatastore(YTest,'OutputType','cell');


%% Neural Network
layers = [
    imageInputLayer([28 28 1])
	
    convolution2dLayer(3,25)
    batchNormalizationLayer
    reluLayer
	
    maxPooling2dLayer(2,Stride=2)
	
    convolution2dLayer(3,50)
    batchNormalizationLayer
    reluLayer
	
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',50,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(combined,layers,options);

finalnetdigits = net;
save finalnetdigits

