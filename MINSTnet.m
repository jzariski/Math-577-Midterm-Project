filenameImagesTrain = 'train-images-idx3-ubyte.gz';
filenameLabelsTrain = 'train-labels-idx1-ubyte.gz';
filenameImagesTest = 't10k-images-idx3-ubyte.gz';
filenameLabelsTest = 't10k-labels-idx1-ubyte.gz';

XTrain = processImagesMNIST(filenameImagesTrain);
YTrain = processLabelsMNIST(filenameLabelsTrain);
XTest = processImagesMNIST(filenameImagesTest);
YTest = processLabelsMNIST(filenameLabelsTest);


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
	
    convolution2dLayer(3,8,Padding="same")
    batchNormalizationLayer
    reluLayer
	
    maxPooling2dLayer(2,Stride=2)
	
    convolution2dLayer(3,24,Padding="same")
    batchNormalizationLayer
    reluLayer
	
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs',10,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');



net = trainNetwork(combined,layers,options);

finalnet = net;
save finalnet

