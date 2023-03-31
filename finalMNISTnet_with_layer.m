
digitsNet = load('finalnetdigits.mat', 'net');

trainImagesFile = "train-images-idx3-ubyte.gz";
testImagesFile = "t10k-images-idx3-ubyte.gz";

trainImagesFiley = "train-labels-idx1-ubyte.gz";
testImagesFiley = "t10k-labels-idx1-ubyte.gz";


XTrain = processImagesMNIST(trainImagesFile);

XTest = processImagesMNIST(testImagesFile);

YTrain = processLabelsMNIST(trainImagesFiley);

YTest = processLabelsMNIST(testImagesFiley);


imageSize = [28 28 1];

%% Neural Network
layers = [
    imageInputLayer(imageSize,Normalization="none")
    rotationLayer('rlayer', digitsNet.net)
	
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



newnet = trainNetwork(XTrain, YTrain, layers,options);

mnist_with_rotation = newnet;
save mnist_with_rotation


