originalNet = load('finalnet.mat', 'net');

changedNet = load('mnist_with_rotation.mat', 'newnet');

trainImagesFile = "train-images-idx3-ubyte.gz";
testImagesFile = "t10k-images-idx3-ubyte.gz";

trainImagesFiley = "train-labels-idx1-ubyte.gz";
testImagesFiley = "t10k-labels-idx1-ubyte.gz";


XTrain = processImagesMNIST(trainImagesFile);

XTest = processImagesMNIST(testImagesFile);

YTrain = processLabelsMNIST(trainImagesFiley);

YTest = processLabelsMNIST(testImagesFiley);


originalPredictions = classify(originalNet.net, XTest);
changedPredictions = classify(changedNet.newnet, XTest);

accuracyOriginal = sum(YTest == originalPredictions)/numel(YTest);

accuracyChanged = sum(YTest == changedPredictions)/numel(YTest);













