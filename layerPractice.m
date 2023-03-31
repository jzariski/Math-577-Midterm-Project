

net = load('finalnetdigits.mat', 'net');


%layer = rotationLayer('rlayer', net)
%validInputSize = [28 28 1];


%checkLayer(layer,validInputSize);

trainImagesFile = "train-images-idx3-ubyte.gz";
testImagesFile = "t10k-images-idx3-ubyte.gz";

trainImagesFiley = "train-labels-idx1-ubyte.gz";
testImagesFiley = "t10k-labels-idx1-ubyte.gz";


XTrain = processImagesMNIST(trainImagesFile);

XTest = processImagesMNIST(testImagesFile);

YTrain = processLabelsMNIST(trainImagesFiley);

YTest = processLabelsMNIST(testImagesFiley);

angles = predictAngles(XTrain, net);

newXTrain = rotate_digits(XTrain, angles);


X = XTrain(:,:,:,1);
predict(net, X)
