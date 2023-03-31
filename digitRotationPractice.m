[XTrain,~,YTrain] = digitTrain4DArrayData;
[XTest,~,YTest] = digitTest4DArrayData;

storeX = arrayDatastore(XTrain,'IterationDimension',4, ...
 'OutputType','cell');
storeY = arrayDatastore(YTrain,'OutputType','cell');

combined = combine(storeX,storeY);


storeXtest = arrayDatastore(XTest,'IterationDimension',4, ...
 'OutputType','cell');
storeYtest = arrayDatastore(YTest,'OutputType','cell');

%load('finalnetdigits.mat')

%%% Actual rotation happens here
angles = predict(net, XTest);
%rotatedXTest = rotate_digits(XTest, angles);



figure
numTrainImages = numel(YTest);
idx = randperm(numTrainImages,20);
for i = 1:numel(idx)
    subplot(4,5,i)    
    imshow(XTest(:,:,:,idx(i)))
    %imshow(rotatedXTest(:,:,:,idx(i)))
    drawnow;
end





