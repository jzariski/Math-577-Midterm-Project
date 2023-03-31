layer = preluLayer('prelu')
validInputSize = [5 7];

X = rand(validInputSize)-0.5;
[Z,memory] = forward(layer,X)

dLdZ = rand(size(Z))
dLdAlpha = rand(size(layer.Alpha))
%[dLdX, dLdAlpha] = backward(layer,X,Z,dLdZ,memory)


checkLayer(layer,validInputSize,'ObservationDimension',1);