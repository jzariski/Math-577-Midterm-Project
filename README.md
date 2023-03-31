# Math-577-Midterm-Project


Important files to note:

MNISTnet.m -- Convolutional neural network used to classify images of digits.

Digitsnet.m -- Convolutional neural network used to predict the rotation angle of images of digits.

rotationLayer.m -- Custom layer built for digit rotation. Uses the weights frozen from Digitsnet.m to predict the angle needed to orient and center the digits, then applies this rotation.

finalMNISTnet_with_layer.m -- Convolutional neural network with added rotation layer, based off of Digitsnet weight. Builds on the MNISTnet.m net.

predictAngles.m -- Uses Digitsnet to predict angles needed to orient digits properly. Implemented as a separate function so it then can be incorporated into rotationLayer.m.

test.m -- Scratch script to test accuracy of classification networks. Using this, I found that adding a rotation layer didn't change the accuracy by much. However, the MNIST network already achieved close to 96% accuracy, so it would be difficult to improve it in the first place.




