function predictedAngles = predictAngles(Digits, net)
    newDigits = extractdata(Digits);
    predictedAngles = predict(net, newDigits);
end