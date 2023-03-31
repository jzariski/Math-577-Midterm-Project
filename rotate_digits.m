function RotatedDigits = rotate_digits(Digits, Angles)
    RotatedDigits=zeros(size(Digits),'like',Digits);
    for j=1:size(Digits,4)
        RotatedDigits(:,:,:,j) = imrotate(Digits(:,:,:,j),Angles(j),'bilinear','crop');        
    end
end
