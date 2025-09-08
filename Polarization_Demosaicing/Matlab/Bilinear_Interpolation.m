function [I0,I45,I90,I135] = Bilinear_Interpolation(I)

II = padarray(I, [1 1], 'replicate');

[W, H] = size(II);

I0 = zeros(size(I));
I45 = zeros(size(I));
I90 = zeros(size(I));
I135 = zeros(size(I));

for i = 2 : W - 1
    for j = 2 : H - 1
        x = mod(i, 2);
        y = mod(j, 2);
        if (x == 1)        %扩充图像奇数行
            if (y == 1)    %扩充图像奇数行奇数列
                I90(i - 1, j - 1) = (II(i - 1, j - 1) + II(i + 1, j + 1) + II(i - 1, j + 1) + II(i + 1, j - 1)) / 4;
                I0(i - 1, j - 1) = II(i, j);
                I135(i - 1, j - 1) = (II(i, j + 1) + II(i, j - 1)) / 2;
                I45(i - 1, j - 1) = (II(i + 1, j) + II(i - 1, j)) / 2;
            else           %扩充图像奇数行偶数列
                I45(i - 1, j - 1) = (II(i - 1, j - 1) + II(i + 1, j + 1) + II(i - 1, j + 1) + II(i + 1, j - 1)) / 4;
                I135(i - 1, j - 1) = II(i, j);
                I0(i - 1, j - 1) = (II(i, j + 1) + II(i, j - 1)) / 2;
                I90(i - 1, j - 1) = (II(i + 1, j) + II(i - 1, j)) / 2;
            end
        else               %扩充图像偶数行
            if (y == 1)    %扩充图像偶数行奇数列                
                I135(i - 1, j - 1) = (II(i - 1, j - 1) + II(i + 1, j + 1) + II(i - 1, j + 1) + II(i + 1, j - 1)) / 4;
                I45(i - 1, j - 1) = II(i, j);
                I90(i - 1, j - 1) = (II(i, j + 1) + II(i, j - 1)) / 2;
                I0(i - 1, j - 1) = (II(i + 1, j) + II(i - 1, j)) / 2;
            else           %扩充图像偶数行偶数列
                I0(i - 1, j - 1) = (II(i - 1, j - 1) + II(i + 1, j + 1) + II(i - 1, j + 1) + II(i + 1, j - 1)) / 4;
                I90(i - 1, j - 1) = II(i, j);
                I45(i - 1, j - 1) = (II(i, j + 1) + II(i, j - 1)) / 2;
                I135(i - 1, j - 1) = (II(i + 1, j) + II(i - 1, j)) / 2;
            end
        end
    end
end


