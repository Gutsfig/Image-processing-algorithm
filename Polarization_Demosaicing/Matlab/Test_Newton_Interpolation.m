clc
close all;
clear all;
%%%%%%%%%%1. Read real DoFP image %%%%%%% 
[imagename0,imagepath0]=uigetfile('I:\图像数据\偏振图像\0405-pol\*.jpg;*.bmp;*.png;*.tif;*.tiff;*.pgm;*.gif','Please choose the first input image'); 
I = double(imread(strcat(imagepath0,imagename0))); 

%%%%%%%%%%2. Demosaicking DoFP image with Newton's polynomial intepolation method %%%%%% 
[PI90,PI45,PI0,PI135] = Newton_Polynomial_Interpolation(I);

%%%%%%%%%%3. Calculating the Stokes parameters %%%% 
ii = 0.5*(PI0 + PI45 + PI90 + PI135);
S0P = im2uint8(mat2gray(ii));
q = PI0 - PI90;
u = PI45 - PI135;
dolp = sqrt(q.*q + u.*u);
Dolp = dolp./ii;
[W, H] = size(Dolp);

DOLPP = im2uint8(mat2gray(Dolp));
aop = (1/2) * atan(u./q);
AOPP = im2uint8(mat2gray(aop));

S0 = im2uint8(mat2gray(ii));
S1 = im2uint8(mat2gray(q));
S2 = im2uint8(mat2gray(u));
I0 = im2uint8(mat2gray(PI0));
I45 = im2uint8(mat2gray(PI45));
I90 = im2uint8(mat2gray(PI90));
I135 = im2uint8(mat2gray(PI135));

%保存图像
imwrite(I0,'I0.BMP');
imwrite(I45,'I45.BMP');
imwrite(I90,'I90.BMP');
imwrite(I135,'I135.BMP');
imwrite(S0,'S0.BMP');
imwrite(S1,'S1.BMP');
imwrite(S2,'S2.BMP');
imwrite(DOLPP,'DOP.BMP');
imwrite(AOPP,'AOP.BMP');
