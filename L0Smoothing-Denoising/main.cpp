#include "l0_smooth.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float* matToFloatArray(const cv::Mat& image) {
    CV_Assert(image.type() == CV_32FC1);
    float* data = new float[image.rows * image.cols];
    memcpy(data, image.data, image.rows * image.cols * sizeof(float));
    return data;
}

void ProcessFrame(cv::Mat& input, cv::Mat& output) {
    cv::Mat gray;
    if (input.channels() == 1) {
        input.convertTo(gray, CV_32FC1, 1.0 / 255.0);
    } else {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_32FC1, 1.0 / 255.0);
    }

    float* imgData = matToFloatArray(gray);
    L0Smoothing(imgData, gray.rows, gray.cols, 0.001f, 4.0f);
    output = cv::Mat(gray.rows, gray.cols, CV_32FC1, imgData);
    output.convertTo(output, CV_8UC1, 255.0);
    delete[] imgData;
}

int main() {
    std::string imagePath = "D:\\image\\input4\\1.jpg";
    
    // 读取图像
    cv::Mat input = cv::imread(imagePath);
    if (input.empty()) {
        std::cerr << "Error: Could not load image " << imagePath << std::endl;
        return -1;
    }

    // 处理图像
    cv::Mat processed;
    ProcessFrame(input, processed);

    // 显示结果
    cv::imshow("Original Image", input);
    cv::imshow("Processed Image", processed);
    
    std::cout << "Displaying images. Press any key to exit." << std::endl;
    cv::waitKey(0);
    
    cudaDeviceReset();
    return 0;
}
