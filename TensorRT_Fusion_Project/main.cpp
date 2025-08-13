#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <filesystem> // <<< ADDED: 用于文件系统操作
#include <algorithm>  // <<< ADDED: 用于排序

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"

namespace fs = std::filesystem; // 为filesystem定义一个命名空间别名

// Logger class (no changes)
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

// printEngineInfo function (no changes)
void printEngineInfo(const std::unique_ptr<nvinfer1::ICudaEngine>& engine) {
    std::cout << "Engine Info:" << std::endl;
    for (int i = 0; i < engine->getNbBindings(); ++i) {
        std::cout << " - Binding " << i << ": " << engine->getBindingName(i)
                  << ", IsInput: " << (engine->bindingIsInput(i) ? "Yes" : "No")
                  << ", Dims: (";
        auto dims = engine->getBindingDimensions(i);
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j] << (j < dims.nbDims - 1 ? ", " : "");
        }
        std::cout << ")" << std::endl;
    }
}


// 将单次推理逻辑封装成一个函数
void fuse_single_pair(
    const std::string& vis_path, 
    const std::string& ir_path, 
    const std::string& save_path,
    nvinfer1::IExecutionContext* context, 
    const std::unique_ptr<nvinfer1::ICudaEngine>& engine) 
{
    // --- 加载图像 ---
    cv::Mat img_vis_bgr = cv::imread(vis_path);
    cv::Mat img_ir_gray = cv::imread(ir_path, cv::IMREAD_GRAYSCALE);

    if (img_vis_bgr.empty() || img_ir_gray.empty()) {
        std::cerr << "Failed to load images: " << vis_path << " or " << ir_path << std::endl;
        return;
    }

    const int input_h = img_vis_bgr.rows;
    const int input_w = img_vis_bgr.cols;
    
    // 确保两张图尺寸一致
    if (img_ir_gray.rows != input_h || img_ir_gray.cols != input_w) {
        cv::resize(img_ir_gray, img_ir_gray, cv::Size(input_w, input_h));
    }

    // --- 预处理 ---
    cv::Mat img_vis_ycrcb;
    cv::cvtColor(img_vis_bgr, img_vis_ycrcb, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> vis_channels;
    cv::split(img_vis_ycrcb, vis_channels);
    cv::Mat img_vis_y;
    vis_channels[0].convertTo(img_vis_y, CV_32F, 1.0 / 255.0);

    cv::Mat img_ir_float;
    img_ir_gray.convertTo(img_ir_float, CV_32F, 1.0 / 255.0);
    
    // --- 分配GPU内存和执行推理 ---
    void* buffers[3];
    const int vis_input_idx = engine->getBindingIndex("image_vis_y");
    const int ir_input_idx = engine->getBindingIndex("image_ir");
    const int output_idx = engine->getBindingIndex("fused_image");
    
    const size_t buffer_size = static_cast<size_t>(1) * input_h * input_w * sizeof(float);
    cudaMalloc(&buffers[vis_input_idx], buffer_size);
    cudaMalloc(&buffers[ir_input_idx], buffer_size);
    cudaMalloc(&buffers[output_idx], buffer_size);

    cudaMemcpy(buffers[vis_input_idx], img_vis_y.ptr<float>(), buffer_size, cudaMemcpyHostToDevice);
    cudaMemcpy(buffers[ir_input_idx], img_ir_float.ptr<float>(), buffer_size, cudaMemcpyHostToDevice);
    
    context->setBindingDimensions(vis_input_idx, nvinfer1::Dims4{1, 1, input_h, input_w});
    context->setBindingDimensions(ir_input_idx, nvinfer1::Dims4{1, 1, input_h, input_w});
    
    context->enqueueV2(buffers, 0, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> output_data(1 * input_h * input_w);
    cudaMemcpy(output_data.data(), buffers[output_idx], buffer_size, cudaMemcpyDeviceToHost);

    // --- 后处理和保存 ---
    cv::Mat fused_y(input_h, input_w, CV_32F, output_data.data());
    fused_y.convertTo(fused_y, CV_8U, 255.0);
    vis_channels[0] = fused_y;
    cv::Mat fused_ycrcb, fused_bgr;
    cv::merge(vis_channels, fused_ycrcb);
    cv::cvtColor(fused_ycrcb, fused_bgr, cv::COLOR_YCrCb2BGR);
    // 关键修改：将 [0, 1] 的输出映射到 [16, 235]

    // --- 后处理和保存算法2 ----------------------------------------------------
    // cv::Mat fused_y_float(input_h, input_w, CV_32F, output_data.data());
    // // 公式：newValue = 16 + oldValue * (235 - 16)
    // cv::Mat fused_y_scaled;
    // fused_y_float.convertTo(fused_y_scaled, CV_32F, 219.0, 16.0); // 等价于 fused_y_float * 219 + 16

    // cv::Mat fused_y_u8;
    // fused_y_scaled.convertTo(fused_y_u8, CV_8U); // 直接转换为 CV_8U，不需要再乘

    // // 将处理好的 Y 通道放回去
    // vis_channels[0] = fused_y_u8;
    // cv::Mat fused_ycrcb, fused_bgr;
    // cv::merge(vis_channels, fused_ycrcb);
    // cv::cvtColor(fused_ycrcb, fused_bgr, cv::COLOR_YCrCb2BGR);

    ////////////////////////////////////////////////////////////

    cv::imwrite(save_path, fused_bgr);
    std::cout << "  -> Saved to " << save_path << std::endl;

    // --- 释放内存 ---
    cudaFree(buffers[vis_input_idx]);
    cudaFree(buffers[ir_input_idx]);
    cudaFree(buffers[output_idx]);
}


int main() {
    Logger gLogger;

    // 1. 初始化 TensorRT 引擎 (只执行一次)
    std::cout << "Initializing TensorRT engine..." << std::endl;
    std::string engine_path = "D:/project/py392/TensorRT_Fusion_Project/model/fusionnet_large.engine"; 
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
        std::cerr << "Error opening engine file: " << engine_path << std::endl;
        return -1;
    }
    engine_file.seekg(0, std::ios::end);
    long int fsize = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(fsize);
    engine_file.read(engine_data.data(), fsize);
    engine_file.close();

    // 反序列化引擎
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};
    std::unique_ptr<nvinfer1::ICudaEngine> engine{runtime->deserializeCudaEngine(engine_data.data(), fsize)};
    std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
    if (!engine || !context) {
        std::cerr << "Failed to create engine or context" << std::endl;
        return -1;
    }
    std::cout << "Engine initialized successfully." << std::endl;
    printEngineInfo(engine);

    // 2. 定义输入输出文件夹路径
    std::string vis_folder = "D:/project/py392/TensorRT_Fusion_Project/test/vis/";
    std::string ir_folder = "D:/project/py392/TensorRT_Fusion_Project/test/ir/";
    std::string save_folder = "D:/project/py392/TensorRT_Fusion_Project/results/";

    // 确保保存目录存在
    fs::create_directories(save_folder);

    // 3. 遍历可见光图像文件夹，寻找配对的红外图像并处理
    std::cout << "\nStarting batch fusion process..." << std::endl;
    std::vector<fs::path> vis_files;
    for (const auto& entry : fs::directory_iterator(vis_folder)) {
        if (entry.is_regular_file()) {
            vis_files.push_back(entry.path());
        }
    }
    // 对文件进行排序，确保处理顺序一致
    std::sort(vis_files.begin(), vis_files.end());

    int count = 0;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (const auto& vis_path : vis_files) {
        std::string filename = vis_path.filename().string();
        fs::path ir_path = fs::path(ir_folder) / filename;
        fs::path save_path = fs::path(save_folder) / filename;

        if (fs::exists(ir_path)) {
            std::cout << "Processing pair " << ++count << ": " << filename << std::endl;
            auto pair_start = std::chrono::high_resolution_clock::now();
            
            fuse_single_pair(vis_path.string(), ir_path.string(), save_path.string(), context.get(), engine);
            
            auto pair_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> pair_elapsed = pair_end - pair_start;
            std::cout << "  -> Time taken: " << pair_elapsed.count() << " ms\n";
        } else {
            std::cerr << "Warning: Corresponding IR image not found for " << vis_path.string() << std::endl;
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;
    std::cout << "\nBatch fusion finished. Processed " << count << " pairs in " 
              << total_elapsed.count() << " seconds." << std::endl;

    return 0;
}