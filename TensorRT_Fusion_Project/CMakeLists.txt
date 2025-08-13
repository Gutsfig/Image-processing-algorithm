cmake_minimum_required(VERSION 3.15)
project(TensorRT_Fusion CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 关键路径设置 ---
# 请根据你的实际安装路径修改
set(OpenCV_DIR "D:/path/to/your/opencv/build")  # <-- 确认这个路径正确
set(TensorRT_DIR "D:/Tools/TensorRT-8.6.1.6")   # <-- 确认这个路径正确
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8") # <-- 已确认这个路径正确

# --- 查找 OpenCV ---
find_package(OpenCV REQUIRED)

# --- 手动配置 TensorRT 和 CUDA 路径 (最直接的方式) ---
# 头文件目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "${TensorRT_DIR}/include"
    "${CUDA_TOOLKIT_ROOT_DIR}/include"
)

# 库文件目录
link_directories(
    ${OpenCV_LIBRARY_DIRS}
    "${TensorRT_DIR}/lib"
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"  # <<< 关键修改：直接指定到 x64 目录
)


# --- 创建可执行文件 ---
add_executable(fusion_demo main.cpp)

# --- 链接具体的 .lib 文件 ---
target_link_libraries(fusion_demo PRIVATE
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart.lib  # <<< 关键修改：明确链接 cudart.lib (不是 cudart_static)
    # 如果还有其他库找不到，比如 cublas.lib，就在这里加上
)

# --- 拷贝 DLL 到生成目录 (方便运行) ---
add_custom_command(TARGET fusion_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${TensorRT_DIR}/lib/nvinfer.dll"
    $<TARGET_FILE_DIR:fusion_demo>
)
add_custom_command(TARGET fusion_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${TensorRT_DIR}/lib/nvinfer_plugin.dll"
    $<TARGET_FILE_DIR:fusion_demo>
)
# 添加 CUDA DLL 的拷贝，以防万一
add_custom_command(TARGET fusion_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_110.dll" # 注意版本号，11.8 对应的是 110
    $<TARGET_FILE_DIR:fusion_demo>
)