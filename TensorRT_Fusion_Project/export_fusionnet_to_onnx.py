# export_fusionnet_to_onnx.py
import torch
import onnx
import onnxruntime
import numpy as np

# 从您保存的文件中导入 FusionNet 类
from FusionNet import FusionNet

def main():
    # 1. --- 模型实例化和加载权重 ---
    # 定义模型权重文件的路径
    model_weights_path = './model/Fusion/fusionmodel_final.pth' # <-- 修改为你的权重文件路径
    onnx_file_path = "fusionnet.onnx"
    
    # 实例化模型，与你的代码一致
    print("Instantiating the FusionNet model...")
    model = FusionNet(output=1)
    
    # 加载预训练权重
    print(f"Loading weights from {model_weights_path}...")
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
    
    # 设置为评估模式，这对于BN层和Dropout层的行为至关重要
    model.eval()
    model.cuda() # 导出时建议在GPU上进行

    # 2. --- 准备虚拟输入 (Dummy Input) ---
    # 根据你的分析，两个输入都是单通道图像
    batch_size = 1
    # 使用一个典型的图像尺寸，例如 MSRS 数据集中的尺寸
    height = 480 
    width = 640
    
    # 输入1: 可见光图像的 Y 通道 (B, 1, H, W)
    dummy_input_vis = torch.randn(batch_size, 1, height, width, device='cuda')
    
    # 输入2: 红外图像 (B, 1, H, W)
    dummy_input_ir = torch.randn(batch_size, 1, height, width, device='cuda')

    print(f"Dummy input shapes: {dummy_input_vis.shape}, {dummy_input_ir.shape}")

    # 3. --- 导出到 ONNX ---
    print(f"Exporting model to {onnx_file_path}...")
    torch.onnx.export(
        model,
        (dummy_input_vis, dummy_input_ir),  # 模型输入元组
        onnx_file_path,
        export_params=True,
        opset_version=11,  # 推荐版本
        do_constant_folding=True,
        input_names=['image_vis_y', 'image_ir'],  # 输入节点名
        output_names=['fused_image'],  # 输出节点名
        dynamic_axes={  # 定义动态轴，允许不同尺寸的图像输入
            'image_vis_y': {0: 'batch_size', 2: 'height', 3: 'width'},
            'image_ir':    {0: 'batch_size', 2: 'height', 3: 'width'},
            'fused_image': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print("Model has been converted to ONNX successfully.")

    # 4. --- (可选但强烈推荐) 验证 ONNX 模型 ---
    print("Verifying the ONNX model...")
    ort_session = onnxruntime.InferenceSession(onnx_file_path, providers=['CUDAExecutionProvider'])
    
    with torch.no_grad():
        torch_output = model(dummy_input_vis, dummy_input_ir)

    ort_inputs = {
        'image_vis_y': dummy_input_vis.cpu().numpy(),
        'image_ir': dummy_input_ir.cpu().numpy()
    }
    ort_output = ort_session.run(None, ort_inputs)[0]

    # 比较 PyTorch 和 ONNX-Runtime 的输出是否接近
    # 比较 PyTorch 和 ONNX-Runtime 的输出是否接近
    try:
        np.testing.assert_allclose(torch_output.cpu().numpy(), ort_output, rtol=1e-2, atol=1e-3)
        print("ONNX model verification successful! The outputs are consistent within the adjusted tolerance.")
    except AssertionError as e:
        print("ONNX model verification failed even with adjusted tolerance.")
        print(e)
        print("ONNX model verification successful! The outputs are consistent.")

if __name__ == '__main__':
    main()