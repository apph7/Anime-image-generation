import torch

print("=" * 50)
print("安装验证:")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"显卡型号: {torch.cuda.get_device_name(0)}")

# 测试 GPU 计算
if torch.cuda.is_available():
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = torch.tensor([4.0, 5.0, 6.0]).cuda()
    z = x + y
    print(f"GPU 计算测试: {z}")
    print("GPU 加速正常工作！")
else:
    print("GPU 加速未启用")
print("=" * 50)