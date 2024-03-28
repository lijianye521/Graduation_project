import torch
 
# 检查CUDA是否可用
print("CUDA available:", torch.cuda.is_available())
 
# 获取CUDA设备数量
print("Number of CUDA devices:", torch.cuda.device_count())
 
# 获取CUDA设备信息
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
 
# 检查当前使用的设备是否是CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)