import torch
from safetensors import safe_open
import os
import glob

# 找到所有权重文件
weight_dir = "./weights"
safetensor_files = glob.glob(os.path.join(weight_dir, "*.safetensors"))

print("=" * 60)
print("权重文件:")
print("=" * 60)

all_tensors = {}
for f in sorted(safetensor_files):
    print(f"\n{os.path.basename(f)}")
    with safe_open(f, framework="pt", device="cpu") as st:
        for key in st.keys():
            tensor = st.get_tensor(key)
            all_tensors[key] = tensor.shape
            print(f"  {key:<60} {str(tensor.shape):<30} {tensor.dtype}")

print("\n" + "=" * 60)
print(f"总计 {len(all_tensors)} 个 tensor")

# 统计参数量
total_params = 0
for f in sorted(safetensor_files):
    with safe_open(f, framework="pt", device="cpu") as st:
        for key in st.keys():
            t = st.get_tensor(key)
            total_params += t.numel()
print(f"总参数量: {total_params / 1e9:.2f}B")

# 打印 config
import json
config_path = os.path.join(weight_dir, "config.json")
if os.path.exists(config_path):
    print("\n" + "=" * 60)
    print("config.json:")
    print("=" * 60)
    with open(config_path) as f:
        config = json.load(f)
    for k, v in config.items():
        print(f"  {k}: {v}")