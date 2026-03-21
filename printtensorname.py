from safetensors import safe_open
import glob, os

weight_dir = "./weights"
safetensor_files = glob.glob(os.path.join(weight_dir, "*.safetensors"))

with safe_open(safetensor_files[0], framework="pt", device="cpu") as st:
    keys = sorted(st.keys())

# 只打印第0层，看清楚命名结构
for k in keys:
    if "layers.0" in k or "embed" in k or "norm" in k or "lm_head" in k:
        print(k)