# weights.py
import torch
from safetensors import safe_open
import glob
import os

def load_weights(model, weight_dir: str, tp_size: int = 1, rank: int = 0):
    """从 safetensors 文件加载权重到 model"""
    
    safetensor_files = sorted(glob.glob(os.path.join(weight_dir, "*.safetensors")))
    state_dict = {}
    for f in safetensor_files:
        with safe_open(f, framework="pt", device="cpu") as st:
            for key in st.keys():
                state_dict[key] = st.get_tensor(key).float()

    # embed_tokens 和 final norm：每卡完整副本，不切
    model.embed_tokens.weight.data = state_dict["model.embed_tokens.weight"]
    model.norm.weight.data = state_dict["model.norm.weight"]

    for i in range(model.cfg.num_hidden_layers):
        layer = model.layers[i]
        p = f"model.layers.{i}"

        # === Attention ===
        # QKV：列切（按 output dim 切，每卡分到一部分 head）
        q_w = state_dict[f"{p}.self_attn.q_proj.weight"]  # (q_size, hidden)
        k_w = state_dict[f"{p}.self_attn.k_proj.weight"]  # (kv_size, hidden)
        v_w = state_dict[f"{p}.self_attn.v_proj.weight"]  # (kv_size, hidden)
        q_b = state_dict[f"{p}.self_attn.q_proj.bias"]
        k_b = state_dict[f"{p}.self_attn.k_proj.bias"]
        v_b = state_dict[f"{p}.self_attn.v_proj.bias"]

        if tp_size > 1:
            q_w = q_w.chunk(tp_size, dim=0)[rank]
            k_w = k_w.chunk(tp_size, dim=0)[rank]
            v_w = v_w.chunk(tp_size, dim=0)[rank]
            q_b = q_b.chunk(tp_size, dim=0)[rank]
            k_b = k_b.chunk(tp_size, dim=0)[rank]
            v_b = v_b.chunk(tp_size, dim=0)[rank]

        layer.attn.qkv_proj.weight.data = torch.cat([q_w, k_w, v_w], dim=0)
        layer.attn.qkv_proj.bias.data = torch.cat([q_b, k_b, v_b], dim=0)

        # O proj：行切（按 input dim 切，每卡拿到部分输入，forward 后 All-Reduce）
        o_w = state_dict[f"{p}.self_attn.o_proj.weight"]  # (hidden, q_size)
        if tp_size > 1:
            o_w = o_w.chunk(tp_size, dim=1)[rank]  # 注意是 dim=1（input dim）
        layer.attn.o_proj.weight.data = o_w

        # === FFN ===
        # gate_up：列切
        gate_w = state_dict[f"{p}.mlp.gate_proj.weight"]  # (intermediate, hidden)
        up_w = state_dict[f"{p}.mlp.up_proj.weight"]      # (intermediate, hidden)
        if tp_size > 1:
            gate_w = gate_w.chunk(tp_size, dim=0)[rank]
            up_w = up_w.chunk(tp_size, dim=0)[rank]
        layer.ffn.gate_up_proj.weight.data = torch.cat([gate_w, up_w], dim=0)

        # down：行切
        down_w = state_dict[f"{p}.mlp.down_proj.weight"]  # (hidden, intermediate)
        if tp_size > 1:
            down_w = down_w.chunk(tp_size, dim=1)[rank]  # dim=1（input dim）
        layer.ffn.down_proj.weight.data = down_w

        # RMSNorm：每卡完整副本，不切
        layer.norm1.weight.data = state_dict[f"{p}.input_layernorm.weight"]
        layer.norm2.weight.data = state_dict[f"{p}.post_attention_layernorm.weight"]

    if rank == 0:
        print(f"权重加载完成，共 {len(state_dict)} 个 tensor, tp_size={tp_size}")
    return model

# 在 weights.py 末尾测试
if __name__ == "__main__":
    from model import Qwen25_15B, ModelConfig
    cfg = ModelConfig()
    model = Qwen25_15B(cfg)
    model = load_weights(model, "./weights")
    print("加载成功")