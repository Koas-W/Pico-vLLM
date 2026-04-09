import torch

def greedy(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (vocab_size,)  只取最后一个 token 的 logits
    return: () 标量，下一个 token 的 id
    """
    return torch.argmax(logits)

def temperature_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    logits: (vocab_size,)
    temperature: 越高越随机，越低越确定，=1.0 时等价于原始分布
    return: () 标量
    """
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze()

def top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """
    logits: (vocab_size,)
    temperature: 先做 temperature 缩放，再做 top-p 截断
    top_p: 累积概率阈值，typical value = 0.9
    return: () 标量
    """
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_index = torch.searchsorted(cumulative_probs, top_p, right=False)
    cutoff_index = min(cutoff_index.item(), len(probs) - 1)  # 确保不越界
    filtered_indices = sorted_indices[:cutoff_index + 1]
    filtered_probs = sorted_probs[:cutoff_index + 1]
    filtered_probs /= filtered_probs.sum()  # 重新归一化
    sampled_index = torch.multinomial(filtered_probs, num_samples=1)
    return filtered_indices[sampled_index].squeeze()

def sample(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
    """
    统一入口：
    - temperature=0 → greedy
    - top_p=1.0     → pure temperature sampling
    - 其他          → top-p sampling
    logits: (vocab_size,)
    return: () 标量
    """
    logits = logits.squeeze()  # 确保始终是 (vocab_size,)
    if temperature == 0:
        return greedy(logits)
    elif top_p >= 1.0:
        return temperature_sample(logits, temperature)
    else:
        return top_p_sample(logits, temperature, top_p)
    

if __name__ == "__main__":
    torch.manual_seed(42)
    logits = torch.randn(151936)

    g = greedy(logits)
    t = temperature_sample(logits, temperature=0.8)
    p = top_p_sample(logits, temperature=0.8, top_p=0.9)
    s0 = sample(logits, temperature=0)
    s1 = sample(logits, temperature=0.8, top_p=1.0)
    s2 = sample(logits, temperature=0.8, top_p=0.9)

    print(f"greedy:      {g.item()}, shape={g.shape}")
    print(f"temperature: {t.item()}, shape={t.shape}")
    print(f"top_p:       {p.item()}, shape={p.shape}")
    print(f"sample(t=0): {s0.item()}, shape={s0.shape}")
    print(f"sample(t=0.8, p=1.0): {s1.item()}, shape={s1.shape}")
    print(f"sample(t=0.8, p=0.9): {s2.item()}, shape={s2.shape}")
    
    # greedy 和 sample(t=0) 应该一致
    assert g.item() == s0.item(), "greedy 和 sample(t=0) 不一致"
    print("通过!")