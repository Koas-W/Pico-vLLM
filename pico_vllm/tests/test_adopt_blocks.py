import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# test_adopt_blocks.py
import torch
from cache import BlockManager, PagedKVCache
from radix_tree import KVCacheRadixTree
from prefix_cache import PrefixCache

BLOCK_SIZE = 16

def test_adopt_blocks_basic():
    bm = BlockManager(
        num_gpu_blocks=50, num_cpu_blocks=0,
        block_size=BLOCK_SIZE, num_layers=4,
        num_kv_heads=2, head_dim=128, dtype=torch.bfloat16,
    )
    cache_kwargs = dict(
        block_manager=bm, num_layers=4, max_seq_len=512,
        num_kv_heads=2, head_dim=128, device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )

    # 模拟 prefix cache 已有 2 个 block
    log = bm.allocate(2)
    phys = [bm.block_mapping[lid][1] for lid in log]

    # 新请求 adopt 这些 block
    new_cache = PagedKVCache(**cache_kwargs)
    new_cache.adopt_blocks(phys, covered_len=32)

    # 验证状态
    assert new_cache.seq_len == 32
    assert new_cache.allocated_cache_block_num == 2
    assert new_cache.physical_block_ids == phys
    assert new_cache.get_block_table().tolist() == phys

    # 继续 prefill：新 token 应该写入新 block
    new_cache._allocate_for_prefill(10)
    assert new_cache.allocated_cache_block_num == 3  # 新加了 1 个 block
    print("✅ test_adopt_blocks_basic")


def test_adopt_then_slot_mapping():
    """adopt 后继续 prefill，slot_mapping 应从正确位置开始"""
    bm = BlockManager(
        num_gpu_blocks=50, num_cpu_blocks=0,
        block_size=BLOCK_SIZE, num_layers=4,
        num_kv_heads=2, head_dim=128, dtype=torch.bfloat16,
    )
    cache_kwargs = dict(
        block_manager=bm, num_layers=4, max_seq_len=512,
        num_kv_heads=2, head_dim=128, device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )

    log = bm.allocate(2)
    phys = [bm.block_mapping[lid][1] for lid in log]

    new_cache = PagedKVCache(**cache_kwargs)
    new_cache.adopt_blocks(phys, covered_len=32)

    # 新 token 10 个，应该从 token 位置 32 开始
    new_cache._allocate_for_prefill(10)
    slots = new_cache.get_prefill_slot_mapping(10)
    # 位置 32 = block 2 的 offset 0
    new_cache._allocate_for_prefill(10)
    slots = new_cache.get_prefill_slot_mapping(10)

    # 第一个 slot 应该在第 3 个 block（block_idx=2）的 offset 0
    new_block = new_cache.physical_block_ids[2]
    expected_first_slot = new_block * BLOCK_SIZE + 0
    assert slots[0].item() == expected_first_slot, \
        f"slot[0] 错误: {slots[0].item()} vs {expected_first_slot}"
    print("✅ test_adopt_then_slot_mapping")


if __name__ == "__main__":
    test_adopt_blocks_basic()
    test_adopt_then_slot_mapping()
    print("🎉")