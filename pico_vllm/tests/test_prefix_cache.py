import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# test_prefix_cache.py
import torch
from cache import BlockManager
from radix_tree import KVCacheRadixTree
from prefix_cache import PrefixCache

BLOCK_SIZE = 16

def make_tokens(n, start=0):
    return list(range(start, start + n))

def setup():
    """创建一套 BlockManager + RadixTree + PrefixCache"""
    bm = BlockManager(
        num_gpu_blocks=50, num_cpu_blocks=0,
        block_size=BLOCK_SIZE, num_layers=4,
        num_kv_heads=2, head_dim=128, dtype=torch.bfloat16,
    )
    tree = KVCacheRadixTree(BLOCK_SIZE)
    cache = PrefixCache(tree, bm)
    return bm, tree, cache


def test_1_match_miss_then_insert():
    """空 cache 匹配失败 → prefill 后 insert → 下次能匹配"""
    bm, tree, cache = setup()

    tokens = make_tokens(32)
    blocks, length = cache.match(tokens)
    assert blocks == [] and length == 0

    # 模拟 prefill 分配了两个 block
    logical = bm.allocate(2)
    physical = [bm.block_mapping[lid][1] for lid in logical]
    # 新分配的 block ref_count=1
    assert all(bm.gpu_block_ref_count[p] == 1 for p in physical)

    # 插入 prefix cache
    cache.insert(tokens, physical)
    # 插入后 RadixTree 持有这些 block，ref_count=2
    assert all(bm.gpu_block_ref_count[p] == 2 for p in physical)

    # 再次 match，命中
    blocks, length = cache.match(tokens)
    assert blocks == physical
    assert length == 32
    # match 后又有一个"请求"持有，ref=3
    # print(bm.gpu_block_ref_count[0:2])
    assert all(bm.gpu_block_ref_count[p] == 3 for p in physical)
    print("✅ test_1_match_miss_then_insert")
    
def test_2_release_decreases_ref():
    bm, tree, cache = setup()
    tokens = make_tokens(32)

    logical = bm.allocate(2)
    physical = [bm.block_mapping[lid][1] for lid in logical]
    cache.insert(tokens, physical)
    # ref=2（prefill 请求 + RadixTree）

    # 模拟另一个请求 match 并随后 release
    blocks, _ = cache.match(tokens)
    # ref=3（+ 新请求）
    cache.release(tokens)
    # ref=2（新请求释放）

    # 原 prefill 请求也结束
    cache.release(tokens)
    # ref=1（只剩 RadixTree）

    assert all(bm.gpu_block_ref_count[p] == 1 for p in physical)
    print("✅ test_2_release_decreases_ref")

def test_3_shared_prefix_ref_counting():
    bm, tree, cache = setup()

    # 请求 A: [0..31]
    tokens_a = make_tokens(32)
    log_a = bm.allocate(2)
    phys_a = [bm.block_mapping[lid][1] for lid in log_a]
    cache.insert(tokens_a, phys_a)
    # phys_a[0].ref=2, phys_a[1].ref=2 （A + RadixTree）

    # 请求 B: [0..15, 100..115]
    tokens_b = make_tokens(16) + make_tokens(16, start=100)

    # B match 命中前缀
    blocks_matched, length = cache.match(tokens_b)
    assert blocks_matched == [phys_a[0]]
    # phys_a[0].ref=3（A + RadixTree + B）

    # B prefill 新 block
    log_b = bm.allocate(1)
    phys_b = [bm.block_mapping[lid][1] for lid in log_b]
    # phys_b[0].ref=1（B）

    # B 完成，insert
    cache.insert(tokens_b, [phys_a[0], phys_b[0]])
    # RadixTree 新持有 phys_b[0]
    # phys_b[0].ref=2（B + RadixTree）
    # phys_a[0].ref=3 不变
    assert bm.gpu_block_ref_count[phys_b[0]] == 2

    # A 结束
    cache.release(tokens_a)
    # A 释放对 phys_a[0] 和 phys_a[1] 的持有
    # phys_a[0].ref=2（RadixTree + B）
    # phys_a[1].ref=1（RadixTree）

    # B 结束
    cache.release(tokens_b)
    # B 释放对 phys_a[0] 和 phys_b[0] 的持有
    # phys_a[0].ref=1（RadixTree）
    # phys_b[0].ref=1（RadixTree）

    assert bm.gpu_block_ref_count[phys_a[0]] == 1
    assert bm.gpu_block_ref_count[phys_a[1]] == 1
    assert bm.gpu_block_ref_count[phys_b[0]] == 1
    print("✅ test_3_shared_prefix_ref_counting")

def test_4_eviction_releases_blocks():
    bm, tree, cache = setup()

    tokens = make_tokens(32)
    log = bm.allocate(2)
    phys = [bm.block_mapping[lid][1] for lid in log]
    cache.insert(tokens, phys)
    # phys[0].ref=2, phys[1].ref=2（prefill + RadixTree）
    # 节点 ref_count = 1（insert 时设）

    # prefill 请求结束，走 release（同时减 BlockManager 和 RadixTree）
    cache.release(tokens)
    # phys[0].ref=1, phys[1].ref=1（只剩 RadixTree）
    # 节点 ref_count = 0（进入 evictable 队列）

    # 触发 evict
    evicted = cache.try_evict(1)
    assert len(evicted) >= 1, f"evict 应返回至少 1 个 block: {evicted}"

    # evict 后 RadixTree 也不再持有，ref 降到 0
    for bid in evicted:
        assert bm.gpu_block_ref_count[bid] == 0, \
            f"evict 后 ref 应为 0: {bid}={bm.gpu_block_ref_count[bid]}"
    print("✅ test_4_eviction_releases_blocks")

def test_5_idempotent_insert():
    """重复 insert 相同内容，ref 不应无限增加"""
    bm, tree, cache = setup()

    tokens = make_tokens(32)
    log = bm.allocate(2)
    phys = [bm.block_mapping[lid][1] for lid in log]

    cache.insert(tokens, phys)
    ref_after_first = [bm.gpu_block_ref_count[p] for p in phys]
    # ref = 2（prefill + RadixTree）

    # 再 insert 一次（幂等）
    cache.insert(tokens, phys)
    ref_after_second = [bm.gpu_block_ref_count[p] for p in phys]

    # block 级 ref 不应变化——RadixTree 没新增持有
    # RadixTree 内部节点的 ref_count 增加了（+1），但那是 RadixTree 内部事务
    assert ref_after_first == ref_after_second, \
        f"幂等 insert 不应改变 block ref: {ref_after_first} vs {ref_after_second}"
    print("✅ test_5_idempotent_insert")


def test_6_shared_block_not_evicted():
    """有引用的 block 不会被 evict"""
    bm, tree, cache = setup()

    tokens = make_tokens(32)
    log = bm.allocate(2)
    phys = [bm.block_mapping[lid][1] for lid in log]
    cache.insert(tokens, phys)

    # 保持 prefill 请求的 ref 不释放
    # RadixTree 节点 ref_count > 0（因为 insert 时初始化为 1）

    # 此时尝试 evict
    evicted = cache.try_evict(2)

    # 应该什么都不能 evict（节点 ref_count > 0）
    assert evicted == [], f"有引用时不应 evict: {evicted}"
    # block ref 不变
    assert all(bm.gpu_block_ref_count[p] == 2 for p in phys)
    print("✅ test_6_shared_block_not_evicted")


def test_7_partial_block_not_cached():
    """不足一个 block 的 prefix 不缓存"""
    bm, tree, cache = setup()

    # 只有 16 个 token（正好一个 block）
    tokens = make_tokens(16)
    log = bm.allocate(1)
    phys = [bm.block_mapping[lid][1] for lid in log]
    cache.insert(tokens, phys)

    # match 8 个 token（不足一个 block）
    blocks, length = cache.match(make_tokens(8))
    assert blocks == [], f"不足一个 block 不应命中: {blocks}"
    assert length == 0
    print("✅ test_7_partial_block_not_cached")


if __name__ == "__main__":
    test_1_match_miss_then_insert()
    test_2_release_decreases_ref()
    test_3_shared_prefix_ref_counting()
    test_4_eviction_releases_blocks()
    test_5_idempotent_insert()
    test_6_shared_block_not_evicted()
    test_7_partial_block_not_cached()
    print("\n🎉 全部测试通过！")