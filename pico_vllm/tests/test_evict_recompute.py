import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# test_recompute_eviction.py
"""
测试 Recompute 驱逐策略：
evictable_queue 弹出的 block 直接释放回 free pool，
下次 prompt 命中时重新 prefill。

所有 insert/release/match 使用逻辑 block id（和 Engine 生产路径一致）。
"""
import torch
from cache import BlockManager, PagedKVCache, pagedblocktype
from radix_tree import KVCacheRadixTree
from prefix_cache import PrefixCache

BLOCK_SIZE = 16
NUM_LAYERS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 128


def setup(num_gpu_blocks):
    bm = BlockManager(
        num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=0,
        block_size=BLOCK_SIZE, num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        dtype=torch.bfloat16,
    )
    tree = KVCacheRadixTree(BLOCK_SIZE)
    cache = PrefixCache(tree, bm)
    bm.set_evict_callback(lambda n: len(cache.try_evict(n)))
    return bm, tree, cache


def make_tokens(n, start=0):
    return list(range(start, start + n))


def admit(bm, cache, tokens, start_offset_in_free_pool=None):
    """
    模拟一个请求完成 prefill 的效果：
      1. allocate 足够的 block（allocate 时已 ref=1）
      2. insert 到 prefix cache（新路径的 block +1，现在 ref=2）
    返回请求的 held_blocks（逻辑 id 列表），之后 release 时用。
    """
    n_blocks = len(tokens) // BLOCK_SIZE
    held = bm.allocate(n_blocks)
    cache.insert(tokens, held)
    return held


# ============================================================
# 测试 1：OOM 时触发驱逐（基础）
# ============================================================
def test_01_evict_on_oom():
    bm, tree, cache = setup(num_gpu_blocks=4)

    ta = make_tokens(32)
    held_a = admit(bm, cache, ta)
    cache.release(ta, held_a)

    tb = make_tokens(32, start=100)
    held_b = admit(bm, cache, tb)
    cache.release(tb, held_b)

    assert bm.num_free_blocks == 0
    assert tree.evictable_queue.qsize() == 2

    # 触发驱逐
    held_c = bm.allocate(2)
    assert len(held_c) == 2

    # 至少一个 prefix 被驱逐
    _, la = cache.match(ta)
    _, lb = cache.match(tb)
    # match 会 inc_ref，手动还原一下以免污染后续
    if la > 0:
        cache.release(ta[:la], [])  # 只 dec radix，不 dec block
    if lb > 0:
        cache.release(tb[:lb], [])
    assert (la == 0) or (lb == 0)
    print("✅ test_01_evict_on_oom")


# ============================================================
# 测试 2：LRU 顺序
# ============================================================
def test_02_lru_order():
    bm, tree, cache = setup(num_gpu_blocks=4)

    ta = make_tokens(32)
    held_a = admit(bm, cache, ta)
    cache.release(ta, held_a)

    tb = make_tokens(32, start=100)
    held_b = admit(bm, cache, tb)
    cache.release(tb, held_b)

    _ = bm.allocate(2)  # 驱逐最老的 A

    _, la = cache.match(ta)
    _, lb = cache.match(tb)
    assert la == 0, f"A 应被驱逐，实际 la={la}"
    assert lb == 32, f"B 不应被驱逐，实际 lb={lb}"
    print("✅ test_02_lru_order")


# ============================================================
# 测试 3：活跃 block 不被驱逐
# ============================================================
def test_03_active_not_evicted():
    bm, tree, cache = setup(num_gpu_blocks=4)

    # A 活跃（不 release）
    ta = make_tokens(32)
    held_a = admit(bm, cache, ta)

    # B 可驱逐
    tb = make_tokens(32, start=100)
    held_b = admit(bm, cache, tb)
    cache.release(tb, held_b)

    _ = bm.allocate(2)  # 只能驱逐 B

    _, la = cache.match(ta)
    _, lb = cache.match(tb)
    assert la == 32
    assert lb == 0
    print("✅ test_03_active_not_evicted")


# ============================================================
# 测试 4：全活跃时 OOM
# ============================================================
def test_04_all_active_oom():
    bm, tree, cache = setup(num_gpu_blocks=4)

    ta = make_tokens(32)
    admit(bm, cache, ta)  # 不 release

    tb = make_tokens(32, start=100)
    admit(bm, cache, tb)  # 不 release

    oom = False
    try:
        bm.allocate(2)
    except RuntimeError:
        oom = True
    assert oom, "全活跃时应 OOM"
    print("✅ test_04_all_active_oom")


# ============================================================
# 测试 5：驱逐后重新插入
# ============================================================
def test_05_evict_then_reinsert():
    bm, tree, cache = setup(num_gpu_blocks=4)

    ta = make_tokens(32)
    held_a = admit(bm, cache, ta)
    cache.release(ta, held_a)

    tb = make_tokens(32, start=100)
    held_b = admit(bm, cache, tb)
    cache.release(tb, held_b)

    held_c = bm.allocate(2)  # 驱逐 A

    _, la = cache.match(ta)
    assert la == 0

    # 用 held_c 重新 insert ta
    cache.insert(ta, held_c)

    _, la2 = cache.match(ta)
    assert la2 == 32

    # 清理状态（match 刚 inc 了 ref）
    cache.release(ta[:la2], [])
    print("✅ test_05_evict_then_reinsert")


# ============================================================
# 测试 6：驱逐后 block_manager 的状态完全恢复
# ============================================================
def test_06_evicted_block_returns_to_free_pool():
    bm, tree, cache = setup(num_gpu_blocks=4)

    ta = make_tokens(32)
    held_a = admit(bm, cache, ta)
    cache.release(ta, held_a)
    # 此时 ref_count(held_a[0]) 应为 1（仅 RadixTree 持有）
    for lid in held_a:
        assert bm.logical_ref_count[lid] == 1, \
            f"release 后应剩 RadixTree 持有，实际 ref={bm.logical_ref_count[lid]}"

    tb = make_tokens(32, start=100)
    held_b = admit(bm, cache, tb)
    cache.release(tb, held_b)

    assert bm.num_free_blocks == 0

    # 触发驱逐 A
    held_c = bm.allocate(2)

    # 驱逐后 held_a 的两个逻辑 block 应该：
    # - ref_count 归 0
    # - block_mapping 被重置为 NONE
    # - 或者 block 被 allocate 给了 C（那就是被复用了）
    a_used_by_c = set(held_a) & set(held_c)
    a_released = set(held_a) - set(held_c)

    for lid in a_released:
        btype, pid = bm.block_mapping[lid]
        assert btype == pagedblocktype.NONE, \
            f"被驱逐的逻辑 block {lid} 应为 NONE，实际 {btype}"
        assert bm.logical_ref_count[lid] == 0

    for lid in a_used_by_c:
        # 被 C 复用，应该有新的 GPU 映射和 ref=1
        btype, _ = bm.block_mapping[lid]
        assert btype == pagedblocktype.GPU
        assert bm.logical_ref_count[lid] == 1

    print("✅ test_06_evicted_block_returns_to_free_pool")


# ============================================================
# 测试 7：共享前缀的部分驱逐
# ============================================================
def test_07_shared_prefix_partial_evict():
    """
    A: [0..31]         2 block: s0, s1
    B: [0..15, 100..115] 2 block: 共享 s0 + 独立 b1
    共享段 s0 被 A 和 B 都经过；独立段 s1、b1 各自所有。

    先 release A 和 B，让所有节点都可驱逐。
    驱逐顺序应该按 LRU。
    """
    bm, tree, cache = setup(num_gpu_blocks=4)

    ta = make_tokens(32)
    held_a = admit(bm, cache, ta)
    cache.release(ta, held_a)

    # B match 到 A 的前 16 token
    tb = make_tokens(16) + make_tokens(16, start=100)
    matched, mlen = cache.match(tb)
    assert mlen == 16, f"B 应匹配 A 的第一个 block: mlen={mlen}"
    assert len(matched) == 1

    # B 再 allocate 1 block 放自己的独立段
    own_b = bm.allocate(1)
    held_b = matched + own_b  # B 持有的全部 block（复用 + 新增）
    cache.insert(tb, held_b)
    cache.release(tb, held_b)

    # 此时占用：s0（共享）、s1（A 独占）、b1（B 独占）= 3 block
    assert bm.num_free_blocks == 1

    # match 两个 prompt 都应完整命中
    _, la = cache.peek(ta)
    _, lb = cache.peek(tb)
    # _, la = cache.match(ta); cache.release(ta[:la], [])
    # _, lb = cache.match(tb); cache.release(tb[:lb], [])
    assert la == 32
    assert lb == 32

    # 触发驱逐：需要 3 block，free 1 个，驱逐 2 个
    _ = bm.allocate(3)

    # 此时 cache 里剩的 block ≤ 1
    _, la2 = cache.peek(ta)
    _, lb2 = cache.peek(tb)
    # _, la2 = cache.match(ta); cache.release(ta[:la2], [])
    # _, lb2 = cache.match(tb); cache.release(tb[:lb2], [])
    # 至少其中一个完整被驱逐
    assert la2 < 32 or lb2 < 32
    print(f"  驱逐后 match A={la2}, B={lb2}")
    print("✅ test_07_shared_prefix_partial_evict")


# ============================================================
# 测试 8：lazy deletion —— 节点在队列里时 ref 回升，不被误驱逐
# ============================================================
def test_08_lazy_deletion_rescue():
    """
    A 进 cache -> release（进 evictable_queue）
    B 来 match 到 A（ref 回升）
    -> C allocate 触发驱逐
    -> 不应驱逐 A（虽然 A 还在队列里但已被"救活"）
    """
    bm, tree, cache = setup(num_gpu_blocks=4)

    # A 占 2 block 然后释放
    ta = make_tokens(32)
    held_a = admit(bm, cache, ta)
    cache.release(ta, held_a)
    # A 现在在 evictable_queue 里

    # 再加一个可驱逐的 B 占 1 block
    tb = make_tokens(16, start=100)
    held_b = admit(bm, cache, tb)
    cache.release(tb, held_b)

    assert bm.num_free_blocks == 1

    # 新请求 D match 到 A 的全部（ref 回升）
    matched_a, mlen = cache.match(ta)
    assert mlen == 32
    # 此时 A 的节点 ref=1，不再可驱逐（但还在队列里，属于 stale 条目）

    # 尝试 allocate 2 个 block（free=1，需要从 evictable 驱逐 1 个）
    _ = bm.allocate(2)

    # A 应该还在
    _, la = cache.match(ta); cache.release(ta[:la], [])
    # B 应该被驱逐
    _, lb = cache.match(tb); cache.release(tb[:lb], [])

    assert la == 32, f"A 被 match 过，不应驱逐: la={la}"
    assert lb == 0, f"B 应被驱逐: lb={lb}"

    # 清理 D 的持有
    cache.release(ta, matched_a)
    print("✅ test_08_lazy_deletion_rescue")


# ============================================================
# 测试 9：match 返回的是逻辑 id
# ============================================================
def test_09_match_returns_logical_id():
    bm, tree, cache = setup(num_gpu_blocks=8)

    ta = make_tokens(32)
    held_a = admit(bm, cache, ta)
    cache.release(ta, held_a)

    matched, mlen = cache.match(ta)
    assert mlen == 32
    assert matched == held_a, \
        f"match 应返回逻辑 id，期望 {held_a}，实际 {matched}"
    cache.release(ta[:mlen], [])
    print("✅ test_09_match_returns_logical_id")


# ============================================================
# 测试 10：驱逐一半共享节点，独立段还能访问
# ============================================================
def test_10_merge_after_evict():
    """
    构造三层：root -> s0 -> s1 -> s2
    用三个嵌套 prompt，s0/s1/s2 各为独立节点（被分裂产生）。
    驱逐最深的 s2，触发父节点 merge 检查。
    """
    bm, tree, cache = setup(num_gpu_blocks=8)

    # A: [0..15]        -> 产生节点 N0 (1 block)
    # B: [0..31]        -> 分裂 N0 保持，新建 N1 (1 block，延续)
    # C: [0..47]        -> 继续延伸 N2 (1 block)
    ta = make_tokens(16)
    tb = make_tokens(32)
    tc = make_tokens(48)

    held_a = admit(bm, cache, ta); cache.release(ta, held_a)
    # B match 到 A 的 16 token
    ma, la = cache.match(tb)
    assert la == 16
    own_b = bm.allocate(1)
    held_b = ma + own_b
    cache.insert(tb, held_b); cache.release(tb, held_b)

    # C match 到 B 的 32 token
    mb, lb = cache.match(tc)
    assert lb == 32
    own_c = bm.allocate(1)
    held_c = mb + own_c
    cache.insert(tc, held_c); cache.release(tc, held_c)

    # 占用 3 block，free 5
    assert bm.num_free_blocks == 5

    # 触发驱逐：需要 6 block，free 5，必须驱逐 1
    # 按 LRU，A 最早 release，但 A 的节点不是叶子（有 B 延续）；
    # B 也不是叶子（有 C 延续）；只有 C 是叶子，最先被驱逐。
    _ = bm.allocate(6)

    # A、B 还应能 match
    _, la2 = cache.match(ta); cache.release(ta[:la2], [])
    _, lb2 = cache.match(tb); cache.release(tb[:lb2], [])
    _, lc2 = cache.match(tc); cache.release(tc[:lc2], [])

    assert lc2 < 48, f"C 的尾部应被驱逐: lc2={lc2}"
    print(f"  驱逐后 match A={la2}, B={lb2}, C={lc2}")
    print("✅ test_10_merge_after_evict")


# ============================================================
# 测试 11：完整一轮 recompute 语义：驱逐 -> 下次 miss -> 重算
# ============================================================
def test_11_recompute_roundtrip():
    """验证 recompute 策略的语义闭环"""
    bm, tree, cache = setup(num_gpu_blocks=4)

    ta = make_tokens(32)
    held_a = admit(bm, cache, ta)
    cache.release(ta, held_a)

    tb = make_tokens(32, start=100)
    held_b = admit(bm, cache, tb)
    cache.release(tb, held_b)

    # 驱逐 A
    held_c = bm.allocate(2)

    # 1. A 现在 miss（recompute 必要条件）
    matched, mlen = cache.match(ta)
    assert mlen == 0 and matched == []
    print(f"  驱逐后 match miss -> 触发重算")

    # 2. 假设 Engine 重新 prefill 了 A（重新 alloc + insert）
    #    这里没有真实 prefill，只模拟 block 分配 + insert
    #    但现在 free pool 没 block 了，要再驱逐
    held_a_new = bm.allocate(2)  # 驱逐 B
    cache.insert(ta, held_a_new)
    cache.release(ta, held_a_new)

    # 3. 之后 A 又能命中
    _, la = cache.match(ta); cache.release(ta[:la], [])
    assert la == 32
    print(f"  重算后 match hit")
    print("✅ test_11_recompute_roundtrip")


if __name__ == "__main__":
    test_01_evict_on_oom()
    test_02_lru_order()
    test_03_active_not_evicted()
    test_04_all_active_oom()
    test_05_evict_then_reinsert()
    test_06_evicted_block_returns_to_free_pool()
    test_07_shared_prefix_partial_evict()
    test_08_lazy_deletion_rescue()
    test_09_match_returns_logical_id()
    test_10_merge_after_evict()
    test_11_recompute_roundtrip()
    print("\n🎉 全部 Recompute 驱逐测试通过！")