import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# test_radix_tree.py
from radix_tree import KVCacheRadixTree

BLOCK_SIZE = 16

def make_tokens(n):
    """生成 n 个 token（block_size 对齐）"""
    return list(range(n))

def test_basic_insert_match():
    """基本 insert + 完全匹配"""
    tree = KVCacheRadixTree(BLOCK_SIZE)
    tokens = make_tokens(32)
    blocks = [10, 20]

    tree.insert(tokens, blocks)
    matched_blocks, matched_len = tree.match(tokens)

    assert matched_blocks == [10, 20], f"blocks 错误: {matched_blocks}"
    assert matched_len == 32, f"matched_len 错误: {matched_len}"
    print("✅ test_basic_insert_match")

def test_prefix_match():
    """查询比插入的长，返回前缀匹配"""
    tree = KVCacheRadixTree(BLOCK_SIZE)
    tokens = make_tokens(32)
    blocks = [10, 20]

    tree.insert(tokens, blocks)

    query = make_tokens(48)  # 比插入的长 16
    matched_blocks, matched_len = tree.match(query)

    assert matched_blocks == [10, 20], f"blocks 错误: {matched_blocks}"
    assert matched_len == 32, f"matched_len 错误: {matched_len}"
    print("✅ test_prefix_match")

def test_no_match():
    """完全不匹配"""
    tree = KVCacheRadixTree(BLOCK_SIZE)
    tokens = make_tokens(32)
    blocks = [10, 20]

    tree.insert(tokens, blocks)

    query = [999, 998, 997] + list(range(29))
    matched_blocks, matched_len = tree.match(query)

    assert matched_blocks == [], f"不应该匹配: {matched_blocks}"
    assert matched_len == 0, f"matched_len 应为 0: {matched_len}"
    print("✅ test_no_match")

def test_split():
    """两段序列有公共前缀，第二次 insert 触发分裂"""
    tree = KVCacheRadixTree(BLOCK_SIZE)

    # 插入 A: [0,1,...,31]
    tokens_a = make_tokens(32)
    blocks_a = [10, 20]
    tree.insert(tokens_a, blocks_a)

    # 插入 B: [0,1,...,15, 100,101,...,115]
    # 前 16 个 token 共享，后 16 个不同
    tokens_b = list(range(16)) + list(range(100, 116))
    blocks_b = [10, 30]
    tree.insert(tokens_b, blocks_b)

    # 匹配 A：应该完整匹配
    matched_a, len_a = tree.match(tokens_a)
    assert matched_a == [10, 20], f"A 匹配错误: {matched_a}"
    assert len_a == 32, f"A matched_len 错误: {len_a}"

    # 匹配 B：应该完整匹配
    matched_b, len_b = tree.match(tokens_b)
    assert matched_b == [10, 30], f"B 匹配错误: {matched_b}"
    assert len_b == 32, f"B matched_len 错误: {len_b}"

    # 匹配公共前缀：只匹配前 16 个
    prefix = make_tokens(16)
    matched_p, len_p = tree.match(prefix)
    assert matched_p == [10], f"前缀匹配错误: {matched_p}"
    assert len_p == 16, f"前缀 matched_len 错误: {len_p}"

    print("✅ test_split")

def test_multi_split():
    """三段序列逐步分裂"""
    tree = KVCacheRadixTree(BLOCK_SIZE)

    # A: [0..31] → blocks [10, 20]
    tokens_a = make_tokens(32)
    tree.insert(tokens_a, [10, 20])

    # B: [0..15, 100..115] → blocks [10, 30]
    tokens_b = list(range(16)) + list(range(100, 116))
    tree.insert(tokens_b, [10, 30])

    # C: [0..15, 100..107, 200..207] → blocks [10, 40]
    # 和 B 共享前 16+8=24 个 token，但 24 不是 block 边界
    # 只能匹配到 16 个 token（1 个完整 block）
    tokens_c = list(range(16)) + list(range(100, 108)) + list(range(200, 208))
    tree.insert(tokens_c, [10, 40])

    # 验证各自匹配
    ma, la = tree.match(tokens_a)
    assert la == 32 and ma == [10, 20], f"A: {ma}, {la}"

    mb, lb = tree.match(tokens_b)
    assert lb == 32 and mb == [10, 30], f"B: {mb}, {lb}"

    mc, lc = tree.match(tokens_c)
    assert lc == 32 and mc == [10, 40], f"C: {mc}, {lc}"

    print("✅ test_multi_split")

def test_idempotent_insert():
    """相同内容 insert 两次不出错"""
    tree = KVCacheRadixTree(BLOCK_SIZE)
    tokens = make_tokens(32)
    blocks = [10, 20]

    tree.insert(tokens, blocks)
    tree.insert(tokens, blocks)  # 第二次

    matched, length = tree.match(tokens)
    assert matched == [10, 20], f"幂等失败: {matched}"
    assert length == 32
    print("✅ test_idempotent_insert")

def test_ref_count_inc_dec():
    """引用计数的增减"""
    tree = KVCacheRadixTree(BLOCK_SIZE)
    tokens = make_tokens(32)
    tree.insert(tokens, [10, 20])

    # insert 时 ref_count 应该是 1
    node = tree.root.children[0]
    initial_ref = node.ref_count
    assert initial_ref >= 1, f"初始 ref_count 应 >= 1: {initial_ref}"

    # inc_ref
    tree.inc_ref(tokens)
    assert node.ref_count == initial_ref + 1, f"inc_ref 后应为 {initial_ref + 1}: {node.ref_count}"

    # dec_ref
    tree.dec_ref(tokens)
    assert node.ref_count == initial_ref, f"dec_ref 后应恢复: {node.ref_count}"

    print("✅ test_ref_count_inc_dec")

def test_ref_count_shared_prefix():
    """共享前缀的引用计数"""
    tree = KVCacheRadixTree(BLOCK_SIZE)

    tokens_a = make_tokens(32)
    tree.insert(tokens_a, [10, 20])

    tokens_b = list(range(16)) + list(range(100, 116))
    tree.insert(tokens_b, [10, 30])

    # 公共前缀节点（前 16 token）的 ref_count 应该 >= 2
    prefix_node = tree.root.children[0]
    assert prefix_node.ref_count >= 2, f"共享节点 ref_count 应 >= 2: {prefix_node.ref_count}"

    # A 结束，dec_ref
    tree.dec_ref(tokens_a)
    assert prefix_node.ref_count >= 1, f"A 结束后共享节点仍应 >= 1: {prefix_node.ref_count}"

    # B 结束，dec_ref
    tree.dec_ref(tokens_b)
    assert prefix_node.ref_count == 0, f"都结束后应为 0: {prefix_node.ref_count}"

    print("✅ test_ref_count_shared_prefix")

def test_evict_basic():
    """基本驱逐：ref_count=0 的叶子节点可以被驱逐"""
    tree = KVCacheRadixTree(BLOCK_SIZE)
    tokens = make_tokens(32)
    tree.insert(tokens, [10, 20])

    # dec_ref 使其可驱逐
    tree.dec_ref(tokens)

    # 驱逐
    evicted = tree.evict(1)
    assert len(evicted) > 0, "应该驱逐了至少一个 block"
    print(f"  驱逐了 {evicted}")

    # 驱逐后 match 应该减少
    matched, length = tree.match(tokens)
    assert length < 32, f"驱逐后匹配长度应减少: {length}"

    print("✅ test_evict_basic")

def test_evict_respects_ref_count():
    """有引用的节点不会被驱逐"""
    tree = KVCacheRadixTree(BLOCK_SIZE)

    tokens_a = make_tokens(32)
    tree.insert(tokens_a, [10, 20])

    tokens_b = list(range(16)) + list(range(100, 116))
    tree.insert(tokens_b, [10, 30])

    # 只释放 A，B 仍在使用
    tree.dec_ref(tokens_a)

    evicted = tree.evict(10)  # 尝试驱逐很多

    # 公共前缀 block 10 不应该被驱逐（B 还在用）
    assert 10 not in evicted, f"共享 block 不应被驱逐: {evicted}"

    # A 独有的 block 20 可以被驱逐
    # （取决于树结构，20 可能在叶子节点上）

    print(f"  驱逐了 {evicted}")
    print("✅ test_evict_respects_ref_count")

def test_delete_and_merge():
    """删除叶子节点后，父节点只剩一个孩子时合并"""
    tree = KVCacheRadixTree(BLOCK_SIZE)

    tokens_a = make_tokens(32)
    tree.insert(tokens_a, [10, 20])

    tokens_b = list(range(16)) + list(range(100, 116))
    tree.insert(tokens_b, [10, 30])

    # 分裂后应该有：root → prefix(16) → {suffix_a(16), suffix_b(16)}
    prefix_node = tree.root.children[0]
    assert len(prefix_node.children) == 2, f"分裂后应有 2 个子节点: {len(prefix_node.children)}"

    # 删除 B 的叶子
    tree.delete(30)

    # 删除后 prefix 应该和 A 的后缀合并
    # 树结构变回：root → single_node(32)
    assert len(tree.root.children) == 1, f"合并后 root 应只有 1 个子节点"
    remaining = list(tree.root.children.values())[0]
    assert len(remaining.key_tokens) == 32, f"合并后应有 32 个 token: {len(remaining.key_tokens)}"

    # A 仍然可以匹配
    matched, length = tree.match(tokens_a)
    assert length == 32 and matched == [10, 20], f"合并后 A 应仍可匹配: {matched}, {length}"

    print("✅ test_delete_and_merge")

def test_delete_then_match():
    """删除后确认不再命中"""
    tree = KVCacheRadixTree(BLOCK_SIZE)
    tokens = make_tokens(32)
    tree.insert(tokens, [10, 20])

    tree.delete(20)  # 删除第二个 block 对应的节点

    matched, length = tree.match(tokens)
    # block 20 被删了，但 block 10 可能还在（取决于树结构）
    assert 20 not in matched, f"删除的 block 不应出现: {matched}"

    print(f"  删除后匹配: blocks={matched}, len={length}")
    print("✅ test_delete_then_match")

def test_empty_match():
    """空树匹配"""
    tree = KVCacheRadixTree(BLOCK_SIZE)
    matched, length = tree.match(make_tokens(32))
    assert matched == [] and length == 0
    print("✅ test_empty_match")

def test_partial_block_match():
    """匹配不足一个 block 的部分，应该返回 0"""
    tree = KVCacheRadixTree(BLOCK_SIZE)

    tokens_a = make_tokens(32)
    tree.insert(tokens_a, [10, 20])

    # 查询只有 8 个 token 匹配（不足一个 block）
    query = list(range(8)) + [999] * 24
    matched, length = tree.match(query)

    # 8 个 token < block_size=16，不够一个完整 block
    assert matched == [], f"不足一个 block 不应返回: {matched}"
    assert length == 0, f"length 应为 0: {length}"
    print("✅ test_partial_block_match")

def test_insert_returns_newly_held_blocks():
    tree = KVCacheRadixTree(BLOCK_SIZE)

    # 无匹配：返回完整列表
    held = tree.insert(make_tokens(32), [10, 20])
    assert held == [10, 20]

    # 完全匹配（幂等）：返回空
    held = tree.insert(make_tokens(32), [10, 20])
    assert held == []

    # 部分匹配分裂：只返回新后缀的 block
    tokens_b = list(range(16)) + list(range(100, 116))
    held = tree.insert(tokens_b, [10, 30])
    assert held == [30]  # block 10 公共前缀不算新持有，block 30 是 B 新加的
    print("✅ test_insert_returns_newly_held_blocks")

if __name__ == "__main__":
    test_empty_match()
    test_basic_insert_match()
    test_prefix_match()
    test_no_match()
    test_split()
    test_multi_split()
    test_idempotent_insert()
    test_ref_count_inc_dec()
    test_ref_count_shared_prefix()
    test_evict_basic()
    test_evict_respects_ref_count()
    test_delete_and_merge()
    test_delete_then_match()
    test_partial_block_match()
    test_insert_returns_newly_held_blocks()

    print("\n🎉 全部测试通过！")