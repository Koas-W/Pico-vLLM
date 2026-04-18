from blockmanager import BlockManager
from radix_tree import KVCacheRadixTree

class PrefixCache:
    """
    协调 RadixTree 和 BlockManager，对外提供 match / insert / release 接口。
    """
    def __init__(self, radix_tree:KVCacheRadixTree, block_manager:BlockManager):
        self.radix_tree = radix_tree
        self.block_manager = block_manager
        self.stats = {
            'match_calls': 0,
            'hit_tokens': 0,
            'miss_tokens': 0,
        }

    def match(self, tokens: list[int]) -> tuple[list[int], int]:
        """
        请求开始，沿 tokens 路径 inc_ref。
        查找 prefix 匹配。命中的 block 自动 inc_ref（防止被驱逐）。
        返回 (matched_block_ids, matched_len)
        """
        self.stats['match_calls'] += 1 # 仅测试启用
        blocks, length = self.radix_tree.match(tokens)
        if blocks:
            self.block_manager.inc_ref(blocks)
            self.radix_tree.inc_ref(tokens[:length])
        self.stats['hit_tokens'] += length
        self.stats['miss_tokens'] += len(tokens) - length
        return blocks, length

    def insert(self, tokens: list[int], block_ids: list[int]):
        """
        请求 prefill 完成后，把新产生的 KV block 加入 cache。
        tokens 和 block_ids 必须对齐到 block_size。
        """
        newly_held_by_tree = self.radix_tree.insert(tokens, block_ids)
        if newly_held_by_tree:
            self.block_manager.inc_ref(newly_held_by_tree)
        return newly_held_by_tree

    # def release(self, tokens: list[int]):
    #     """
    #     请求完全结束（不会再进入Decode和prefill队列）时调用，沿 tokens 路径 dec_ref。
    #     """
    #     blocks, matched_len = self.radix_tree.match(tokens)
    #     if blocks:
    #         self.radix_tree.dec_ref(tokens[:matched_len])
    #         # self.block_manager.dec_ref(blocks)
    def release(self, radix_path_tokens: list[int], held_blocks: list[int]):
        """
        radix_path_tokens: 这个请求在 radix 上 inc 过 ref 的路径（用来 dec radix 节点的内部 ref）
        held_blocks: 这个请求在 block_manager 上持有引用的所有 block（= kv_cache.logical_block_ids）
        """
        if radix_path_tokens:
            self.radix_tree.dec_ref(radix_path_tokens)
        if held_blocks:
            self.block_manager.dec_ref(held_blocks)

    def try_evict(self, num_blocks_needed: int) -> list[int]:
        """
        BlockManager 显存不足时调用，从 radix tree 驱逐可驱逐的叶子节点。
        返回被释放的物理 block id 列表。
        """    
        evicted_blocks = self.radix_tree.evict(num_blocks_needed)
        if evicted_blocks:
            # RadixTree 不再持有这些 block，给 BlockManager 的 ref_count -1
            # 如果 -1 后到 0，BlockManager 自动把 block 加回 free pool
            self.block_manager.dec_ref(evicted_blocks)
        return evicted_blocks
    
    def peek(self, tokens: list[int]) -> tuple[list[int], int]:
        """只查询，不改 ref。用于测试或只读检查。"""
        return self.radix_tree.match(tokens)

    def hit_rate(self):
        total = self.stats['hit_tokens'] + self.stats['miss_tokens']
        return self.stats['hit_tokens'] / total if total > 0 else 0