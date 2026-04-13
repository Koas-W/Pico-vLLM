# topo.py
import torch
from torch import Tensor, dtype
from dataclasses import dataclass

@dataclass
class ClusterConfig:
    world_size: int          # 总卡数
    p_ranks: list[int]       # P 组的 rank 列表，如 [0, 1]
    d_ranks: list[int]       # D 组的 rank 列表，如 [2, 3]
    tp_size_p: int             # TP 并行度（同构时 P 和 D 一样）
    tp_size_d: int             # TP 并行度（同构时 P 和 D 一样）
    
    @property
    def p_tp_groups(self) -> list[list[int]]:
        """P 组内的 TP 分组"""
        return [self.p_ranks[i:i+self.tp_size_p] 
                for i in range(0, len(self.p_ranks), self.tp_size_p)]
    
    @property
    def d_tp_groups(self) -> list[list[int]]:
        """D 组内的 TP 分组"""
        return [self.d_ranks[i:i+self.tp_size_d]
                for i in range(0, len(self.d_ranks), self.tp_size_d)]