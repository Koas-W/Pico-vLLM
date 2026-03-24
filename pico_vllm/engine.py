import torch
from cache import KVCache
import sampler

''' Engine 负责管理模型和采样器，提供统一接口供外部调用
- 第一阶段的计划是支持单卡、单模型、单batch，无KV cache，单步采样
'''
class Engine:
    def __init__(self, model, tokenizer, cache_cls, device='cuda'):
        self.model = model.to(device)
        # self.sampler = sampler
        self.tokenizer = tokenizer
        self.device = device
        self.kv_cache_cls = cache_cls
        # self.kv_cache = cache_cls(
        #     num_layers=model.cfg.num_hidden_layers,
        #     max_seq_len=model.cfg.max_position_embeddings,
        #     num_kv_heads=model.cfg.num_attention_heads,
        #     head_dim=model.cfg.hidden_size // model.cfg.num_attention_heads,
        #     device=device,
        #     dtype=next(model.parameters()).dtype)
    
    # ''' 
    # 不区分prefill和decode的naive实现，后续会discrepancy掉
    # 生成接口，输入 prompt 和采样参数，返回生成的字符串
    # - prompt: 输入文本
    # - max_new_tokens: 最多生成多少个 token
    # - temperature: 采样温度，0 表示 greedy
    # - top_p: top-p 截断，1.0 表示不截断
    # return: 生成的完整字符串（含 Prompt）
    # '''
    # def generate(self,
    #              prompt: str,
    #              max_new_tokens: int = 100,
    #              temperature: float = 1.0,
    #              top_p: float = 1.0) -> str:
    #     # 返回生成的完整字符串（含Prompt）
    #     # 把模型切换到 eval 模式，关闭 dropout 等训练时机制
    #     self.model.eval()
    #     # output_ids: (1, seq_len)，初始为 prompt 的 token ids，后续不断 append 新生成的 token id
    #     output_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
    #     num_new_tokens = 0
    #     while not (output_ids[0, -1].item() == self.tokenizer.eos_token_id or num_new_tokens >= max_new_tokens):
    #         logits = self.step(output_ids, temperature, top_p)
    #         next_token_id = sampler.sample(logits, temperature, top_p)
    #         output_ids = torch.cat([output_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
    #         num_new_tokens += 1
    #     return self.tokenizer.decode(output_ids[0])

    # ''' 模型单步前向，输入当前的 token ids，返回下一个 token 的 logits和已经生成的 token nums
    # - input_ids: 当前的 token ids，shape (1, seq_len)
    # - temperature: 采样温度，传递给 sampler
    # - top_p: top-p 截断，传递给 sampler
    # return: 下一个 token 的 logits，shape (vocab_size,)
    # '''
    # def step(self, input_ids: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    #     with torch.no_grad():
    #         outputs = self.model(input_ids)
    #         logits = outputs[0, -1, :]  # 取最后一个 token 的 logits，shape (vocab_size,)
    #     return logits


    ## reserved for naive kv cache & paged kv cache
    ''' 一次性前向，输入完整的 prompt，返回最后一个 token 的 logits
    - input_ids: 当前的 token ids，shape (1, seq_len)，包含整个 prompt
    return: 最后一个 token 的 logits，shape (vocab_size,)
    '''
    def prefill(self, input_ids: torch.Tensor, kvcache: KVCache) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids, kv_cache=kvcache)
            logits = outputs[:, -1, :]  # 取最后一个 token 的 logits，shape (vocab_size,)
        return logits

    ''' decode 接口，输入当前的 token ids 和 KV cache，返回下一个 token 的 logits
    - input_ids: 当前的 token ids，shape (1, seq_len)，只包含当前 step 的 token
    - kv_cache: 当前的 KV cache，包含历史 token 的 KV
    return: 下一个 token 的 logits，shape (vocab_size,)
    '''
    def decode_step(self, input_ids: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids, kv_cache=kv_cache)
            logits = outputs[:, -1, :]  # 取最后一个 token 的 logits，shape (vocab_size,)
        return logits
    
    ''' 生成接口，输入 prompt 和采样参数，返回生成的字符串
    - prompt: 输入文本
    - max_new_tokens: 最多生成多少个 token
    - temperature: 采样温度，0 表示 greedy
    - top_p: top-p 截断，1.0 表示不截断
    return: 生成的完整字符串（含 Prompt）'''
    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 100, 
                 temperature: float = 1.0, 
                 top_p: float = 1.0) -> str:
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        kv_cache = self.kv_cache_cls(
            num_layers=self.model.cfg.num_hidden_layers,
            max_seq_len=4096,  # 先写死，模型的 max_position_embeddings 会OOM
            num_kv_heads=self.model.cfg.num_key_value_heads,
            head_dim=self.model.cfg.hidden_size // self.model.cfg.num_attention_heads,
            device=self.device,
            dtype=next(self.model.parameters()).dtype)
        
        # prefill
        logits = self.prefill(input_ids, kv_cache)
        
        # output_ids = input_ids
        generated_ids = []
        num_new_tokens = 0
        while not (generated_ids and generated_ids[-1] == self.tokenizer.eos_token_id) and num_new_tokens < max_new_tokens:
            next_token_id = sampler.sample(logits, temperature, top_p)
            generated_ids.append(next_token_id.item())
            num_new_tokens += 1
            
            # decode step
            logits = self.decode_step(next_token_id.unsqueeze(0).unsqueeze(0), kv_cache)
        
        full_ids = input_ids[0].tolist() + generated_ids
        return self.tokenizer.decode(full_ids)