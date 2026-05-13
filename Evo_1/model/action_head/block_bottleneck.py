import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int):
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len)
        return self.pe[:, :seq_len, :]

    def _extend_pe(self, new_max_len):
        old_max_len, dim = self.pe.size(1), self.pe.size(2)
        if new_max_len <= old_max_len:
            return
        extra_positions = torch.arange(old_max_len, new_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        extra_pe = torch.zeros(new_max_len - old_max_len, dim)
        extra_pe[:, 0::2] = torch.sin(extra_positions * div_term)
        extra_pe[:, 1::2] = torch.cos(extra_positions * div_term)
        extra_pe = extra_pe.unsqueeze(0)
        new_pe = torch.cat([self.pe, extra_pe.to(self.pe.device)], dim=1)
        self.pe = new_pe

class CategorySpecificLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_categories: int = 1):
        super().__init__()
        self.num_categories = num_categories
        if num_categories <= 1:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.weight = nn.Parameter(torch.randn(num_categories, in_dim, out_dim))
            self.bias = nn.Parameter(torch.randn(num_categories, out_dim))

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):

        if self.num_categories <= 1:
            return self.linear(x)

        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]) 
        if category_id.dim() == 0:
       
            cid = category_id.item()
            out = x_flat @ self.weight[cid] + self.bias[cid]
        else:
           
            category_id = category_id.view(-1)  
            weight_selected = self.weight[category_id]        
            bias_selected = self.bias[category_id]        
            out = torch.bmm(x_flat.unsqueeze(1), weight_selected).squeeze(1) + bias_selected
        out_shape = orig_shape[:-1] + (out.shape[-1],)
        return out.view(out_shape)

class CategorySpecificMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_categories: int = 1):
        super().__init__()
        self.fc1 = CategorySpecificLinear(input_dim, hidden_dim, num_categories)
        self.fc2 = CategorySpecificLinear(hidden_dim, output_dim, num_categories)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):
        out = self.activation(self.fc1(x, category_id))
        out = self.fc2(out, category_id)
        return out

class MultiEmbodimentActionEncoder(nn.Module):

    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int, horizon: int, num_categories: int = 1):
        super().__init__()
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.num_categories = num_categories
        
        self.W1 = CategorySpecificLinear(action_dim, hidden_dim, num_categories)
        self.W2 = CategorySpecificLinear(hidden_dim, hidden_dim, num_categories)
        self.W3 = CategorySpecificLinear(hidden_dim, embed_dim, num_categories)
   
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=horizon)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, action_seq: torch.Tensor, category_id: torch.LongTensor):

        B, H, D = action_seq.shape
        assert H == self.horizon, "Action sequence length must match horizon"
       
        x = action_seq.reshape(B * H, D) 
      
        if category_id.dim() == 0:
           
            cat_ids = category_id.repeat(H * B)
        else:
            cat_ids = category_id.unsqueeze(1).repeat(1, H).reshape(B * H)
        out = self.activation(self.W1(x, cat_ids))            
    
        pos_enc = self.pos_encoding(H).to(out.device)       
        pos_enc = pos_enc.repeat(B, 1, 1).reshape(B * H, -1) 
        out = out + pos_enc
        out = self.activation(self.W2(out, cat_ids))         
        out = self.W3(out, cat_ids)                        
        out = out.view(B, H, self.embed_dim)
        return out

class BlockRoutedCrossAttention(nn.Module):
    def __init__(self, action_dim: int, cond_dim: int, block_size_a: int, block_size_c: int,topk: int, num_heads: int, dropout, use_scale: bool = True,):
        super().__init__()
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.block_size_a = block_size_a
        self.block_size_c = block_size_c
        self.topk = topk
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale = use_scale
        
        # 关于Block Ai 与 Block Cj的相关系数计算
        self.route_dim = action_dim // 4
        
        self.action_route_proj = nn.Linear(action_dim, self.route_dim)
        self.cond_route_proj = nn.Linear(cond_dim, self.route_dim)
        
        # 池化
        self.action_block_pool = nn.AdaptiveAvgPool1d(1)
        self.context_block_pool = nn.AdaptiveAvgPool1d(1)
        
        # 共享cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=action_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=cond_dim,
            vdim=cond_dim,
        )
        self.out_dropout = nn.Dropout(dropout)
    
    
    @staticmethod
    def _pad_to_block(x: torch.Tensor, block_size: int):
        """
        x: [B, T, D]
        return:
            x_pad: [B, T_pad, D]
            pad_len: int
            防止token不能够整块被分割
        """
        B, T, D = x.shape
        pad_len = (block_size - (T % block_size)) % block_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len), mode="constant", value=0.0)
        return x, pad_len
    
    def forward(self, action_tokens: torch.Tensor, context_tokens: torch.Tensor, return_aux: bool = False):
        """
        action_tokens: [B, Ta, Da]
        context_tokens: [B, Tc, Dc]
        """
        assert action_tokens.dim() == 3
        assert context_tokens.dim() == 3

        B, Ta, Da = action_tokens.shape
        B2, Tc_total, Dc = context_tokens.shape
        assert B == B2
        assert Da == self.action_dim
        assert Dc == self.cond_dim
        assert Tc_total >= 1, "context_tokens 至少需要包含最后一个 state token"
        normal_context_tokens = context_tokens[:, :-1, :]   # [B, Tc, Dc]
        state_token = context_tokens[:, -1:, :]             # [B, 1, Dc]
        Tc = normal_context_tokens.size(1)
        # 1) pad
        action_pad, pad_a = self._pad_to_block(action_tokens, self.block_size_a)   # [B, Ta_pad, Da]
        context_pad, pad_c = self._pad_to_block(normal_context_tokens, self.block_size_c) # [B, Tc_pad, Dc]

        Ta_pad = action_pad.size(1)
        Tc_pad = context_pad.size(1)

        num_a_blocks = Ta_pad // self.block_size_a
        num_c_blocks = Tc_pad // self.block_size_c
        topk_eff = min(self.topk, num_c_blocks)

        # 2) 分块
        action_blocks = action_pad.view(B, num_a_blocks, self.block_size_a, Da)    # [B, Na, Sa, Da]
        context_blocks = context_pad.view(B, num_c_blocks, self.block_size_c, Dc)  # [B, Nc, Sc, Dc]

        # 3) 块内 pooling（使用 nn.AdaptiveAvgPool1d）
        action_blocks_for_pool = action_blocks.reshape(
            B * num_a_blocks, self.block_size_a, Da
        ).transpose(1, 2)  # [B*Na, Da, Sa]

        action_summary = self.action_block_pool(action_blocks_for_pool).squeeze(-1)
        action_summary = action_summary.view(B, num_a_blocks, Da)  # [B, Na, Da]

        context_blocks_for_pool = context_blocks.reshape(
            B * num_c_blocks, self.block_size_c, Dc
        ).transpose(1, 2)  # [B*Nc, Dc, Sc]

        context_summary = self.context_block_pool(context_blocks_for_pool).squeeze(-1)
        context_summary = context_summary.view(B, num_c_blocks, Dc)  # [B, Nc, Dc]

        # 4) 路由分数
        action_route = self.action_route_proj(action_summary)   # [B, Na, R]
        context_route = self.cond_route_proj(context_summary)   # [B, Nc, R]

        routing_scores = torch.matmul(action_route, context_route.transpose(-1, -2))
        if self.use_scale:
            routing_scores = routing_scores / math.sqrt(self.route_dim)

        # 5) top-k
        topk_indices = routing_scores.topk(k=topk_eff, dim=-1).indices  # [B, Na, K]

        # 6) gather 条件块
        context_blocks_flat = context_blocks.reshape(B * num_c_blocks, self.block_size_c, Dc)
        batch_offsets = torch.arange(B, device=context_tokens.device).view(B, 1, 1) * num_c_blocks
        flat_indices = topk_indices + batch_offsets
        # 通过展平取索引的方式来得到各个block
        selected_context_blocks = context_blocks_flat[flat_indices]  # [B, Na, K, Sc, Dc]
        selected_context_tokens = selected_context_blocks.reshape(
            B, num_a_blocks, topk_eff * self.block_size_c, Dc
        )  # [B, Na, K*Sc, Dc]
        
        # 8) 单独拼接state_token，再与action做cross-attention
        state_token_expand = state_token.unsqueeze(1).expand(B, num_a_blocks, 1, Dc)  # [B, Na, 1, Dc]
        selected_context_tokens = torch.cat(
            [selected_context_tokens, state_token_expand], dim=2
        )  # [B, Na, K*Sc+1, Dc]
        # 7) 向量化共享 cross-attn
        action_blocks_flat = action_blocks.reshape(B * num_a_blocks, self.block_size_a, Da)
        selected_context_flat = selected_context_tokens.reshape(
            B * num_a_blocks, selected_context_tokens.size(2), Dc
        )
        # By my is "for" but AI choose matrix
        attn_out, _ = self.cross_attn(
            query=action_blocks_flat,
            key=selected_context_flat,
            value=selected_context_flat,
            need_weights=False,
        )

        attn_out = self.out_dropout(attn_out)

        # 8) 拼回原序列
        attn_out = attn_out.reshape(B, num_a_blocks, self.block_size_a, Da)
        attn_out = attn_out.reshape(B, Ta_pad, Da)
        attn_out = attn_out[:, :Ta, :]

        if not return_aux:
            return attn_out

        aux = {
            "routing_scores": routing_scores,
            "topk_indices": topk_indices,
            "num_a_blocks": num_a_blocks,
            "num_c_blocks": num_c_blocks,
            "pad_a": pad_a,
            "pad_c": pad_c,
        }
        return attn_out, aux
        

class ActionBlockAggregation(nn.Module):
    def __init__(self, action_dim: int, action_len: int, block_size_a: int, num_heads: int, dropout: float = 0.0, use_layernorm: bool = True,):
        super().__init__()
        assert action_dim > 0
        assert action_len > 0
        assert block_size_a > 0
        assert num_heads > 0
        assert action_len % block_size_a == 0, (
            f"action_len={action_len} must be divisible by block_size_a={block_size_a}"
        )
        assert action_dim % num_heads == 0, (
            f"action_dim={action_dim} must be divisible by num_heads={num_heads}"
        )
        self.action_dim = action_dim
        self.action_len = action_len
        self.block_size_a = block_size_a
        self.num_blocks = action_len // block_size_a
        # block -> summary
        self.block_pool = nn.AdaptiveAvgPool1d(1)
        # self-attn over summaries
        self.summary_norm = nn.LayerNorm(action_dim) if use_layernorm else nn.Identity()
        self.summary_attn = nn.MultiheadAttention(
            embed_dim=action_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.summary_dropout = nn.Dropout(dropout)

        # gamma_b in R^{Sa x 1}, one gamma per token position in each block
        # shape: [1, Nb, Sa, 1]
        self.gamma = nn.Parameter(torch.zeros(1, self.num_blocks, self.block_size_a, 1))

    def forward(self, action_tokens: torch.Tensor, return_aux: bool = False):
        """
        action_tokens: [B, Ta, Da]
        """
        assert action_tokens.dim() == 3, f"Expected [B, Ta, Da], got {action_tokens.shape}"
        B, Ta, Da = action_tokens.shape
        assert Ta == self.action_len, f"Expected action_len={self.action_len}, got {Ta}"
        assert Da == self.action_dim, f"Expected action_dim={self.action_dim}, got {Da}"

        # [B, Ta, Da] -> [B, Nb, Sa, Da]
        action_blocks = action_tokens.view(B, self.num_blocks, self.block_size_a, Da)

        # pool summaries
        # [B, Nb, Sa, Da] -> [B*Nb, Da, Sa] -> [B*Nb, Da, 1] -> [B, Nb, Da]
        action_blocks_for_pool = action_blocks.reshape(
            B * self.num_blocks, self.block_size_a, Da
        ).transpose(1, 2)

        block_summaries = self.block_pool(action_blocks_for_pool).squeeze(-1)
        block_summaries = block_summaries.view(B, self.num_blocks, Da)  # [B, Nb, Da]

        # self-attention on summaries
        summaries_norm = self.summary_norm(block_summaries)
        summary_delta, _ = self.summary_attn(
            summaries_norm, summaries_norm, summaries_norm, need_weights=False
        )
        summary_delta = self.summary_dropout(summary_delta)
        block_summaries_hat = block_summaries + summary_delta
            
        # inject back to each block
        injected_summary = self.gamma * block_summaries_hat.unsqueeze(2)  # [B, Nb, Sa, Da]
        aggregated_blocks = action_blocks + injected_summary               # [B, Nb, Sa, Da]

        aggregated_action_tokens = aggregated_blocks.reshape(B, Ta, Da)

        if not return_aux:
            return aggregated_action_tokens

        aux = {
            "block_summaries": block_summaries,         # [B, Nb, Da]
            "block_summaries_hat": block_summaries_hat, # [B, Nb, Da]
            "gamma": self.gamma,                        # [1, Nb, Sa, 1]
        }
        return aggregated_action_tokens, aux

        
        
        
class BasicTransformerBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, action_tokens: torch.Tensor, context_tokens: torch.Tensor, time_emb: torch.Tensor):

        x = self.norm1(action_tokens)
        attn_out, _ = self.attn(x, context_tokens, context_tokens)

        x = action_tokens + attn_out

        x2 = self.norm2(x)

        if time_emb is not None:
            x2 = x2 + time_emb.unsqueeze(1)
        ff_out = self.ff(x2)
        x = x + ff_out
        return x

class BlockRoutedTransformerBlock(nn.Module):
    def __init__(
        self, embed_dim: int, cond_dim: int, action_len: int, num_heads: int, hidden_dim: int, block_size_a: int, block_size_c: int, topk: int, dropout: float = 0.0,use_scale: bool = True,agg_num_heads: int = None,):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.routed_attn = BlockRoutedCrossAttention(
            action_dim=embed_dim,
            cond_dim=cond_dim,
            block_size_a=block_size_a,
            block_size_c=block_size_c,
            topk=topk,
            num_heads=num_heads,
            dropout=dropout,
            use_scale=use_scale,
        )
        
        self.action_block_agg = ActionBlockAggregation(
            action_dim=embed_dim,
            action_len=action_len,
            block_size_a=block_size_a,
            num_heads=num_heads,
            dropout=dropout,
            use_layernorm=True,
        )
        

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        action_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        time_emb: torch.Tensor,
        return_aux: bool = False,
    ):
        # route attention
        x_norm = self.norm1(action_tokens)

        if return_aux:
            attn_out, aux = self.routed_attn(x_norm, context_tokens, return_aux=True)
        else:
            attn_out = self.routed_attn(x_norm, context_tokens, return_aux=False)
            aux = None
        
        # aggrate attention
        if return_aux:
            attn_out, agg_aux = self.action_block_agg(attn_out, return_aux=True)
        else:
            attn_out = self.action_block_agg(attn_out, return_aux=False)
            agg_aux = None
        
        x = action_tokens + attn_out
        
        # FFN
        x2 = self.norm2(x)
        if time_emb is not None:
            x2 = x2 + time_emb.unsqueeze(1)

        ff_out = self.ff(x2)
        x = x + ff_out

        if return_aux:
            return x, aux
        return x

class BlockBottleneckActionHead(nn.Module):

    def __init__(self, config=None,
                 embed_dim: int = 896, 
                 hidden_dim: int = 1024,
                 action_dim: int = 16*7,
                 horizon: int = 16,
                 per_action_dim: int = 7,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 dropout: float = 0.0,
                 num_inference_timesteps: int = 20,
                 num_categories: int = 1,
                 block_size_a: int = 5,
                 block_size_c = 32,
                 topk = 4,):
        super().__init__()

        if config is not None:
            embed_dim = getattr(config, "embed_dim", embed_dim)
            hidden_dim = getattr(config, "hidden_dim", hidden_dim)
            action_dim = getattr(config, "action_dim", action_dim)
            horizon = getattr(config, "horizon", horizon)
            per_action_dim = getattr(config, "per_action_dim", per_action_dim)
            num_heads = getattr(config, "num_heads", num_heads)
            num_layers = getattr(config, "num_layers", num_layers)
            dropout = getattr(config, "dropout", dropout)
            num_inference_timesteps = getattr(config, "num_inference_timesteps", num_inference_timesteps)
            num_categories = getattr(config, "num_categories", num_categories)
            block_size_a = getattr(config, "block_size_a", block_size_a)
            block_size_c = getattr(config, "block_size_c", block_size_c)
            topk = getattr(config, "topk", topk)
            self.config = config
        else:
            from types import SimpleNamespace
            self.config = SimpleNamespace(embed_dim=embed_dim, hidden_dim=hidden_dim,
                                          action_dim=action_dim, horizon=horizon,
                                          per_action_dim=per_action_dim,
                                          num_heads=num_heads, num_layers=num_layers,
                                          dropout=dropout, num_inference_timesteps=num_inference_timesteps,
                                          num_categories=num_categories,
                                          block_size_a=block_size_a, block_size_c=block_size_c, topk=topk)
        print(f"num_inference_timesteps {num_inference_timesteps}")
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.per_action_dim = per_action_dim
        self.action_dim = action_dim
        self.block_size_a = block_size_a
        self.block_size_c = block_size_c
        self.topk = topk


        self.time_pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=1000)

        self.transformer_blocks = nn.ModuleList([
            BlockRoutedTransformerBlock(
                embed_dim=embed_dim,
                cond_dim=embed_dim,
                action_len=horizon,
                num_heads=num_heads,
                hidden_dim=embed_dim * 4,
                block_size_a=self.block_size_a,
                block_size_c=self.block_size_c,
                topk=self.topk,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
       
        self.norm_out = nn.LayerNorm(embed_dim)
        self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim)

        self.mlp_head = CategorySpecificMLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                                            output_dim=action_dim, num_categories=num_categories)

        self.state_encoder = None
        if hasattr(self.config, "state_dim") and self.config.state_dim is not None:
       
            state_hidden = getattr(self.config, "state_hidden_dim", embed_dim)
        
            self.state_encoder = CategorySpecificMLP(input_dim=self.config.state_dim,
                                                    hidden_dim=state_hidden,
                                                    output_dim=embed_dim,
                                                    num_categories=num_categories)

        self.action_encoder = None
        if horizon > 1:
          
            per_action_dim = getattr(self.config, "per_action_dim", None)
            if per_action_dim is None:
            
                per_action_dim = action_dim // horizon if action_dim % horizon == 0 else action_dim
            self.action_encoder = MultiEmbodimentActionEncoder(action_dim=per_action_dim,
                                                               embed_dim=embed_dim,
                                                               hidden_dim=embed_dim,  
                                                               horizon=horizon,
                                                               num_categories=num_categories)

    def forward(self, fused_tokens: torch.Tensor, state: torch.Tensor = None,
                actions_gt: torch.Tensor = None, embodiment_id: torch.LongTensor = None, 
                state_mask: torch.Tensor = None, action_mask: torch.Tensor = None):

        if actions_gt is None:
            return self.get_action(fused_tokens, state=state, embodiment_id=embodiment_id)
        B = fused_tokens.size(0)
        device = fused_tokens.device

        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        context_tokens = fused_tokens 
        if state is not None and self.state_encoder is not None:

            state_emb = self.state_encoder(state, embodiment_id)  
            state_emb = state_emb.unsqueeze(1) 

            context_tokens = torch.cat([context_tokens, state_emb], dim=1) 

        t = torch.distributions.Beta(2, 2).sample((B,)).clamp(0.02, 0.98).to(device).to(dtype=self.dtype)

        
                    
        time_index = (t * 1000).long()  
        time_emb = self.time_pos_enc(1000)[:, time_index, :].squeeze(0) 
    
        action_shape = actions_gt.shape[1]  
    

        actions_gt_seq = actions_gt  


        noise = torch.rand_like(actions_gt) * 2 - 1  

        if action_mask is not None:
            action_mask = action_mask.to(dtype=noise.dtype, device=noise.device)
            assert action_mask.shape == noise.shape, f"action_mask shape {action_mask.shape} != noise shape {noise.shape}"
            noise = noise * action_mask


        if self.horizon > 1:
            noise_seq = noise.view(B, self.horizon, self.per_action_dim)
            
        else:
            noise_seq = noise.unsqueeze(1)

        if self.horizon > 1:
            t_broadcast = t.view(B, 1, 1)
        else:
            t_broadcast = t.view(B, 1)
        action_intermediate_seq = (1 - t_broadcast) * noise_seq + t_broadcast * actions_gt_seq  

        if self.horizon > 1 and self.action_encoder is not None:
     
            action_tokens = self.action_encoder(action_intermediate_seq, embodiment_id)  
        else:

            if not hasattr(self, "single_action_proj"):
                self.single_action_proj = nn.Linear(self.per_action_dim, self.embed_dim).to(device)
            action_tokens = self.single_action_proj(action_intermediate_seq) 

        x = action_tokens  
        for block in self.transformer_blocks:
            x = block(x, context_tokens, time_emb)

        x = self.norm_out(x)  

        if self.horizon > 1:
 
            x_flat = x.reshape(B, -1)  

            if not hasattr(self, "seq_pool_proj"):
              
                self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim).to(device)
            x_pooled = self.seq_pool_proj(x_flat)  
        else:
          
            x_pooled = x.squeeze(1) 

        pred_velocity = self.mlp_head(x_pooled, embodiment_id) 

        return pred_velocity, noise

    def get_action(self, fused_tokens: torch.Tensor, state: torch.Tensor = None, embodiment_id: torch.LongTensor = None, action_mask: torch.Tensor = None):

        print(f"action_mask shape: {action_mask.shape if action_mask is not None else 'None'}")
        print(f"one sample action_mask: {action_mask[0] if action_mask is not None else 'None'}")

        if action_mask is None:
            raise ValueError("action_mask must be provided for BlockBottleneckActionHead inference.")

        B = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        context_tokens = fused_tokens
        if state is not None and self.state_encoder is not None:

            state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1) 
            context_tokens = torch.cat([context_tokens, state_emb], dim=1)

        action_dim_total = getattr(self.config, "action_dim", None)
        if action_dim_total is None:
          
            action_dim_total = self.action_dim
       
        if self.horizon > 1:
            per_action_dim = getattr(self.config, "per_action_dim", action_dim_total // self.horizon)
        else:
            per_action_dim = action_dim_total

        action = (torch.rand(B, action_dim_total, device=device) * 2 - 1)
        print(f"action shape: {action.shape}")
        print(f"one sample action: {action[0]}")

        if self.horizon > 1:
            action_seq = action.view(B, self.horizon, per_action_dim)

        else:
            action_seq = action.view(B, 1, per_action_dim)

        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        expected_mask_shape = (B, per_action_dim)
        if tuple(action_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"Inference action_mask must have shape {expected_mask_shape}, got {tuple(action_mask.shape)}"
            )
        action_mask = action_mask.view(B, 1, per_action_dim).repeat(1, self.horizon, 1)

        print(f"action_mask: {action_mask}")
        print(f"one sample action_mask: {action_mask[0]}")

        action_mask = action_mask.to(dtype=action_seq.dtype, device=action_seq.device)
        assert action_mask.shape == action_seq.shape, f"action_mask shape {action_mask.shape} != noise shape {action_seq.shape}"
        action_seq = action_seq * action_mask
        print(f"action shape: {action_seq.shape}")
        print(f"one sample action: {action_seq[0]}")

        N = int(getattr(self.config, "num_inference_timesteps", 32))
        dt = 1.0 / N
        for i in range(N):
            t = i / N 

            time_index = int(t * 1000)
            time_emb = self.time_pos_enc(1000)[:, time_index, :].to(device).squeeze(0)  
            time_emb = time_emb.unsqueeze(0).repeat(B, 1)  


            if self.horizon > 1 and self.action_encoder is not None:

                action_seq = action_seq * action_mask
                action_tokens = self.action_encoder(action_seq, embodiment_id) 
            else:
                if hasattr(self, "single_action_proj"):
                    action_tokens = self.single_action_proj(action_seq)  
                else:

                    self.single_action_proj = nn.Linear(per_action_dim, self.embed_dim).to(device)
                    action_tokens = self.single_action_proj(action_seq)

            x = action_tokens
            for block in self.transformer_blocks:
                x = block(x, context_tokens, time_emb)
            x = self.norm_out(x)

            if self.horizon > 1:
                x_flat = x.reshape(B, -1)
                if hasattr(self, "seq_pool_proj"):
                    x_pooled = self.seq_pool_proj(x_flat)
                else:
                   
                    self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim).to(device)
                    x_pooled = self.seq_pool_proj(x_flat)
            else:
                x_pooled = x.squeeze(1)
         
            pred = self.mlp_head(x_pooled, embodiment_id)  
  
            action = action + dt * pred
          
            if self.horizon > 1:
                action_seq = action.view(B, self.horizon, per_action_dim)
            else:
                action_seq = action.view(B, 1, per_action_dim)
      
        return action

    @property
    def device(self):
      
        return next(self.parameters()).device

    @property
    def dtype(self):
        
        return next(self.parameters()).dtype

