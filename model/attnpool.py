import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, num_cls_token:int =0, output_dim: int = None):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.num_cls_token = num_cls_token
        if self.num_cls_token > 0:
            self.cls_token = nn.Parameter(torch.randn(self.num_cls_token, 1, embed_dim))

        self.ln = nn.LayerNorm(embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + max(1, self.num_cls_token), embed_dim))

    def forward(self, x, cls_token=None, bcn=True):
        x = x.flatten(start_dim=2).permute(2, 0, 1) # NCHW -> (HW)NC
        n, b, c = x.shape
        if cls_token is None:
            if self.num_cls_token == 0:
                cls_token = x.mean(dim=0, keepdim=True)
            else:
                cls_token = torch.tile(self.cls_token, (1, b, 1))

        x = torch.cat([cls_token, x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        
        x = self.ln(x) 

        x, _ = F.multi_head_attention_forward(
            query=x[:max(1, self.num_cls_token)], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        if bcn:
            return torch.permute(x, (1, 2, 0)) # (1 or self.num_cls_token), N, C -> N, C, (1 or self.num_cls_token)
        else:
            return x
