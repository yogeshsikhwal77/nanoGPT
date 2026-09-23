import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from nanogpt.config import ModelConfig

class CausalSelfAttention(nn.Module):
    def __init__(self,cfg: ModelConfig):
        super().__init__()
        assert cfg.dim % cfg.heads == 0, "dim must be divisible by heads"
        self.heads = cfg.heads
        self.head_dim = cfg.dim // cfg.heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self,x):
        B,T,C = x.shape
        q,k,v = self.qkv(x).split(C,dim=2)
        q = q.view(B,T,self.heads,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.heads,self.head_dim).transpose(1,2)

        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y= y.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim,4*cfg.dim,bias=False)
        self.fc2 = nn.Linear(4*cfg.dim,cfg.dim,bias=False)

    def forward(self,x):
        return self.fc2(F.gelu(self.fc1(x)))

    
class Block(nn.Module):
    def __init__(self,cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = MLP(cfg)

    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self,cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim,cfg.vocab_size,bias=False)

        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # smaller init on layers that write into the residual stream
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module,nn.Linear):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
        elif isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def num_params(self) -> int:
        params = sum(p.numel() for p in self.parameters())
        return params

    def forward(self,idx,targets=None):
        B,T = idx.shape
        assert T<= self.cfg.context_len, f"sequence {T} > context {self.cfg.context_len}"

        pos = torch.arange(T,device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1,logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits,loss

if __name__ == "__main__":
    from nanogpt.config import load_base_config

    cfg = load_base_config("configs/base.yaml").model
    model = GPT(cfg)
    print(f"parameters: {model.num_params():,}")