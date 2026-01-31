import torch.nn as nn
import torch
from torch.nn import Embedding, Parameter
from einops import rearrange, einsum, reduce


class Linear(nn.Module):
    """linear transformation
    (y = x.W^T)
    """
    def __init__(self,
                in_features : int,
                out_features : int,
                device : torch.device | None = None,
                dtype : torch.dtype | None = None
                ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        
        std = (2/(in_features+out_features))
        weights = torch.empty((out_features,in_features), **factory_kwargs)
        self.weight = Parameter(
            nn.init.trunc_normal_(
                (weights),
                mean=0.0,
                std=std,
                a=-3*std,
                b=3*std))
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        #equivalent to x @ W^T
        return einsum(x,self.weight,"... d_in, d_out d_in -> ... d_out")
    

class Embedding(nn.Module):
    def __init__(self,
                num_embeddings : int,
                embedding_dim : int,
                device : torch.device | None = None,
                dtype : torch.dtype | None = None):
        factory_kwargs = {'device':device, 'dtype':dtype}
        super().__init__()
        self.vocab = num_embeddings
        self.d_model = embedding_dim
        
        embeddings = torch.empty((self.vocab,self.d_model),**factory_kwargs)
        self.weight = Parameter(
            nn.init.trunc_normal_(
                embeddings,
                mean=0,
                std=1,
                a=-3,
                b=3))
        
    def forward(self, token_ids : torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self,
                d_model  : int,
                eps : float = 1e-5,
                device : torch.device | None = None,
                dtype : torch.dtype | None = None):
        factory_kwargs = {'device':device, 'dtype':dtype}
        super().__init__()
        self.d_model = d_model 
        self.eps = eps
        self.weight = Parameter(torch.ones((d_model), **factory_kwargs))

    def forward(self, x : torch.Tensor):
        in_dtype = x.dtype
        if  x.dtype == torch.float32:
            pass
        else:
            x = x.to(torch.float32)
            
        rms = torch.sqrt(reduce(x**2, "... d_model -> ... 1", "mean")+self.eps)
        result = einsum(x, self.weight, "... d_model, d_model -> ... d_model")/rms
        return result.to(in_dtype)
        

class SWIGLU(nn.Module):
    def __init__(self,
                d_model : int,
                d_ff : int,
                device : torch.device | None = None,
                dtype : torch.dtype | None = None):
        factory_kwargs = {'device':device, 'dtype':dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model,d_ff,**factory_kwargs)
        self.w2 = Linear(d_ff,d_model,**factory_kwargs)
        self.w3 = Linear(d_model,d_ff,**factory_kwargs)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        def SiLU(x):
            return einsum(x,torch.sigmoid(x), "... d_ff, ... d_ff -> ... d_ff")
            #return x*torch.sigmoid(x)
        
        #return (SiLU(x@self.W1.T) * (x@self.W3.T)) @ self.W2.T
        gate = SiLU(einsum(x,self.w1.weight, "... d_model, d_ff d_model -> ... d_ff"))
        linear = einsum(x,self.w3.weight, "... d_model, d_ff d_model -> ... d_ff")
        intermediate = einsum(gate,linear, "... d_ff, ... d_ff -> ... d_ff")
        return einsum(intermediate,self.w2.weight, "... d_ff, d_model d_ff -> ... d_model")
        

class RoPE(nn.Module): #revisit
    def __init__(self,
                theta : float,
                d_k : int,
                max_seq_len : int,
                device : torch.device | None = None):
        super().__init__()
        factory_args = {'device':device}
        k = torch.arange(1, (d_k//2)+1, **factory_args)
        theta_ = theta**(2*(k-1)/d_k)
        sequence = torch.arange(max_seq_len,**factory_args)
        grid = einsum(sequence, 1.0/theta_, "i, k -> i k") #(seq_len,d_k/2)
        self.register_buffer("cos", torch.cos(grid), persistent=False)
        self.register_buffer("sin", torch.sin(grid), persistent=False)
        
    def forward(self, x : torch.Tensor, token_positions : torch.Tensor) -> torch.Tensor:
        #x = (batch,heads,seq_len,d_model)
        cos = self.cos[token_positions] 
        sin = self.sin[token_positions]
        
        x1 = x[...,0::2] #even indices
        x2 = x[...,1::2] #odd indices
        
        while cos.ndim < x1.ndim:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        
        ans = torch.empty_like(x)
        ans[..., 0::2] = x1 * cos - x2 * sin
        ans[..., 1::2] = x1 * sin + x2 * cos
        return ans
    

def softmax(x : torch.Tensor, dim : int) -> torch.Tensor:
    max_el = torch.max(x, dim=dim, keepdim=True).values
    x = x-max_el
    num = torch.exp(x)
    denom = torch.sum(num, dim=dim, keepdim=True)
    return num/denom

def self_attention(q : torch.Tensor,
                   k : torch.Tensor,
                   v : torch.Tensor,
                   mask = None):     
    d_k = k.shape[-1]
    scores = (q@k.transpose(-2,-1))/(d_k**0.5)
    if mask is not None:
        scores = torch.where(mask.bool(), scores, float("-inf"))
        
    return softmax(scores, dim=-1)@v

    
class MHA(nn.Module):
    def __init__(self, d_model : int,
                num_heads : int,
                theta : int | None = None,
                max_seq_len : int | None = None,
                device : torch.device | None = None,
                dtype : torch.dtype | None = None):
        factory_kwargs = {'device':device, 'dtype':dtype}
        super().__init__()
    
        self.d_model = d_model
        self.num_heads = num_heads
        self.qkv_proj = Linear(d_model,3*d_model,**factory_kwargs)
        self.output_proj = Linear(d_model,d_model,**factory_kwargs)
        if theta is not None:
            self.rope = RoPE(theta,d_model//num_heads,max_seq_len,device)
        else:
            self.rope = None
        
        
    def forward(self, x : torch.Tensor, token_positions : torch.Tensor = None) -> torch.Tensor:
        seq_len = x.shape[1]
        # q = q.view(b, seq, num_heads, d_model//num_heads)
        # q = q.permute(0,2,1,3)
        qkv = self.qkv_proj(x)
        q,k,v = rearrange(qkv, "b s (three h d) -> three b h s d", three = 3, h = self.num_heads)
        
        if self.rope is not None and token_positions is not None:
            q = self.rope(q,token_positions)
            k = self.rope(k,token_positions)
        
        mask = torch.tril(torch.ones((seq_len,seq_len),device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        o = self_attention(q,k,v,mask)
        output = rearrange(o, "b heads seq d_head -> b seq (heads d_head)")
        return self.output_proj(output)
    

class LMBlock(nn.Module):
    def __init__(self, 
                d_model : int,
                num_heads : int,
                d_ff : int,
                theta : float,
                max_seq_len : int,
                device : torch.device | None = None,
                dtype : torch.dtype | None = None,
                ):
        factory_kwargs = {'device':device, 'dtype':dtype}
        super().__init__()
        
        self.ln1 = RMSNorm(d_model,**factory_kwargs)
        self.ln2 = RMSNorm(d_model,**factory_kwargs)
        self.attn = MHA(d_model,num_heads,theta,max_seq_len,**factory_kwargs)
        self.ffn = SWIGLU(d_model,d_ff,**factory_kwargs)
        
    def forward(self, x : torch.Tensor, token_positions : torch.Tensor) -> torch.Tensor:
        y_ = x + self.attn(self.ln1(x),token_positions)
        return y_ + self.ffn(self.ln2(y_))
    

class LLM(nn.Module):
    def __init__(self,
                d_model : int,
                d_ff : int,
                num_layers : int,
                num_heads : int,
                vocab_size : int,
                context_length : int,
                theta : float,
                device : torch.device | None = None,
                dtype : torch.dtype | None = None
                ):
        factory_kwargs = {'device':device, 'dtype':dtype}
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LMBlock(d_model,
                                          num_heads,
                                          d_ff,
                                          theta,
                                          context_length,
                                          **factory_kwargs))
        self.token_embeddings = Embedding(vocab_size,d_model,**factory_kwargs)
        self.ln_final = RMSNorm(d_model,**factory_kwargs)
        self.lm_head = Linear(d_model,vocab_size,**factory_kwargs)
            
    def forward(self, x : torch.Tensor):
        x = self.token_embeddings(x)
        b, s, _ = x.shape
        token_positions = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        for block in self.layers:
            x = block(x,token_positions)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x