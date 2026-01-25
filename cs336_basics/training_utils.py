from collections.abc import Callable, Iterable
from typing import Optional
import math
import numpy as np
import os
import typing

import torch
import torch.nn as nn
from einops import rearrange, einsum, reduce


def cross_entropy_loss(logits : torch.Tensor,
                       targets : torch.Tensor,
                       ): 
    
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1))
    logsumexp = torch.logsumexp(input=logits, dim=-1, keepdim=True)
    loss = -target_logits + logsumexp
    return torch.mean(loss,dim=0, keepdim=False)


class AdamW(torch.optim.Optimizer):
    
    def __init__(self,
                params : torch.nn.Parameter,
                lr : float = 1e-3,
                weight_decay : float = 1e-5,
                betas : tuple[float] = (0.9,0.95),
                eps : float = 1e-8
                ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr":lr, "betas":betas, "decay_rate":weight_decay, "eps":eps}
        super().__init__(params, defaults)
    
    @torch.no_grad()    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            decay = group["decay_rate"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
        
                state = self.state[p] # Get state associated with p.
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    
                m, v = state["m"], state["v"]    
                state["t"] += 1    
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                
                if decay > 0:
                    p.mul_(1 - lr * decay)
                    
                grad = p.grad.data
                
                m.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                v.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])
                lr_t = lr*((math.sqrt(1-betas[1]**t))/(1-betas[0]**t))
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-lr_t)
        
        return loss
    

def lr_scheduler(step : int,
                max_lr : float,
                min_lr : float,
                warmup_steps : int,
                cosine_steps : int
                ):
    if step<warmup_steps:
        return max_lr*(step/warmup_steps)
    
    elif warmup_steps<=step<=cosine_steps:
        return min_lr + (0.5*(1+math.cos(math.pi*(step-warmup_steps)/(cosine_steps-warmup_steps)))*(max_lr-min_lr))
    else:
        return min_lr
    

def clip_gradients(params: Iterable[torch.nn.Parameter], max_norm: float):
    """
    Implements global gradient clipping in-place.
    """
    eps = 1e-6
    grads = [p.grad for p in params if p.grad is not None]
    
    if len(grads) == 0:
        return
    
    total_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for g in grads:
            g.mul_(clip_coef)
    
    
def data_loading(dataset : np.ndarray,
                batch_size : int, 
                context_length : int, 
                device : str):
    data_len = len(dataset)
    ids = torch.randint(0, data_len - context_length, (batch_size,))

    x = torch.stack([torch.from_numpy(dataset[i : i+context_length].astype(np.int64)) for i in ids])
    y = torch.stack([torch.from_numpy(dataset[i+1 : i+context_length+1].astype(np.int64)) for i in ids])
    
    return x.to(device), y.to(device)          
            

            
def save_checkpoint(
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    iteration : int,
    out : str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src : str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration

@torch.no_grad()
def decode(model : torch.nn.Module,
           prompt : torch.Tensor,
           max_tokens : int,
           special_token : int,
           temperature : float = 1,
           top_p : float = 1):
    assert temperature>=0, "temperature must be >= 0"
    from utils import softmax
    
    new_token = None
    generated_tokens = 0
    while new_token != special_token or generated_tokens<max_tokens:
        logits = model(prompt)
        last_logits = logits[:,-1,:]
        
        if temperature == 0:
            new_token = torch.argmax(last_logits,dim=-1,keepdim=False).indices       
        else:    
            probs = softmax(last_logits/temperature, dim=-1)
            new_token = torch.argmax(probs,dim=-1,keepdim=False).indices
        
        generated_tokens+=1
        prompt = torch.concat((prompt,new_token),dim=-1)
        
    return prompt
        