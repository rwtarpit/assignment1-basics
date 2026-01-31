import torch
from cs336_basics.tokenizer import tokenizer


@torch.no_grad()
def decode(model : torch.nn.Module,
           prompt : torch.Tensor,
           max_tokens : int,
           special_token : int,
           temperature : float = 1.0,
           top_p : float = 1.0):
    assert prompt.shape[0] == 1, "can only process one prompt at a time"
    assert temperature>=0, "temperature must be >= 0"
    assert 0<top_p<=1, "nucleus sampling should be btw (0,1]"
    
    from utils import softmax
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    new_token = -1
    generated_tokens = 0
    while new_token != special_token and generated_tokens<max_tokens:
        logits = model(prompt)
        last_logits = logits[:,-1,:]
        
        if temperature == 0:
            new_token = torch.argmax(last_logits,dim=-1,keepdim=True)     
        else:    
            probs = softmax(last_logits/temperature, dim=-1)
            sorted_probs, indices = torch.sort(probs,dim=-1,descending=True)
            
            cum_prob = torch.cumsum(sorted_probs,dim=-1)
            ids_to_remove = cum_prob > top_p
            ids_to_remove[...,1:] = ids_to_remove[...,:-1].clone()
            ids_to_remove[...,0] = False
            not_sampled_indices = indices[ids_to_remove]
            
            probs.scatter_(-1, not_sampled_indices.unsqueeze(0), 0.0)
            new_token = torch.multinomial(probs,num_samples=1)

        generated_tokens+=1
        prompt = torch.concat((prompt,new_token),dim=-1)
        
    return prompt


def generate(model : torch.nn.Module,
            tokenizer : tokenizer,
            prompt : str,
            max_tokens : int,
            special_token : str,
            temperature : float = 1.0,
            top_p : float = 1.0
            ) -> str:
    
    device = next(model.parameters()).device
    encoded_prompt = tokenizer.encode(prompt)
    encoded_tokens = torch.tensor(encoded_prompt,device=device)
    special_token_id = tokenizer.encoding_vocab[special_token.encode("utf-8")]
    decoding_args = {"model" : model,
                     "prompt" : encoded_tokens,
                     "max_tokens" : max_tokens,
                     "special_token" : special_token_id,
                     "temperature" : temperature,
                     "top_p" : top_p}
    
    generated_tokens = decode(**decoding_args)
    return tokenizer.decode(generated_tokens.squeeze(0).detach().cpu().tolist())
    
