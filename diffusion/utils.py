import torch
import numpy as np

def clamp_to_nearest_embedding(input_emb, vocab_emb):
    """
    Clamp input_emb to the nearest embedding in vocab embedding
    """
    # input_emb: (batch_size, seq_len, emb_dim)
    # vocab_emb: (vocab_size, emb_dim)
    # output: (batch_size, seq_len, emb_dim)
    
    batch_size, seq_len, emb_dim = input_emb.size()

    input_emb_norm = (input_emb ** 2).sum(dim=-1, keepdim=True) # (batch_size, seq_len, 1)
    vocab_emb_norm = (vocab_emb ** 2).sum(dim=-1, keepdim=True) # (vocab_size, 1)

    input_emb = input_emb.view(-1, emb_dim) # (batch_size * seq_len, emb_dim)
    input_emb_norm = input_emb_norm.view(-1, 1) # (batch_size * seq_len, 1)

    dist = input_emb_norm + vocab_emb_norm.transpose(0, 1) - 2 * torch.matmul(input_emb, vocab_emb.transpose(0, 1)) # (batch_size * seq_len, vocab_size)
    dist = torch.clamp(dist, min=0.0, max=np.inf) # (batch_size * seq_len, vocab_size)

    top1 = torch.argmin(dist, dim=-1) # (batch_size * seq_len)
    top1 = top1.view(batch_size, seq_len) # (batch_size, seq_len)

    top1_emb = vocab_emb[top1] # (batch_size, seq_len, emb_dim)

    return top1_emb