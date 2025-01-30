from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.
    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...
    # query_read (bs, seq, n_local_heads, head_dim // 2) 
    
    m = torch.arange(seqlen).unsqueeze(-1).to(device)
    # print(m)
    Theta = theta ** (-2*torch.arange(0, head_dim//2) / head_dim).to(device)
    # print(Theta)
    # print(Theta.shape)
    angle = m * Theta        # (seqlen, head_dim//2)  给m增加一个维度，用于广播
    # print(angle)
    cos = torch.cos(angle).view(1,seqlen, 1, head_dim//2)  # 可以直接利用广播机制与query_real, query_imag进行广播
    sin = torch.sin(angle).view(1,seqlen, 1, head_dim//2)  # 同上
    # 为什么能直接进行广播？因为旋转位置编码只与位置（seqlen）和 每个token对应的向量维度（head_dim）有关，只要这两个维度上的数值确定，其他的都一样。
    # 在旋转位置编码（Rotary Position Embedding, RoPE）中，不同的头（attention heads）共享相同的旋转编码方案，即不同的头在位置编码上没有差异
    # 交错拼接的方法 torch.stack([a,b], dim=-1).view(query.shape)
    query_out = torch.stack([query_real * cos - query_imag * sin, query_real * sin + query_imag * cos], dim=-1).view(query.shape)
    key_out = torch.stack([key_real * cos - key_imag * sin, key_real * sin + key_imag * cos], dim=-1).view(key.shape) 

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out