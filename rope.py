from typing import Tuple
import torch
import numpy as np

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

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.

    # This is the unoptimized version
    '''
    query_real_rot = torch.zeros_like(query)
    key_real_rot = torch.zeros_like(key)
    query_imag_rot = torch.zeros_like(query)
    key_imag_rot = torch.zeros_like(key)
    for i in range(seqlen):
        for j in range(1, (head_dim // 2) + 1):
            theta_d =  theta ** ((-2 * (j - 1)) / head_dim)
            cos_term = np.cos(i * theta_d)
            sin_term = np.sin(i * theta_d)
            query_real_rot[:, i, :, 2 * (j - 1)] = cos_term * query_real[:, i, :, j-1]
            query_real_rot[:, i, :, 2 * (j - 1) + 1] = sin_term * query_real[:, i, :, j-1]
            query_imag_rot[:, i, :, 2 * (j - 1)] = -sin_term * query_imag[:, i, :, j-1]
            query_imag_rot[:, i, :, 2 * (j - 1) + 1] = cos_term * query_imag[:, i, :, j-1]
            key_real_rot[:, i, :, 2 * (j - 1)] = cos_term * key_real[:, i, :, j-1]
            key_real_rot[:, i, :, 2 * (j - 1) + 1] = sin_term * key_real[:, i, :, j-1]
            key_imag_rot[:, i, :, 2 * (j - 1)] = -sin_term * key_imag[:, i, :, j-1]
            key_imag_rot[:, i, :, 2 * (j - 1) + 1] = cos_term * key_imag[:, i, :, j-1]
    

    # Return the rotary position embeddings for the query and key tensors

    query_out = query_real_rot + query_imag_rot
    key_out = key_real_rot + key_imag_rot'''

    # Optimized version
    # Reference (https://github.com/lucidrains/rotary-embedding-torch/blob/783d17820ac1e75e918ae2128ab8bbcbe4985362/rotary_embedding_torch/rotary_embedding_torch.py#L96)
    # Only for getting the freqs
    freqs = 1. / (theta ** (torch.arange(0, head_dim, 2)[:(head_dim // 2)].float() / head_dim)).to(device)
    seq_ids = torch.arange(seqlen).float().to(device)
    # Reference (https://pytorch.org/docs/stable/generated/torch.einsum.html)
    freqs_cis = torch.einsum('i,j->ij', seq_ids, freqs)
    freqs_cis_cos = freqs_cis.cos()
    freqs_cis_sin = freqs_cis.sin()
    freqs_broadcast_cos = reshape_for_broadcast(freqs_cis_cos, query_real)
    freqs_broadcast_sin = reshape_for_broadcast(freqs_cis_sin, query_imag)
    
    query_real_rot = freqs_broadcast_cos * query_real - freqs_broadcast_sin * query_imag
    query_imag_rot = freqs_broadcast_sin * query_real + freqs_broadcast_cos * query_imag
    key_real_rot = freqs_broadcast_cos * key_real - freqs_broadcast_sin * key_imag
    key_imag_rot = freqs_broadcast_sin * key_real + freqs_broadcast_cos * key_imag

    # Return the modified query and key tensors
    query_out = torch.stack((query_real_rot, query_imag_rot), dim = -1).reshape(query.shape)
    key_out = torch.stack((key_real_rot, key_imag_rot), dim = -1).reshape(key.shape)

    return query_out, key_out
