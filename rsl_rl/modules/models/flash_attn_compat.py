import torch
import torch.nn as nn

from flash_attn import flash_attn_qkvpacked_func
from flash_attn.modules.mha import MHA as _MHA


# ---------------------------------------------------------------------------
# FlashAttention  (v1 class → wraps v2 functional API)
# ---------------------------------------------------------------------------

class FlashAttention(nn.Module):
    def __init__(self, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, need_weights=False, causal=False):
        out = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )
        return out, None


# ---------------------------------------------------------------------------
# FlashMHA  (v1 class → subclasses v2 MHA directly)
# ---------------------------------------------------------------------------

class FlashMHA(_MHA):
    """v1-compatible FlashMHA that subclasses flash_attn v2.8.3 MHA directly.

    Subclassing (rather than wrapping) ensures the PyTorch module hierarchy
    is flat, so state_dict keys match checkpoints saved under the v1 FlashMHA:
        ...attention.Wqkv.weight      ✓  (direct attribute of FlashMHA)
        ...attention.out_proj.weight  ✓

    v1 __init__ signature:
        FlashMHA(embed_dim, num_heads, bias=True, batch_first=True,
                 attention_dropout=0.0, causal=False, device=None, dtype=None)

    v1 forward signature:
        forward(x, key_padding_mask=None, need_weights=False) -> (output, None)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        batch_first: bool = True,   # v2 MHA is always batch-first; param kept for compat
        attention_dropout: float = 0.0,
        causal: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_proj_bias=bias,
            out_proj_bias=bias,
            dropout=attention_dropout,
            causal=causal,
            device=device,
            dtype=dtype,
        )

    def forward(self, x, key_padding_mask=None, need_weights=False, **kwargs):
        # v2 MHA.forward returns a bare Tensor when return_residual=False (default)
        out = super().forward(x, key_padding_mask=key_padding_mask, **kwargs)
        if isinstance(out, torch.Tensor):
            return out, None
        # safety guard: return_residual=True gives (out, residual)
        return out[0], None
