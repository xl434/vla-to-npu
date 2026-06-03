"""Generate test inputs + PyTorch bf16 reference output matching MiniVit in vision_block_bf16.py."""
import numpy as np
import torch
import torch.nn as nn
from ml_dtypes import bfloat16 as np_bfloat16

SEQ, EMBD, N_HEAD = 1024, 768, 12
FFN_HID = EMBD * 4

torch.manual_seed(0)
np.random.seed(0)

class MiniVit(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(EMBD, N_HEAD, batch_first=True)
        self.ln_1 = nn.LayerNorm(EMBD, elementwise_affine=True)
        self.ffn_up = nn.Linear(EMBD, FFN_HID, bias=False)
        self.ffn_down = nn.Linear(FFN_HID, EMBD, bias=False)
        self.gelu = nn.GELU()
        self.ln_2 = nn.LayerNorm(EMBD, elementwise_affine=True)
        self.attn.in_proj_bias.data.zero_()
        self.attn.out_proj.bias.data.zero_()

    def forward(self, x):
        residual = x
        x = self.ln_1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = attn_out + residual
        residual = x
        x = self.ln_2(x)
        x = self.ffn_down(self.gelu(self.ffn_up(x)))
        return residual + x

def bf16(a): return a.astype(np_bfloat16)
def save_bf16(name, arr): bf16(arr).view(np.uint16).tofile(name)

model = MiniVit().eval()
p = {n: v.detach().numpy() for n, v in model.named_parameters()}

x_float = torch.randn(SEQ, EMBD)

with torch.no_grad():
    ref_out = model.to(torch.bfloat16)(x_float.to(torch.bfloat16)).float().numpy()

# Save inputs (same weight layout as vision_block_bf16.py)
save_bf16("x.data",     x_float.numpy())
save_bf16("wq.data",    p["attn.in_proj_weight"][:EMBD, :].T)
save_bf16("wk.data",    p["attn.in_proj_weight"][EMBD:2*EMBD, :].T)
save_bf16("wv.data",    p["attn.in_proj_weight"][2*EMBD:, :].T)
save_bf16("wo.data",    p["attn.out_proj.weight"].T)
save_bf16("wup.data",   p["ffn_up.weight"].T)
save_bf16("wdown.data", p["ffn_down.weight"].T)
save_bf16("w1.data",    p["ln_1.weight"])
save_bf16("b1.data",    p["ln_1.bias"])
save_bf16("w2.data",    p["ln_2.weight"])
save_bf16("b2.data",    p["ln_2.bias"])
bf16(ref_out).view(np.uint16).tofile("ref_out.data")

print("Test data written.")
print(f"Reference output shape: {ref_out.shape}, sample: {ref_out[0,:4]}")
