
import torch
import torch.nn.functional as F


query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")

with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True):
  ref_result = F.scaled_dot_product_attention(query,key,value)

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
  out_result = F.scaled_dot_product_attention(query,key,value)

print(torch.isclose(ref_result, out_result, atol=1e-03))
