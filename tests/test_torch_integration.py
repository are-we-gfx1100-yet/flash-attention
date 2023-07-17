
import torch
import torch.nn.functional as F

import torch.utils.benchmark as benchmark


def benchmark_forward(fn, *inputs, repeats=10, desc='', verbose=True, amp=False,
                      amp_dtype=torch.float16, **kwinputs):
    """ Use Pytorch Benchmark on the forward pass of an arbitrary function. """
    if verbose:
        print(desc, '- Forward pass')
    def fn_amp(*inputs, **kwinputs):
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)
    for _ in range(repeats):  # warmup
        fn_amp(*inputs, **kwinputs)
    t = benchmark.Timer(
            stmt='fn_amp(*inputs, **kwinputs)',
            globals={'fn_amp': fn_amp, 'inputs': inputs, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")

with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True):
  ref_result = F.scaled_dot_product_attention(query,key,value)

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
  out_result = F.scaled_dot_product_attention(query,key,value)

print(torch.allclose(ref_result, out_result, atol=1e-03))

benchmarks = []

def flash_func(q, k, v):
  with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    F.scaled_dot_product_attention(query,key,value)

def math_func(q, k, v):
  with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True):
    F.scaled_dot_product_attention(query,key,value)

flash_time, flash_measurement = benchmark_forward(flash_func, query, key, value, repeats=50, desc='Flash Attention')
math_time, math_measurement = benchmark_forward(math_func, query, key, value, repeats=50, desc='PyTorch Math Attention')

relative_perf = ((math_measurement.mean - flash_measurement.mean) / math_measurement.mean) * 100

benchmarks.append([math_measurement.mean, flash_measurement.mean, relative_perf])

print(f'Flash Attention Speedup: {relative_perf}\n')
