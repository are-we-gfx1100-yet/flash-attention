// BSD 3 Clause
// Copyright 2023 Advanced Micro Devices, Inc.
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// #ifdef USE_FLASH_ATTENTION_ROCM

#include "src/fmha.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

namespace pytorch_fmha {

void set_params_fprop(FmhaFpropParams &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t d,
                      // device pointers
                      const at::Tensor& q,
                      const at::Tensor& k,
                      const at::Tensor& v,
                      at::Tensor& out,
                      const at::Tensor& cu_seqlens_q,
                      const at::Tensor& cu_seqlens_k,
                      void *o_tmp_d,
                      void *s_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal,
                      bool is_deterministic) {

    auto acc_type = torch::kFloat32;
    auto data_type = q.dtype();

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = (q.dtype() == at::kBFloat16);

    // S = softmax(P)     //TO DO
    // params.s_ptr = s_d;
    // params.s_stride_in_bytes = get_size_in_bytes(b * h * seqlen_k, data_type);

    // Set the dimensions.
    params.b = b;                 // batch_size
    params.h = h;                 // num_heads
    params.seqlen_q = seqlen_q;   // seqlen q
    params.seqlen_k = seqlen_k;   // seqlen k
    params.d = d;                 // head_dim
    if(cu_seqlens_q.device().type() == c10::kCUDA){
        params.host_seqlens_q = std::vector<int>(params.b+1);
        params.host_seqlens_k = std::vector<int>(params.b+1);
        FMHA_CHECK_HIP(hipMemcpy(params.host_seqlens_q.data(), cu_seqlens_q.data_ptr(), (params.b+1)*sizeof(int), hipMemcpyDeviceToHost));
        FMHA_CHECK_HIP(hipMemcpy(params.host_seqlens_k.data(), cu_seqlens_k.data_ptr(), (params.b+1)*sizeof(int), hipMemcpyDeviceToHost));
    }else{
        params.host_seqlens_q = std::vector<int>(static_cast<int*>(cu_seqlens_q.data_ptr()), static_cast<int*>(cu_seqlens_q.data_ptr())+params.b+1);
        params.host_seqlens_k = std::vector<int>(static_cast<int*>(cu_seqlens_k.data_ptr()), static_cast<int*>(cu_seqlens_k.data_ptr())+params.b+1);
    }

    char* q_ptr = reinterpret_cast<char*>(q.data_ptr());
    char* k_ptr = reinterpret_cast<char*>(k.data_ptr());
    char* v_ptr = reinterpret_cast<char*>(v.data_ptr());

    char* out_ptr = reinterpret_cast<char*>(out.data_ptr());
    char* lse_ptr = reinterpret_cast<char*>(softmax_lse_d);
    char* s_ptr = reinterpret_cast<char*>(s_d);

    for (int i = 0; i < b; i++){
        int temp_seqlen_q = params.host_seqlens_q[i+1] - params.host_seqlens_q[i];
        int temp_q_stride = get_size_in_bytes(d * h * temp_seqlen_q, data_type);
        int temp_seqlen_k = params.host_seqlens_k[i+1] - params.host_seqlens_k[i];
        int temp_k_stride = get_size_in_bytes(d * h * temp_seqlen_k, data_type);
        if(q.is_contiguous()){
            params.q_ptr.push_back(reinterpret_cast<void*>(q_ptr));
            q_ptr = q_ptr + temp_q_stride;
        }else{
            auto q_each_tmp = q.index({torch::indexing::Slice(params.host_seqlens_q[i], params.host_seqlens_q[i+1])}).contiguous();
            params.q_tensors.push_back(q_each_tmp);
            params.q_ptr.push_back(reinterpret_cast<void*>(q_each_tmp.data_ptr()));          
        }
        if(k.is_contiguous()){
            params.k_ptr.push_back(reinterpret_cast<void*>(k_ptr));
            k_ptr = k_ptr + temp_k_stride;
        }else{
            auto k_each_tmp = k.index({torch::indexing::Slice(params.host_seqlens_k[i], params.host_seqlens_k[i+1])}).contiguous();
            params.k_tensors.push_back(k_each_tmp);
            params.k_ptr.push_back(reinterpret_cast<void*>(k_each_tmp.data_ptr()));
        }

        if(v.is_contiguous()){
            params.v_ptr.push_back(reinterpret_cast<void*>(v_ptr));     
            v_ptr = v_ptr + temp_k_stride;
        }else{
            auto v_each_tmp = v.index({torch::indexing::Slice(params.host_seqlens_k[i], params.host_seqlens_k[i+1])}).contiguous();
            params.v_tensors.push_back(v_each_tmp);
            params.v_ptr.push_back(reinterpret_cast<void*>(v_each_tmp.data_ptr()));
        }
        
        params.o_ptr.push_back(reinterpret_cast<void*>(out_ptr));
        out_ptr = out_ptr + temp_q_stride;

        params.softmax_lse_ptr.push_back(reinterpret_cast<void*>(lse_ptr));
        int temp_lse_stride = get_size_in_bytes(h * seqlen_q, acc_type);
        lse_ptr = lse_ptr + temp_lse_stride;

        if(s_d){
            params.s_ptr.push_back(reinterpret_cast<void*>(s_ptr + i * h * seqlen_q * seqlen_k * sizeof(int)));
        }
        else{
            params.s_ptr.push_back(nullptr);
        }
    }

    // Set the different scale values.
    // const float scale_bmm1 = 1.f / sqrtf(d);
    params.scale_bmm1f = softmax_scale;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = p_dropout;
    params.is_causal = is_causal;
    params.is_deterministic = is_deterministic;
}

void set_params_dgrad(FmhaDgradParams &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t h,
                      const size_t d,
                      // device pointers
                      const at::Tensor& q,
                      const at::Tensor& k,
                      const at::Tensor& v,
                      const at::Tensor& y,
                      const at::Tensor& ygrad,
                      at::Tensor &dq,
                      at::Tensor &dk,
                      at::Tensor &dv,
                      const at::Tensor& cu_seqlens_q,
                      const at::Tensor& cu_seqlens_k,
                      void *s_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal,
                      bool is_deterministic,
                      bool is_performance_mode) {

    auto acc_type = torch::kFloat32;
    auto data_type = q.dtype();

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = q.dtype() == at::kBFloat16;

    // params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    // params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // S = softmax(P)
    // params.s_ptr = s_d;
    // params.s_stride_in_bytes = get_size_in_bytes(b * h * seqlen_k, data_type);

    // Softmax sum
    // params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = d;
    if(cu_seqlens_q.device().type()==c10::kCUDA){
        params.host_seqlens_q = std::vector<int>(params.b+1);
        params.host_seqlens_k = std::vector<int>(params.b+1);

        FMHA_CHECK_HIP(hipMemcpy(params.host_seqlens_q.data(), cu_seqlens_q.data_ptr(), (params.b+1)*sizeof(int), hipMemcpyDeviceToHost));
        FMHA_CHECK_HIP(hipMemcpy(params.host_seqlens_k.data(), cu_seqlens_k.data_ptr(), (params.b+1)*sizeof(int), hipMemcpyDeviceToHost));
    }else{
        params.host_seqlens_q = std::vector<int>(static_cast<int*>(cu_seqlens_q.data_ptr()), static_cast<int*>(cu_seqlens_q.data_ptr())+params.b+1);
        params.host_seqlens_k = std::vector<int>(static_cast<int*>(cu_seqlens_k.data_ptr()), static_cast<int*>(cu_seqlens_k.data_ptr())+params.b+1);
    }

    char* q_ptr = reinterpret_cast<char*>(q.data_ptr());
    char* k_ptr = reinterpret_cast<char*>(k.data_ptr());
    char* v_ptr = reinterpret_cast<char*>(v.data_ptr());

    char* dq_ptr = reinterpret_cast<char*>(dq.data_ptr());
    char* dk_ptr = reinterpret_cast<char*>(dk.data_ptr());
    char* dv_ptr = reinterpret_cast<char*>(dv.data_ptr());

    char* y_ptr = reinterpret_cast<char*>(y.data_ptr());
    char* lse_ptr = reinterpret_cast<char*>(softmax_lse_d);
    char* ygrad_ptr = reinterpret_cast<char*>(ygrad.data_ptr());
    
    for (int i = 0; i < b; i++){
        int temp_seqlen_q = params.host_seqlens_q[i+1] - params.host_seqlens_q[i];
        int temp_q_stride = get_size_in_bytes(d * h * temp_seqlen_q, data_type);
        int temp_dq_stride = get_size_in_bytes(d * h * temp_seqlen_q, dq.dtype());
        int temp_seqlen_k = params.host_seqlens_k[i+1] - params.host_seqlens_k[i];
        int temp_k_stride = get_size_in_bytes(d * h * temp_seqlen_k, data_type);
        int temp_dk_stride = get_size_in_bytes(d * h * temp_seqlen_k, dk.dtype());
        if(q.is_contiguous()){
            params.q_ptr.push_back(reinterpret_cast<void*>(q_ptr));
            params.qgrad_ptr.push_back(reinterpret_cast<void*>(dq_ptr));
            q_ptr = q_ptr + temp_q_stride;
            dq_ptr = dq_ptr + temp_dq_stride;
        }else{
            auto q_each_tmp = q.index({torch::indexing::Slice(params.host_seqlens_q[i], params.host_seqlens_q[i+1])}).contiguous();
            auto qgrad_each_tmp = dq.index({torch::indexing::Slice(params.host_seqlens_q[i], params.host_seqlens_q[i+1])}).contiguous();
            params.q_tensors.push_back(q_each_tmp);
            params.qgrad_tensors.push_back(qgrad_each_tmp);
            params.q_ptr.push_back(reinterpret_cast<const void*>(q_each_tmp.data_ptr()));
            params.qgrad_ptr.push_back(reinterpret_cast<void*>(qgrad_each_tmp.data_ptr()));
        }
        if(k.is_contiguous()){
            params.k_ptr.push_back(reinterpret_cast<void*>(k_ptr));
            params.kgrad_ptr.push_back(reinterpret_cast<void*>(dk_ptr));
            k_ptr = k_ptr + temp_k_stride;
            dk_ptr = dk_ptr + temp_dk_stride;
        }else{
            auto k_each_tmp = k.index({torch::indexing::Slice(params.host_seqlens_k[i], params.host_seqlens_k[i+1])}).contiguous();
            auto kgrad_each_tmp = dk.index({torch::indexing::Slice(params.host_seqlens_k[i], params.host_seqlens_k[i+1])}).contiguous();
            params.k_tensors.push_back(k_each_tmp);
            params.kgrad_tensors.push_back(kgrad_each_tmp);
            params.k_ptr.push_back(reinterpret_cast<const void*>(k_each_tmp.data_ptr()));
            params.kgrad_ptr.push_back(reinterpret_cast<void*>(kgrad_each_tmp.data_ptr()));
        }
        if(v.is_contiguous()){
            params.v_ptr.push_back(reinterpret_cast<void*>(v_ptr)); 
            params.vgrad_ptr.push_back(reinterpret_cast<void*>(dv_ptr));
            v_ptr = v_ptr + temp_k_stride;   
            dv_ptr = dv_ptr + temp_dk_stride;
        }else{
            auto v_each_tmp = v.index({torch::indexing::Slice(params.host_seqlens_k[i], params.host_seqlens_k[i+1])}).contiguous();
            auto vgrad_each_tmp = dv.index({torch::indexing::Slice(params.host_seqlens_k[i], params.host_seqlens_k[i+1])}).contiguous();
            params.v_tensors.push_back(v_each_tmp);
            params.vgrad_tensors.push_back(vgrad_each_tmp);
            params.v_ptr.push_back(reinterpret_cast<const void*>(v_each_tmp.data_ptr()));
            params.vgrad_ptr.push_back(reinterpret_cast<void*>(vgrad_each_tmp.data_ptr()));
        }

        params.z_ptr.push_back(nullptr);
        params.y_ptr.push_back(reinterpret_cast<const void*>(y_ptr));
        params.lse_ptr.push_back(reinterpret_cast<const void*>(lse_ptr));
        params.ygrad_ptr.push_back(reinterpret_cast<const void*>(ygrad_ptr));

        int temp_lse_stride = get_size_in_bytes(h * seqlen_q, acc_type);
        y_ptr += temp_q_stride;
        ygrad_ptr += temp_q_stride;
        lse_ptr += temp_lse_stride;
    }

    // Set the different scale values.
    // const float scale_bmm1 = 1.f / sqrtf(d);
    params.scale_bmm1f = softmax_scale;
    //set_alpha(params.scale_bmm1, scale_bmm1, data_type);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = p_dropout;
    params.is_causal = is_causal;
    params.is_deterministic = is_deterministic;
    params.is_performance_mode = is_performance_mode;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd(const at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        at::Tensor &out,
        const at::Tensor &cu_seqlens_q,
        const at::Tensor &cu_seqlens_k,
        const int max_seqlen_q,
        const int max_seqlen_k,
        const float p_dropout,
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        const bool return_softmax,
        const int num_splits) {
    int is_deterministic = 1;

    auto dprops = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentHIPStream().stream();
    bool is_dropout = p_dropout > 0.0;
    LaunchParams<FmhaFpropParams> launch_params(dprops, stream, is_dropout, return_softmax);

    auto q_dtype = q.dtype();

    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16);
    TORCH_CHECK(k.dtype() == q_dtype);
    TORCH_CHECK(v.dtype() == q_dtype);
    TORCH_CHECK(out.dtype() == q_dtype);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32);
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32);

    TORCH_CHECK(q.is_cuda());
    TORCH_CHECK(k.is_cuda());
    TORCH_CHECK(v.is_cuda());
    TORCH_CHECK(out.is_cuda());
    // TORCH_CHECK(cu_seqlens_q.is_cuda());
    // TORCH_CHECK(cu_seqlens_k.is_cuda());

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(k.stride(-1) == 1);
    TORCH_CHECK(v.stride(-1) == 1);
    TORCH_CHECK(out.stride(-1) == 1);
    TORCH_CHECK(cu_seqlens_q.is_contiguous());
    TORCH_CHECK(cu_seqlens_k.is_contiguous());

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = sizes[TOTAL_DIM];
    const int num_heads = sizes[H_DIM];
    const int head_size = sizes[D_DIM];
    const int total_k = k.size(TOTAL_DIM);

    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK((head_size % 8 == 0) && (head_size <= 128));

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads, head_size);
    CHECK_SHAPE(v, total_k, num_heads, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    at::cuda::HIPGuard device_guard{(char)q.get_device()};
    // bool loop = false;

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    // at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = at::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
    // auto softmax_lse = torch::full({batch_size, num_heads, max_seqlen_k}, -std::numeric_limits<float>::infinity(), opts.dtype(at::kFloat));
    at::Tensor s;
    if (return_softmax) { s = torch::empty({ batch_size, num_heads, max_seqlen_q, max_seqlen_k }, opts.dtype(at::kInt)); }
    if (zero_tensors) {
        out.zero_();
        //softmax_lse.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) { s.zero_(); }
    }

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

    set_params_fprop(launch_params.params,
                     batch_size,
                     max_seqlen_q,
                     max_seqlen_k,
                     num_heads,
                     head_size,
                     q, k, v, out,
                     cu_seqlens_q,
                     cu_seqlens_k,
                     nullptr,
                     return_softmax ? s.data_ptr() : nullptr,
                     //return_softmax ? z_device_buf.GetDeviceBuffer() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     is_causal,
                     is_deterministic);

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    

    // at::PhiloxCudaState rng_engine_inputs;

    if( is_dropout ) {
        // See Note [Acquire lock when using random generators]
        int64_t counter_offset = launch_params.params.b * launch_params.params.h * 32;
        std::lock_guard<std::mutex> lock(gen->mutex_);
        launch_params.params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    run_fmha_fp16_bf16_gfx90a(launch_params);

    // TODO EVS: dropout isn't implemented for now
    auto seed_t = at::empty({}, at::dtype(at::kLong));
    auto offset_t = at::empty({}, at::dtype(at::kLong));

    // s -> flash_softmax
    return {softmax_lse, seed_t, offset_t, s};
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd(const at::Tensor &dout,  // total_q x num_heads, x head_size
        const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const at::Tensor &k,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &v,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &out,   // total_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x s softmax logsumexp
        at::Tensor &dq,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        at::Tensor &dk,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        at::Tensor &dv,   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const at::Tensor &cu_seqlens_q,  // b+1
        const at::Tensor &cu_seqlens_k,  // b+1
        const int max_seqlen_q,
        const int max_seqlen_k,          // max sequence length to choose the kernel
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const bool zero_tensors,
        const bool is_causal,
        const int num_splits,
        at::Tensor philox_seed,
        at::Tensor philox_offset
) {
    int is_deterministic = 1;
    int is_performance_mode = 0;

    auto dprops = at::cuda::getCurrentDeviceProperties();

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentHIPStream().stream();
    LaunchParams<FmhaDgradParams> launch_params(dprops, stream, is_dropout, false);

    auto q_dtype = q.dtype();

    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16);
    TORCH_CHECK(k.dtype() == q_dtype);
    TORCH_CHECK(v.dtype() == q_dtype);
    TORCH_CHECK(out.dtype() == q_dtype);
    TORCH_CHECK(dout.dtype() == q_dtype);
    TORCH_CHECK(dq.dtype() == q_dtype);
    TORCH_CHECK(dk.dtype() == q_dtype);
    TORCH_CHECK(dv.dtype() == q_dtype);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32);
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32);

    TORCH_CHECK(q.is_cuda());
    TORCH_CHECK(k.is_cuda());
    TORCH_CHECK(v.is_cuda());
    TORCH_CHECK(out.is_cuda());
    TORCH_CHECK(dout.is_cuda());
    TORCH_CHECK(softmax_lse.is_cuda());
    // TORCH_CHECK(cu_seqlens_q.is_cuda());
    // TORCH_CHECK(cu_seqlens_k.is_cuda());

    TORCH_CHECK(q.stride(-1) == 1);
    TORCH_CHECK(k.stride(-1) == 1);
    TORCH_CHECK(v.stride(-1) == 1);
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(dout.is_contiguous());
    TORCH_CHECK(dq.stride(-1) == 1);
    TORCH_CHECK(dk.stride(-1) == 1);
    TORCH_CHECK(dv.stride(-1) == 1);
    TORCH_CHECK(cu_seqlens_q.is_contiguous());
    TORCH_CHECK(cu_seqlens_k.is_contiguous());

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    const int total_q = sizes[TOTAL_DIM];
    const int num_heads = sizes[H_DIM];
    const int head_size = sizes[D_DIM];
    const int total_k = k.size(TOTAL_DIM);
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK((head_size % 8 == 0) && (head_size <= 128));

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads, head_size);
    CHECK_SHAPE(v, total_k, num_heads, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size);
    CHECK_SHAPE(dq, total_q, num_heads, head_size);
    CHECK_SHAPE(dk, total_k, num_heads, head_size);
    CHECK_SHAPE(dv, total_k, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    // int blocksize_c = (head_size > 64 || (head_size > 32)) ? 128 : 256;
    at::cuda::HIPGuard device_guard{(char)q.get_device()};
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    // at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    // It's possible the softmax_lse_ from the fwd has a different length since blocksize_c could be different.
    // auto softmax_lse = softmax_lse_.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, max_seqlen_q)}).contiguous();

    // at::Tensor softmax_d = at::empty(dq.sizes(), dq.options()).contiguous();
    at::Tensor softmax_d;

    if (zero_tensors) {
        dq.zero_();
        dk.zero_();
        dv.zero_();
        // softmax_d.zero_();
    }
    
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

    if(!is_performance_mode){
        at::Tensor dq_tmp = at::empty(dq.sizes(), dq.options().dtype(at::kFloat)).contiguous();
        at::Tensor dk_tmp = at::empty(dk.sizes(), dk.options().dtype(at::kFloat)).contiguous();
        at::Tensor dv_tmp = at::empty(dv.sizes(), dv.options().dtype(at::kFloat)).contiguous();
        dq_tmp.zero_();
        dk_tmp.zero_();
        dv_tmp.zero_();
        set_params_dgrad(launch_params.params,
                        batch_size,
                        max_seqlen_q,
                        max_seqlen_k,
                        num_heads,
                        head_size,
                        q, k, v, out,
                        dout, dq_tmp, dk_tmp, dv_tmp,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        nullptr,
                        softmax_lse.data_ptr(),
                        p_dropout,
                        softmax_scale,
                        is_causal,
                        is_deterministic,
                        is_performance_mode);
        
        if( is_dropout ) {
            // See Note [Acquire lock when using random generators]
            int64_t counter_offset = launch_params.params.b * launch_params.params.h * 32;
            std::lock_guard<std::mutex> lock(gen->mutex_);
            launch_params.params.philox_args = gen->philox_cuda_state(counter_offset);
        }

        run_fmha_dgrad_fp16_bf16_gfx90a(launch_params);
        if(!q.is_contiguous()){
            dq_tmp.copy_(torch::cat(launch_params.params.qgrad_tensors, 0).contiguous(), true);
        }
        if(!k.is_contiguous()){
            dk_tmp.copy_(torch::cat(launch_params.params.kgrad_tensors, 0).contiguous(), true);
        }
        if(!v.is_contiguous()){
            dv_tmp.copy_(torch::cat(launch_params.params.vgrad_tensors, 0).contiguous(), true);
        }

        dq.copy_(dq_tmp, true);
        dk.copy_(dk_tmp, true);
        dv.copy_(dv_tmp, true);
    }else{
        set_params_dgrad(launch_params.params,
                         batch_size,
                         max_seqlen_q,
                         max_seqlen_k,
                         num_heads,
                         head_size,
                         q, k, v, out,
                         dout, dq, dk, dv,
                         cu_seqlens_q,
                         cu_seqlens_k,
                         nullptr,
                         softmax_lse.data_ptr(),
                         p_dropout,
                         softmax_scale,
                         is_causal,
                         is_deterministic,
                         is_performance_mode);
        
        if( is_dropout ) {
            // See Note [Acquire lock when using random generators]
            int64_t counter_offset = launch_params.params.b * launch_params.params.h * 32;
            std::lock_guard<std::mutex> lock(gen->mutex_);
            launch_params.params.philox_args = gen->philox_cuda_state(counter_offset);
        }

        run_fmha_dgrad_fp16_bf16_gfx90a(launch_params);

        if(!q.is_contiguous()){
            dq.copy_(torch::cat(launch_params.params.qgrad_tensors, 0), true);
        }
        if(!k.is_contiguous()){
            dk.copy_(torch::cat(launch_params.params.kgrad_tensors, 0), true);
        }
        if(!v.is_contiguous()){
            dv.copy_(torch::cat(launch_params.params.vgrad_tensors, 0), true);
        }
    }
    return { dq, dk, dv, softmax_d };
}

} // namespace pytorch_fmha

// #endif
