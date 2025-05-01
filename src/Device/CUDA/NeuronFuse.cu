
// #include "../ggex/GG_util.hpp"       //ugly  "__builtin_ia32_ldtilecfg" is undefined
#include "./cuda_common.h"
#include "./cublas_common.h"
#include "./llm_c/matmul.cuh"
#include "./llm_c/layernorm.cuh"
#include "./llm_c/encoder.cuh"
#include "./llm_c/fused_classifier.cuh"
// #include "./TE/fused_attn/fused_attn_fp8.cu"
#include "../../Manifold/Neuron.hpp"
#include "../../Manifold/Fish.hpp"
#include "./EDevice.hpp"
// #include "./mfu.h"
#define NOMINMAX

cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
cublasLtHandle_t cublaslt_handle;
void* cublaslt_workspace = NULL;
cudaStream_t main_stream=nullptr;
cudaDeviceProp deviceProp;


int EDGE_DEVICES::GPU_::MAX_COUNT = 16;     //  16    
std::vector<EDGE_DEVICES::GPU_> EDGE_DEVICES::GPU_::cudaGetDevice(int flag) {
    std::vector<GPU_> devices;
#ifdef __HIP_PLATFORM_AMD__
    // Workaround for a rocBLAS bug when using multiple graphics cards:
    // https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1346
    {
        int major_version = 0;
        size_t version_length = 0;
        if (rocblas_get_version_string_size(&version_length) == rocblas_status_success) {
            std::vector<char> version(version_length+1, '\0');
            if (rocblas_get_version_string(version.data(), version.size()) == rocblas_status_success) {
                version.resize(::strlen(version.data()));
                int parsed_value = 0;
                if (std::from_chars(version.data(), version.data() + version.size(), parsed_value).ec == std::errc()) {
                    major_version = parsed_value;
                }
            }
        }
        if (major_version < 4) {
            _LOG_DEBUG(_CUDA_NAME " calling rocblas_initialize as a workaround for a rocBLAS bug\n");
            rocblas_initialize();
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
#endif
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);    //CUDA functions do not throw exceptions, why?
    if (err != cudaSuccess) {
        _ERROR("%s: failed to initialize CUDA: %s\n", __func__, cudaGetErrorString(err));
        return devices;
    }
    // assert(device_count <= GPU_DEVICE::MAX_COUNT);

    int64_t total_vram = 0;
#ifdef _CUDA_FORCE_MMQ
    _INFO("%s: _CUDA_FORCE_MMQ:    yes\n", __func__);
#else
    _INFO("%s: _CUDA_FORCE_MMQ:    no\n", __func__);
#endif // _CUDA_FORCE_MMQ
#ifdef _CUDA_FORCE_CUBLAS
    _INFO("%s: _CUDA_FORCE_CUBLAS: yes\n", __func__);
#else
    _INFO("%s: _CUDA_FORCE_CUBLAS: no\n", __func__);
#endif // _CUDA_FORCE_CUBLAS
    _INFO("%s: found %d CUDA devices:\n", __func__, device_count);
    devices.resize(device_count);
    for (int id = 0; id < device_count; ++id) {
        int device_vmm = 0;

#if defined(_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = id;
            CU_CHECK(cuMemGetAllocationGranularity(&devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(_USE_VMM)
        devices[id].vmm = !!device_vmm;
        cudaDeviceProp prop;
        cudaCheck(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
        devices[id].nsm       = prop.multiProcessorCount;
        devices[id].smpb      = prop.sharedMemPerBlock;
        devices[id].warp_size = prop.warpSize;
#if defined(_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
        devices[id].smpbo = prop.sharedMemPerBlock;

        devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((devices[id].cc & 0xff00) == 0x0) {
            _LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                devices[id].cc = _CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                devices[id].cc += prop.minor * 0x10;
            }
        }
        _INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d\n",
                        id, prop.name, prop.gcnArchName, devices[id].cc & 0xffff,
                        device_vmm ? "yes" : "no", prop.warpSize);
#elif defined(_USE_MUSA)
        // TODO: refine the .cc to reflect MUSA's actual CC capabilities
        devices[id].smpbo = prop.sharedMemPerBlockOptin;
        devices[id].cc = 100*prop.major + 10*prop.minor;
        _INFO("  Device %d: %s, compute capability %d.%d, VMM: %s\n",
                        id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");
#else
        devices[id].smpbo = prop.sharedMemPerBlockOptin;
        devices[id].cc = 100*prop.major + 10*prop.minor;
        _INFO("  Device %d: %s, compute capability %d.%d, VMM: %s\n",
                        id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");
#endif // defined(_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    }

    return devices;
}



hGTensor huTensor::_Multiply(const hGTensor& b) {
    huTensor *cuB=dynamic_cast<huTensor *>(b.get());
    assert(cuB!=nullptr);
    return nullptr;
}

bool TokenEmbed::UpdateBucket(int type,int flag){
    num_c_groups = CEIL_DIV(C, (WARP_SIZE * x128::size));
    if (bucket_info != NULL)
        return false;
    
    assert((size_t)(B * T) * num_c_groups < (1ULL<<31ULL)); // todo - maybe an issue for llama3-400B(?)
    workload_indices = new int[B * T * num_c_groups];
    bucket_info = new int4[B * T * num_c_groups];
    return true;
}
void TokenEmbed::WorkloadOnBucker(int *inputs_cpu,int flag ){
    // if(num_buckets>0) 
    //     return;

    // Step 1: Sort inputs into buckets
    int total_items = 0;
    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    for (uint64_t bt = 0; bt < B * T; bt++) {
        for (uint64_t c_group = 0; c_group < num_c_groups; c_group++) {
            // todo - passing c_group/inputs_cpu[bt] in data to avoid a second hash lookup is a bit hacky
            uint64_t data = bt + (c_group<<32ULL) + ((uint64_t)inputs_cpu[bt]<<42ULL);
            buckets[c_group + num_c_groups * inputs_cpu[bt]].push_back(data);
            total_items++;
        }
    }
    // Step 2: Sort buckets by size in descending order
    // this is so the largest buckets are processed first by the GPU
    // otherwise, if they started late, they would still be running with the rest of the GPU idle
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sortedBuckets(buckets.begin(), buckets.end());
    std::sort(sortedBuckets.begin(), sortedBuckets.end(), // ugly because we don't have a typedef for the std::pair
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a, const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();
              });

    num_buckets = buckets.size();
    int bucket_index = 0;
    int workload_index = 0;
    for (const auto& bucket : sortedBuckets) {
        bucket_info[bucket_index].x = workload_index; // bucket start
        bucket_info[bucket_index].y = bucket.second.size(); // bucket size
        bucket_info[bucket_index].z = (bucket.second[0] >> 42ULL) & ((1ULL<<20ULL)-1); // bucket ix
        bucket_info[bucket_index].w = (bucket.second[0] >> 32ULL) & ((1ULL<<10ULL)-1); // bucket c

        for (uint64_t idx : bucket.second) {
            workload_indices[workload_index++] = (int)(idx & ((1ULL<<31ULL)-1ULL));
        }
        bucket_index++;
    }

    floatX *scratch=(floatX *)GTensor::buff;
    // Step 3: Copy data from host to device (async until the last one to avoid synchronising CPU/GPU twice)
    // todo - could use CUDA events (even without streams) to avoid CPU/GPU synchronisation completely
    int4* d_bucket_info = (int4*)scratch;
    int*  d_workload_indices = (int*)(scratch + B*T*num_c_groups * sizeof(int4));
    cudaCheck(cudaMemcpyAsync(d_bucket_info, bucket_info, num_buckets * sizeof(int4), cudaMemcpyHostToDevice, main_stream));
    cudaCheck(cudaMemcpyAsync(d_workload_indices, workload_indices, total_items * sizeof(int), cudaMemcpyHostToDevice, main_stream));
}


hGTensor TokenEmbed::OnEmbed(const int* tokens, int seed){
try{
    int OC=w->ne[1],Vp=padded_nCls;  
    hGTensor cur = out;
    if(isForward()){
        grid_size = CEIL_DIV(B*T*C, block_size);
        // encoder_forward(ToX(cur), tokens, ToX(w), ToX0(b), B, T, C, main_stream);
        CU_embed_forw_<<<grid_size, block_size, 0, main_stream>>>(ToX(cur), tokens, ToX(w), ToX0(b), B, T, C);
        w->Print("wte",0,0);        //ToX(w),true,Vp,C
        PrintTensor<floatX>("wpe",ToX0(b),true,T,C);
        PrintTensor<int>("inputs",tokens,true,B,T);            PrintTensor<floatX>("GetRow",ToX(cur),true,B,T,C);
        if(maec!=nullptr){
            maec->ENC(cur);
        }
    }else{        
        UpdateBucket(0x0);
        WorkloadOnBucker(hBatch->host,0x0);
        floatX *scratchX=(floatX *)GTensor::buff;
        hGTensor delta =GTensor::delta,cur=delta;  
        if(maec!=nullptr){
            cur = maec->ENC(cur);
        }     
        // encoder_backward_1(ToG(w), ToG0(b), ToX(cur), tokens, B, T, C, seed, main_stream); 
            encoder_backward(ToG(w), ToG0(b), scratchX, workload_indices, bucket_info, ToX(cur), tokens, hBatch->host, B, T, C, seed, main_stream);
            
    // PrintTensor<floatX>("grad of wte",grads.wte,true,Vp,C);         PrintTensor<float>("losses",acts.losses,true,B,T);
    // PrintTensor<floatX>("grad of wpe",grads.wpe,true,T,C);
    }
    return cur;
}catch(...){
    assert(0);
    return nullptr;
}
}

//  seed - use stochastic rounding to go from FP32 to BF16
hGTensor TokenEmbed::SubW(hGTensor hSamp,bool isForw,  hGTensor wOut, int flag){
try{
    int nSamp = hSamp->size(),*samps=TO<int>(hSamp),nLayer=hFish->config.nLayer();
    int OC=w->ne[1],Vp=padded_nCls,seed=42,T=nSamp,B=1;  
    grid_size = CEIL_DIV(B*T*C, block_size);
    hGTensor cur = wOut,wSrc=flag==0 ? w : wInv;
    
    if(isForw){
        // encoder_forward(ToX(cur), samps, ToX(wSrc), nullptr, 1, T, C, main_stream);
        CU_embed_forw_<<<grid_size, block_size, 0, main_stream>>>(ToX(cur), samps, ToX(wSrc), T, C,Vp,flag==1);    
        cur->Print("subW",0,0);     
    }else{             
        CU_embed_back_<<<grid_size, block_size, 0, main_stream>>>(ToG(wSrc), samps, ToX(cur), T, C,Vp,1.0,flag==1);    
        // encoder_backward(ToG(wSrc), nullptr, scratchX, workload_indices, bucket_info, ToX(cur), samps, hBatch->host, 1, T, C, seed, main_stream);     
    }
    return cur;
}catch(...){
    assert(0);
    return nullptr;
}
}
int SLP::Forw(hGTensor rhs_0,hGTensor lhs_,hGTensor gelu,int flag){
try{
    floatX *rhs=ToX(rhs_0),*pre_gelu = ToX0(gelu),*wX=ToX(w);//,*inp=ToX(lhs_);
    int OC=nOut,IC=nIn;
    // assert(C==w->ne[0]);
    assert(rhs_0->size()>=B*T*OC);        //  ne of scatch
    float* dbias_buffer=nullptr;
    inp = lhs_;
    bool transAW = true;
    // if(isTransW)        
    //     transAW = false;
    // matmul_forward_cublaslt(rhs, inp, wX, ToX0(b), B, T, C, OC, main_stream,pre_gelu,gelu_fusion);
    if(compression==SAMPLE && subw!=nullptr){
        subw->SubW(hSamps,true,GTensor::tmpW,samp_type);
        wX = ToX(GTensor::tmpW);        //assert(nSample==OC || nSample==IC);
        // GTensor::tmpW->Print("subW",0,-1);    
        // encoder_forward(wX, samples, ToX(w), nullptr, 1, nSample, C, main_stream);
    }
    if (gelu_fusion < 1 && pre_gelu) {
        matmul_cublaslt(pre_gelu, wX, ToX(lhs_), ToX0(b), OC, B*T, IC, main_stream, transAW, false, 0, 0, 0, 0, false, NULL, false);
        gelu_forward(rhs, pre_gelu, B*T*OC, main_stream);
    } else {
        matmul_cublaslt(rhs, wX, ToX(lhs_), ToX0(b), OC, B*T, IC, main_stream, transAW, false, 0, 0, 0, 0, false, pre_gelu, false);
    }
    if(compression==SAMPLE) {
        // rhs_0->Print(rhs_0->name,0,-1);
    }
        // PrintTensor<floatX>("l_qkvw",l_qkvw,true,3*C,C);       PrintTensor<floatX>("l_qkvb",l_qkvb,true,3*C,1);
        // PrintTensor<floatX>("l_qkvr",l_qkvr,true,B,T,3*C);
    
    return 0x0;
}catch(...){
    assert(0);
    return -1;
}
}
int SLP::Back(hGTensor delta,hGTensor inp,hGTensor deltaIn,hGTensor gelu,float* dbias_buffer,int flag){
try{
    floatX *pre_gelu = ToX0(gelu),*wX=ToX(w),*gW=ToG(w);
    int OC=nOut,IC=nIn;     
    assert(delta!=nullptr);
    deltaIn->Print("delta_in",0,flag);
    if(compression==SAMPLE && subw!=nullptr){    //remater to get wX
        subw->SubW(hSamps,true,GTensor::tmpW,samp_type);
        wX = ToX(GTensor::tmpW);        //assert(nSample==OC || nSample==IC);
        gW = ToX(GTensor::tmpGW);       
        cudaCheck(cudaMemsetAsync(gW, 0, GTensor::tmpGW->nByte(), main_stream));
    }
    matmul_backward(ToX(delta), gW, ToG0(b),ToX(deltaIn),ToX(inp), wX, dbias_buffer, B, T, IC, OC, main_stream,isTransW, pre_gelu);
    if(compression==SAMPLE && subw!=nullptr){
        subw->SubW(hSamps,false,GTensor::tmpGW,samp_type);
    } 
    delta->Print("delta",0,flag);
    return 0x0;
}catch(...){
    assert(0);
    return -1;
}
}
int SLP::FUSE_cuda_block(hGTensor rhs,hGTensor lhs,hGTensor gelu,bool isForw,int flag){
    return 0x0;
}

//  hIn = QKV->out
hGTensor FFN::FUSE_cuda(hGTensor hIn,floatX *scratch,int flag){
    floatX *ff2=ToX(down.out),*ff1=ToX(up.out);
    // gelu just inplace operation on ff1, maybe could share memory!    
    hGTensor tGelu=GTensor::tmpFF1;    
    tGelu = GTensor::scratch;
    bool isBias = up.b!=nullptr;  
    
    if(isForward()){  
        if(fuseNorm==nullptr){
            norm.FUSE_cuda(hIn);       
        }
        floatX * inp1_ = ToX(norm.out);         
        if(remater_ffn)  {
            input_1 = inp1_;
            ff1=ToX(GTensor::tmpFF1);              
        } 
        assert(ff1!=nullptr);       // ff1=gelu_forward(out, l_fch_gelu, B*T*OC, stream);
        floatX *scratch = ToX(GTensor::delta);    
        if(!gate.Empty()){
            gate.Forw(tGelu,norm.out,remater_ffn?GTensor::tmpFF1:up.out);        
        }
        up.Forw(tGelu,norm.out,remater_ffn?GTensor::tmpFF1:up.out);        
        // PrintTensor<floatX>("inp1",ToX(norm.out),true,B,T,C,1,-1);          PrintTensor<floatX>("ff1",ff1,true,B,T,latent,1,-1);  
        down.Forw(GTensor::delta,tGelu,nullptr,isSymmetric);       
        // PrintTensor<floatX>("inp1",ToX(norm.out),true,B,T,C,1,-1);
        PrintTensor<floatX>("ffn",scratch,true,B,T,C);

        // fused_residual_forward5(ToX(out), normed,mean,rstd, ToX(hIn), scratch, ToX(fuseNorm->w), xb, B*T, C, main_stream);
        residual_forward(ToX(out), ToX(hIn), scratch, B*T*C, main_stream);
        if(fuseNorm!=nullptr){
            return fuseNorm->FUSE_cuda(out);   
            // layernorm_forward(ToX(fuseNorm->out), TO<float>(fuseNorm->mean),TO<float>(fuseNorm->rstd), ToX(out),ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T, 1, C, main_stream);
            // return fuseNorm->out;
        }
        
        // PrintTensor<floatX>("inp1",ToX(norm.out),true,B,T,C,1,-1);
        out->Print("residual3",0,0);
    }else{
        assert(delta!=nullptr);
        // floatX *dl_bt4c = ToX(GTensor::bt4c),*dresidual = ToX(GTensor::delta),*gNb=norm.b==nullptr?nullptr:ToG(norm.b); 
        float*  scratchF = (float*) scratch;   // not the same inp1 of forward !!!
        if(input_1!=nullptr){
            input_1 =  ToX(norm.out);
            ff1=ToX(GTensor::tmpFF1);  
            up.Forw(tGelu,norm.out,GTensor::tmpFF1);            
            // fuMM(l_fch_gelu,input_1, (floatX*)up.w->data, ToX0(up.b), B, T, C, latent, main_stream, ff1, gelu_fusion);
            // norm.out->Print("inp1",0,-1);          PrintTensor<floatX>("ff1",ff1,true,B,T,latent,-1);  
        }else
            gelu_forward(ToX(tGelu), ff1, B*T*latent, main_stream);  
        assert(ff1!=nullptr);   
        down.Back(GTensor::bt4c,tGelu,GTensor::delta,GTensor::tmpFF1,scratchF);
        // matmul_backward(dl_bt4c, ToG(down.w), ToG0(down.b), dresidual, ToX(tGelu), ToX(down.w), scratchF, B, T, latent, C, main_stream, ff1, gelu_fusion);
        // PrintTensor<floatX>("back of ffn1",dl_bt4c,true,B,T,latent);
        up.Back(delta,norm.out,GTensor::bt4c,nullptr,scratchF);
        // matmul_backward(ToX(delta), ToG(up.w), ToG0(up.b), dl_bt4c, ToX(norm.out), ToX(up.w), scratchF, B, T, C, latent, main_stream);
        // // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        // norm.FUSE_cuda(residual,scratchF,tmpDelta);
        float *_mean = norm.mean==nullptr?nullptr : TO<float>(norm.mean);
        layernorm_backward(ToX(GTensor::delta), ToG(norm.w), ToG0(norm.b), scratchF, ToX(delta), ToX(hIn), ToX(norm.w), _mean, TO<float>(norm.rstd), B, T, C, main_stream);
        // lastQKV->proj_cat.Back(delta,lastQKV->attn,GTensor::delta,nullptr,scratchF);        // matmul_backward(ToX(delta), ToG(lastQKV->proj_cat.w), ToG0(lastQKV->proj_cat.b), ToX(GTensor::delta), ToX(lastQKV->attn), ToX(lastQKV->proj_cat.w), scratchF, B, T, C, C, main_stream);
        delta->Print("back of ffn0",0,0);
    }
    
    return out;
}

/*
    layernorm_forward(floatX* out, float* mean, float* rstd, floatX* inp, const floatX* weight, const floatX* bias,         int B, int T, int C, cudaStream_t stream)
    layernorm_backwar(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,const floatX* dout, const floatX* inp, const floatX* weight, const float* mean, const float* rstd,          int B, int T, int C, cudaStream_t stream)
*/
hGTensor huTensor::Normal(hGTensor hOut,hGTensor _mean,hGTensor _rstd,hGTensor w,hGTensor b,bool isForward,int flag) {
    assert(!hOut->isEmpty());
    int B=hOut->ne[0],T=hOut->ne[1],C=w->ne[0];
    // assert(b!=nullptr);     
    floatX *weight=(floatX*)(w->data),*bias=ToX0(b);    //b==nullptr?nullptr:(floatX*)(b->data);    
    floatX *out=(floatX*)(hOut->data); // (B, T, C)
    if(isForward)
        layernorm_forward(out, (float*)_mean->data, (float*)_rstd->data, (floatX *)data,weight,bias, B, T, C, main_stream);
    else{
        layernorm_backward(nullptr, (floatX*)(w->grad), ToG0(b), nullptr, nullptr,nullptr, weight, 
            (float*)_mean->data, (float*)_rstd->data, B, T, C, main_stream);
    }
    
    return hOut;
}

hGTensor LayerNormal::FUSE_cuda(hGTensor inpL,float* scratch,hGTensor deltaIn,int flag) {
    float* _mean = mean==nullptr ? nullptr : TO<float>(mean);
    if(isForward()){    //cur = cur->Normal(out,mean,rstd,w,b); 
        inp = inpL;  
        layernorm_forward(ToX(out), _mean,  TO<float>(rstd), ToX(inpL),ToX(w),ToX0(b), B, T, C, main_stream);
    }        
    else{   
        assert(deltaIn!=nullptr);       // const floatX* deltaIn=ToX(GTensor::bt4c);
        // floatX* dresidual = ToX(GTensor::delta);
        layernorm_backward(ToX(delta), ToG(w), ToG0(b), scratch, ToX(deltaIn),ToX(inpL), ToX(w), _mean,  TO<float>(rstd), B, T, C, main_stream);
        delta->Print("back of normal",0,0);
    }
    return out;
}

//void fused_classifier(Type* logits, float* cuLoss,const float dloss, const int* targets,int B, int T, int V, int P, std::bool_constant<WriteDLogits> write_dlogits, cudaStream_t stream) {
//float huTensor::FusedLoss(float dloss,hGTensor hLoss,hGTensor hTarget,hGTensor hLastLayer, hGTensor w,int V,bool isForward,int flag){
hGTensor OutCLS::FUSE_cuda(hGTensor inp,int flag)   {
    int V=nCls,Vp=padded_nCls, gelu_fusion=1;
    assert(proj.b==nullptr);
    mean_loss = 0.0f;
    const int *targets = (int*)(target->data);
    float* cuLoss = (float*)out->data;  
    hGTensor cur = preLogits,w = proj.w;  //==nullptr?token_embed:proj.w;    
    if(isForward()){        
        if(maec!=nullptr){
            inp = maec->DEC(inp,true);   C = inp->ne[2];
        }   
        floatX *z0=ToX(inp),*pre_gelu=nullptr;  //* errLogits = ToX(preLogits),
        cudaCheck(cudaMemset(cuLoss, 0, B*T*sizeof(float)));
        assert( target->isSameShape(out) );
        constexpr std::bool_constant<true> cuFalse;    
        for(size_t i=0;i<B;i+=dB){
            size_t off=i*T*Vp,n1=i*T,nZ=i*T*C;
            off=0;      //reduce memory            
            // fuMM(ToX(cur)+off, z0+nZ, ToX(w), NULL, dB, T, C, Vp, main_stream);  //[32,1024,50304]=[32,1024,768]*[768,50304]
            matmul_cublaslt(ToX(cur)+off, ToX(w), z0+nZ, NULL, Vp, dB*T, C, main_stream, true, false, 0, 0, 0, 0, false);
            fused_classifier(ToX(cur)+off, cuLoss+n1, rLoss, targets+n1, dB, T, V, Vp, cuFalse, main_stream);        //target=[32,1024]
            if(ToG0(w)!=nullptr && delta!=nullptr){
                matmul_cublaslt(ToX(delta)+nZ, ToX(w), ToX(cur)+off, NULL, C, dB*T, Vp, main_stream, false, false, 0, 0, 0, 0, false,gelu_fusion >= 2 ? pre_gelu : NULL, true);   
                matmul_cublaslt(ToG(w), z0+nZ, ToX(cur)+off, NULL /*dbias*/, C, Vp, dB*T, main_stream, false, true, 0, 0, 0, 0,true /* accumulate */, NULL, true);                
            }                         
        }
        // fused_classifier(errLogits, cuLoss, rLoss, targets, B, T, V, Vp, cuFalse, main_stream);        //target=[32,1024]
        cudaCheck(cudaMemcpy(hostLoss, cuLoss, B * T * sizeof(float), cudaMemcpyDeviceToHost));                 
        cudaCheck(cudaDeviceSynchronize());
        w->Print("oucls.proj.w",1,-1);
         
        /*if(flag==0x1001 && gw!=nullptr && errOut!=nullptr){            //matmul_backward(errOut, gw, NULL, errLogits, z0, w, NULL, B, T, C, Vp, main_stream);      //accumulate=true  
            matmul_cublaslt(errOut, w, errLogits, NULL, C, B*T, Vp, main_stream, false, false, 0, 0, 0, 0, false,gelu_fusion >= 2 ? pre_gelu : NULL, true);
            if (gelu_fusion < 2 && pre_gelu) {
                gelu_backward_inplace(errOut, pre_gelu, B*T*C, main_stream);
            }
            matmul_cublaslt(gw, z0, errLogits, NULL , C, Vp, B*T, main_stream, false, true, 0, 0, 0, 0,true , NULL, true);
        }*/
            
        for (int i = 0; i < B*T; i++) {
            assert(!std::isnan(hostLoss[i]));
            mean_loss += hostLoss[i];
        }   
        mean_loss /= B*T;
    }else{        
        // matmul_backward(errOut, gw, NULL, errLogits, z0, w, NULL, B, T, C, Vp, main_stream);
        if(maec!=nullptr){
            cur = maec->DEC(delta,false);
            return cur;
        }  else
            return delta;
    }
    cudaCheck(cudaGetLastError());
    return preLogits;
}

huTensor::~huTensor()  {
    Free();

}

