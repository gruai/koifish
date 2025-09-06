/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief
 *  \author Yingshi Chen
 */

#pragma once

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif
#include "../CLI_params.hpp"
#include "../g_stddef.hpp"
#include "../Tensor/GTensor.hpp"

#if defined(GGML_USE_ACCELERATE)
    #include <Accelerate/Accelerate.h>
#if defined(GGML_USE_CLBLAST) // allow usage of CLBlast alongside Accelerate functions
    #include "ggml-opencl.h"
#endif
#elif defined(GGML_USE_OPENBLAS)
    #if defined(GGML_BLAS_USE_MKL)
        #include <mkl.h>
    #else
        #include <cblas.h>
    #endif
#elif defined(GGML_USE_CUBLAS)
    #include "ggml-cuda.h"
#elif defined(GGML_USE_CLBLAST)
    #include "ggml-opencl.h"
#elif defined(GGML_USE_VULKAN)
    #include "ggml-vulkan.h"
#elif defined(GGML_USE_SYCL)
    #include "ggml-sycl.h"
#endif

#define TO_DO   assert(0);

static char buffer[GTensor::MAX_NAME];
static const char * _NAM_( const char *format,... )	{
    va_list args;
    va_start( args, format );
    vsnprintf( buffer,GTensor::MAX_NAME,format,args );
    va_end(args);
    return buffer;
}
//return tensor name
static const char * TN( const char *format,... )	{
    va_list args;
    va_start( args, format );
    vsnprintf( buffer,GTensor::MAX_NAME,format,args );
    va_end(args);
    assert(strlen(buffer)+strlen(".weight")<=GTensor::MAX_NAME);
    return strcat(buffer,".weight");
}

static const char * TNs( const char *format,const char *suffix,... )	{
    va_list args;
    va_start( args, suffix );       //  va_start( args, format );
    vsnprintf( buffer,GTensor::MAX_NAME,format,args );
    va_end(args);
    const char*w = ".weight";
    if(strlen(buffer) > 7 && strcmp(buffer+strlen(buffer)-7, ".weight")==0){
        w = "";
    }
    assert(strlen(buffer)+strlen(w)+strlen(suffix)<=GTensor::MAX_NAME);
    strcat(buffer,w);
    strcat(buffer,suffix);        
    return buffer;
}

#define GGUF_GET_KEY(ctx, dst, func, type, req, key) \
{ \
    const std::string skey(key); \
    const int kid = gguf_find_key(ctx, skey.c_str()); \
    if (kid >= 0) { \
        enum gguf_type ktype = gguf_get_kv_type(ctx, kid); \
        if (ktype != (type)) { \
            die_fmt("key %s has wrong type: %s", skey.c_str(), gguf_type_name(ktype)); \
        } \
        (dst) = func(ctx, kid); \
    } else if (req) { \
        die_fmt("key not found in model: %s", skey.c_str()); \
    } \
}

enum GD_METHOD {
    ADAMw=0x0,          
    SGD,               
    SGD_v,
    SGD_blk_v,
    SGD_HYBRID,   
    LION,
    ADAM_MINI,          //  https://arxiv.org/pdf/2406.16793
    ADAM_S,             //  https://arxiv.org/pdf/2505.16363
    ADAM_GALORE, 
    ADAM_spike, 
    MUON,
    // ADAMw_cuda,
};
static string GD_NAME[]={
    "ADAMw","SGD","SGD_v","SGD_blk_v","SGD_HYBRID","LION","ADAM_MINI","ADAM_S","ADAM_GALORE","MUON"
}; 

void _T_repr_(hGensor t,const char*tab,char *buf,int typ=0x0);
void _T_repr_(hGensor t,const char*tab,char *buf,const GENSOR_INFO&info);


extern "C"  inline void gg_print_tensor_(const char* title, hGensor t, int n = 10);

//BLAS has better performance
template <typename T> 
double NRM_2( const T *X,size_t dim )		{	
    double sum=0;
    const T *x=X;
    for(size_t i=0;i<dim;i++,x++){
        sum += (*x)*(*x);
    }
    return sqrt(sum);
}
double P_softmax(int idx, float *logits, int size);
float SOFT_MAX(const int n, float * y, const float * x);

//Py=Py-Px
float SOFT_MAX_minus(const int n, float * y, const float * x);
inline float SOFT_MAX(std::vector<float>&x){
    size_t n=x.size();
    float *y=new float[n];
    float sum = SOFT_MAX(n,y,x.data());
    memcpy(x.data(),y,sizeof(float)*n);
    delete[] y;
    return sum;
}

#define GGML_FILE_MAGIC   0x67676d6c // "ggml"
inline std::ifstream GGML_Load(const std::string&mode_path,int flag=0x0) {
    fprintf(stderr, "%s: loading model from '%s' - please wait ...\n", __func__, mode_path.c_str());    
    std::ifstream fin = std::ifstream(mode_path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, mode_path.c_str());
        return fin;
    }
    uint32_t magic;
    fin.read((char *) &magic, sizeof(magic));
    if (magic != GGML_FILE_MAGIC) { //0x67676d6c
        fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, mode_path.c_str());
        return fin;
    }
    return fin;
}

void Gensor2float_(const hGensor w,float *A,int flag=0x0);
inline float *Gensor2float(struct ggml_context * ctx0,const hGensor w,int flag=0x0)   {
    size_t ne00 = tELEM(w),nbyte = tBYTE(w); 
    void *data_0 = w->data;
    float *A=new float[ne00];
    Gensor2float_(w,A,flag);    
    return A;
}


/*must called after ggml_build_forward/backwrad_expand
inline void ggml_graph_stat(struct ggml_cgraph * cgraph, int flag=0x0) {
    if(cgraph==nullptr)     return;
    
    int i,nT=cgraph->n_leafs+cgraph->n_nodes,nQ=0,nF16=0;
       
    for(i=0;i<cgraph->n_leafs;i++)  {
        auto type = cgraph->leafs[i]->type;
        if(ggml_is_quantized(type) )
            nQ++;
        if(type == GGML_TYPE_F16 )
            nF16++;
    }
    for(i=0;i<cgraph->n_nodes;i++)  {
        auto type = cgraph->nodes[i]->type;
        if(ggml_is_quantized(type) )
            nQ++;
        if(type == GGML_TYPE_F16 )
            nF16++;
    }
    
    _INFO("%s cgraph(%d,%d) nQ=%d nF16=%d",__func__,cgraph->n_leafs,cgraph->n_nodes,nQ,nF16);
}*/

#include <signal.h>
// ctrl+C handling
inline void Handle_CtrlC(int flag=0x0)  {
    assert(0);
/*
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_params->interactive) {
            is_interacting = true;
        } else {
            console::cleanup();
            printf("\n");
            llama_print_timings(*g_ctx);
            write_logfile(*g_ctx, *g_params, *g_model, *g_input_tokens, g_output_ss->str(), *g_output_tokens);
            _exit(130);
        }
    }
}
#endif
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif*/
}

inline void add_to_f32(struct ggml_context * ctx,struct ggml_tensor* a,struct ggml_tensor* b) {
    // if (ggml_is_quantized(a->type) || a->type == GGML_TYPE_F16) {
    //     return ggml_add_cast(ctx, a, b, GGML_TYPE_F32);
    // } else if (a->type == GGML_TYPE_F32) {
    //     return ggml_add(ctx, a, b);
    // } else {
    //     fprintf(stderr,"%s: ggml_add_cast on tensors with type '%s' is not yet supported.\n",__func__, ggml_type_name(a->type));
    //     exit(1);
    // }
};

void _WANDB_log(double a);

int Gensor_loab(struct ggml_context * ctx0,hGensor w,int nHeavy,hGensor ga,hGensor gb,int flag=0x0);
int Gensor_SVD(struct ggml_context * ctx0,hGensor w,int nHeavy,hGensor U,hGensor D,hGensor V,int flag=0x0);

float LOSS_cross_entropy_1(int n,const float*preP,int target,int&isMatch,int flag=0x0);
hGensor  ggml_cross_entropy_loss_1(struct ggml_context * ctx,hGensor  a, hGensor  b);
int CHECK_SAME_TENSORS(const string& desc,const std::vector<hGensor>&arrA,const std::vector<hGensor>&arrB,int flag=0x0);
size_t F_SIZE(const std::string&fpath,FILE *fp0=NULL,int flag=0x0); 
struct ggml_context *InitCTX(size_t msize,int flag=0x0);
/*
inline hGensor To4D(struct ggml_context * ctx_build,hGensor cur,int64_t n1,int64_t n2,int64_t n3,int64_t n4){
    cur = ggml_reshape_4d(ctx_build, cur, n1, n2,n3,n4);
    return cur;
}*/

#ifdef __USE_GGML__
    ggml_cgraph * GG_dup_graph(ggml_context * ctx, ggml_cgraph *src);
    hGensor GG_SCAL(struct ggml_context * ctx,struct ggml_tensor  * a,float s,int flag=0x0);
    hGensor GG_map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor);
    hGensor GradOf(struct ggml_cgraph *cgraph,hGensor node,int flag=0);
    #ifdef __cplusplus
    extern "C" {
    #endif
        //
        // For TGraph
        //
            size_t ggml_hash_size(size_t min_sz);
            size_t ggml_graph_nbytes(size_t size, bool grads);
            struct ggml_object * ggml_new_object(struct ggml_context * ctx, enum ggml_object_type type, size_t size);        
            
            void clear_numa_thread_affinity(void);
            int ggml_get_n_tasks(hGensor  node, int n_threads);
    #ifdef __cplusplus
    }
    #endif
#endif
void assert_shape_1d(hGensor  tensor, int64_t ne0);
void assert_shape_2d(hGensor  tensor, int64_t ne0, int64_t ne1);
void assert_shape_3d(hGensor  tensor, int64_t ne0, int64_t ne1, int64_t ne2);
void assert_shape_4d(hGensor  tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

inline bool CHECK_SHAPE(const SHAPE&shape){
    bool isValid = shape.size()>0;
    for(auto s : shape){
        if(s<0) {
            isValid = false;        break;
        }
        if(s>1024*1024)        {
            isValid = false;        break;
        }
    }
    assert(isValid);
    return isValid;
}

