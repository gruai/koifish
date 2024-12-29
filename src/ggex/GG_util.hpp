/**
 *  Copyright 2023-2024 by Grusoft 
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

#include <cassert>
#include <cstdarg>
#include <complex>
#include <memory>
#include <vector>
#include <map>
#include <typeinfo>
#include <float.h>
#include <stdio.h>
#include <threads.h>
#include <atomic>
#include <inttypes.h> 
#include <fstream> 
#include <regex> 
using namespace std;
#include "ggml.h"
// #include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
// #include "common-ggml.h"
#include "../CLI_params.hpp"
#include "../g_stddef.hpp"
#include "train.h"          //struct train_params_common common;
#include "common.h" 

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

static char buffer[GGML_MAX_NAME];
static const char * _NAM_( const char *format,... )	{
    va_list args;
    va_start( args, format );
    vsnprintf( buffer,GGML_MAX_NAME,format,args );
    va_end(args);
    return buffer;
}
//return tensor name
static const char * TN( const char *format,... )	{
    va_list args;
    va_start( args, format );
    vsnprintf( buffer,GGML_MAX_NAME,format,args );
    va_end(args);
    assert(strlen(buffer)+strlen(".weight")<=GGML_MAX_NAME);
    return strcat(buffer,".weight");
}

static const char * TNs( const char *format,const char *suffix,... )	{
    va_list args;
    va_start( args, suffix );       //  va_start( args, format );
    vsnprintf( buffer,GGML_MAX_NAME,format,args );
    va_end(args);
    const char*w = ".weight";
    if(strlen(buffer) > 7 && strcmp(buffer+strlen(buffer)-7, ".weight")==0){
        w = "";
    }
    assert(strlen(buffer)+strlen(w)+strlen(suffix)<=GGML_MAX_NAME);
    strcat(buffer,w);
    strcat(buffer,suffix);        
    return buffer;
}
//set name of a tensor if its name is "\0" & its grad
int gTN(struct ggml_tensor *,const char *format,...);
//clear then set name of a tensor & its grad
int gTN0(struct ggml_tensor *cur,const char *format,... );

inline void GG_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

inline void GG_log_internal_v(ggml_log_level level, const char * format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[256];
    int len = vsnprintf(buffer, 256, format, args);
    if (len < 256) {
        GG_log_callback_default(level, buffer, nullptr);
    } else {
        char* buffer2 = new char[len+1];
        vsnprintf(buffer2, len+1, format, args_copy);
        buffer2[len] = 0;
        GG_log_callback_default(level, buffer2, nullptr);
        delete[] buffer2;
    }
    va_end(args_copy);
}

inline void GG_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    GG_log_internal_v(level, format, args);
    va_end(args);
}

#define _INFO(...)  GG_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define _WARN(...)  GG_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define _ERROR(...) GG_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define _INFO_IF(...)   {   if(DUMP())  GG_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__);}
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

typedef struct ggml_tensor* hGensor;
struct GENSOR_INFO{
    string sX;
    int level=-1,ID=-1,dad,c_id;
    bool isAdam = true;
    // for Optimizer
    hGensor gm=nullptr;  // first moment
    hGensor gv=nullptr;  // second moment
    hGensor gpf=nullptr; // past function values
    
    GENSOR_INFO(){;}
    GENSOR_INFO(int id_,int l,int d,int c) : ID(id_),level(l),dad(d),c_id(c)  {
        string suffix,prefix;
        sX = __repr__(suffix,prefix);
    }
    string __repr__(string& suffix,string& prefix,int flag=0x0) const {  
        char buf[512]="\0";
        if(dad==-1){
            sprintf(buf+strlen(buf),"ROOT"); 
        }else
            sprintf(buf+strlen(buf),"[%d %d.%d l=%d]",ID,dad,c_id,level); 
        return buf;
    }

    static bool comp(GENSOR_INFO& a, GENSOR_INFO& b) {
        return a.ID < b.ID;
    }
};
struct GENSORS{
    // name and gg_tensor
    std::map<std::string, hGensor> nag;
    std::map<hGensor, GENSOR_INFO> infos;
    virtual bool has(hGensor gensor){
        assert(nag.size()==infos.size());
        bool b1 = nag.find(gensor->name) != nag.end(),b2=infos.find(gensor) != infos.end();
        assert(b1==b2);
        return b2;
    }
    //  Deprecated
    void Insert(hGensor gensor){
        // const char* key = ggml_get_name(gensor);
        // assert(strlen(key)>0);
        // assert(nag.find(key) == nag.end());
        // nag[key] = gensor;
    }   
    void Insert(std::vector<hGensor> gs){
        for(auto gensor : gs){
            Insert(gensor);
        }
    }   
    void Insert(hGensor gensor,const GENSOR_INFO&gi,int flag=0x0){
        const char* key = ggml_get_name(gensor);
        assert(strlen(key)>0);
        assert(nag.find(key) == nag.end());
        nag[key] = gensor;

        assert(infos.find(gensor) == infos.end());
        infos[gensor] = gi;    
        infos[gensor].sX = gensor->name;
    }

    void Insert(const std::map<std::string, struct ggml_tensor *>& src){
        nag.insert(src.begin(), src.end());
    }
    size_t size()   {   return nag.size();  }
    virtual hGensor Get(const char *name, int flag = 0x0)    {        
        if(flag==0x100){    //  .weight=>.w
            for(auto ng:nag){
                if(strstr(name,ng.first.c_str())!= NULL){
                    return ng.second;
                }
            }
            return nullptr;
        }else{
            if(nag.find(name) == nag.end()){
                assert(0);  return nullptr;
            }
            return nag[name];
        }
        
    } 
    virtual void Clear() {   
        nag.clear();    
        infos.clear();
    }

    virtual void TopoOrder()    {
        // sort(gimap.begin(), gimap.end(), comp);
    }
};

enum GD_METHOD {
    ADAMw=0x0,          
    SGD,               
    SGD_v,
    SGD_blk_v,
    SGD_HYBRID,   
};
 

void _T_repr_(hGensor t,const char*tab,char *buf,int typ=0x0);
void _T_repr_(hGensor t,const char*tab,char *buf,const GENSOR_INFO&info);


extern "C"
inline void gg_print_tensor_(const char* title, struct ggml_tensor * t, int n = 10) {
    if(strlen(title)>0) printf("%s\n", title);
    int nElems = ggml_nelements(t),nz=0;
    float * data = (float *)t->data,a1=-FLT_MAX,a0=FLT_MAX;
    if(t->type!=GGML_TYPE_F32){
        data = new float[nElems];
        if(t->type==GGML_TYPE_F16){
            ggml_fp16_t *src_ = (ggml_fp16_t *)(t->data);
            for (int i = 0; i < nElems; i++) {
                data[i] = src_[i];
            }
        }else{  //need dequant
            data = nullptr;     n = 0;
        }
    }    
    double sum = 0.0;
    if(data!=nullptr){
        for (int i = 0; i < nElems; i++) {
            sum += data[i];
            if(data[i]==0)      nz++;
            a1 = std::max(a1,data[i]);      a0 = std::min(a0,data[i]);
        }        
    }
    // printf("sum:  %f\n\n", sum);
    if(nElems==1){
        printf("T%d:%s: %s\t data=%f \n",-1/*t->id*/,t->name, ggml_type_name(t->type),a0);
    }else
        printf("T%d:%s: %.4g(M)\t[% " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " %s] sum=%g data=[%f : %f] rZ=%.3g%%\n", 
            -1/*t->id*/,t->name,nElems/1.0e6,t->ne[0], t->ne[1], t->ne[2], t->ne[3], ggml_type_name(t->type),sum,a0,a1,nz*100.0/nElems);
    if(n>0){
        printf("\t{");
        for (int i = 0; i < std::min((int) (t->ne[0]*t->ne[1]), n); i++) {
            printf("%.5f ", data[i]);
            if (i != 0 && i % t->ne[0] == 0) {
                // printf("\n");
            }
        }
        printf("...");
        for (int i = 0; i < std::min((int) (t->ne[0]*t->ne[1]), n); i++) {
            printf("%.5f ", data[ggml_nelements(t) - n + i]);
            if ((ggml_nelements(t) - n + i) % t->ne[0] == 0) {
                // printf("\n");
            }
        }
        printf("}\n");
    }

    if(t->type!=GGML_TYPE_F32){
        delete[] data;
    } 
}

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

inline bool gg_load_weights(std::ifstream&fin, const ggml_ftype ftype,std::map<std::string, struct ggml_tensor *>& tensors,
    const std::vector<std::string>& to_quant,const std::vector<std::string>&to_skip,int flag=0x0)     {
    int n_tensors = 0;
    size_t total_size = 0;
    std::vector<uint8_t>     data_u8;
    std::vector<ggml_fp16_t> data_f16;
    std::vector<float>       data_f32;
    std::vector<int64_t> hist_all(1 << 4, 0);
    ggml_type qtype = GGML_TYPE_F32;
    //const ggml_ftype ftype = ggml_parse_ftype("2");     //2:faster but less precise     3:slower but more precise
    switch (ftype) {
        case GGML_FTYPE_MOSTLY_Q4_0: qtype = GGML_TYPE_Q4_0; break;
        case GGML_FTYPE_MOSTLY_Q4_1: qtype = GGML_TYPE_Q4_1; break;
        case GGML_FTYPE_MOSTLY_Q5_0: qtype = GGML_TYPE_Q5_0; break;
        case GGML_FTYPE_MOSTLY_Q5_1: qtype = GGML_TYPE_Q5_1; break;
        case GGML_FTYPE_MOSTLY_Q8_0: qtype = GGML_TYPE_Q8_0; break;
        case GGML_FTYPE_MOSTLY_Q2_K: qtype = GGML_TYPE_Q2_K; break;
        case GGML_FTYPE_MOSTLY_Q3_K: qtype = GGML_TYPE_Q3_K; break;
        case GGML_FTYPE_MOSTLY_Q4_K: qtype = GGML_TYPE_Q4_K; break;
        case GGML_FTYPE_MOSTLY_Q5_K: qtype = GGML_TYPE_Q5_K; break;
        case GGML_FTYPE_MOSTLY_Q6_K: qtype = GGML_TYPE_Q6_K; break;
        case GGML_FTYPE_UNKNOWN:
        case GGML_FTYPE_ALL_F32:
        case GGML_FTYPE_MOSTLY_F16:
        case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16:
        case GGML_FTYPE_MOSTLY_IQ2_XXS:
        case GGML_FTYPE_MOSTLY_IQ2_XS:
        case GGML_FTYPE_MOSTLY_IQ3_XXS:
        case GGML_FTYPE_MOSTLY_IQ1_S:
                {
                    fprintf(stderr, "%s: invalid model type %d\n", __func__, ftype);
                    return false;
                }
    };


    fprintf(stderr, "+ %s: ......\n", __func__);
    while (!fin.eof()) {
        int32_t n_dims,length,ttype;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fin.read(reinterpret_cast<char *>(&length), sizeof(length));
        fin.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));
        if (fin.eof()) {
            return false;
        }
        ggml_type gType = (ggml_type) ttype;
        int nelements = 1;
        int ne[4] = { 1, 1, 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            int32_t ne_cur;
            fin.read(reinterpret_cast<char *>(&ne_cur), sizeof(ne_cur));
            ne[i] = ne_cur;
            nelements *= ne[i];
        }

        std::string name(length, 0);
        fin.read(&name[0], length);
        printf("%-64s - [%5d, %5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type) ttype));        

        if (tensors.find(name.data()) == tensors.end()) {
            fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
            return false;
        }
        if(name=="image_encoder.blocks.0.attn.rel_pos_w"){  //only for debug
            auto tensor = tensors[name.data()];
            int hahaha=1;
        }

        auto tensor = tensors[name.data()];
        //printf("ne0 = %jd, ne1 = %jd, ne2 = %jd, ne3 = %jd\n", ne[0], ne[1], ne[2], ne[3]);
        if (ggml_nelements(tensor) != nelements) {
            fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %d, expected %d\n",
                    __func__, name.data(), (int) nelements, (int) ggml_nelements(tensor));
            return false;
        }
        if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2] || tensor->ne[3] != ne[3]) {
            fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d, %d], expected [%d, %d, %d, %d]\n",
                    __func__, name.data(),
                    (int) ne[0], (int) ne[1], (int) ne[2], (int) ne[3],
                    (int) tensor->ne[0], (int) tensor->ne[1], (int) tensor->ne[2], (int) tensor->ne[3]);
            return false;
        }

        bool quantize = false;
        for (const auto & s : to_quant) {
            if(name.find(s)!=std::string::npos )    {   //if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }
        for (const auto & s : to_skip) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = false;
                break;
            }
        }        
        quantize &= (n_dims == 2);// quantize only 2D tensors
        if (quantize && false) {
            if (ttype != GGML_TYPE_F32 && ttype != GGML_TYPE_F16) {
                fprintf(stderr, "%s: unsupported ttype %d (%s) for integer quantization\n", __func__, ttype, ggml_type_name((ggml_type) ttype));
                return false;
            }
            if (ttype == GGML_TYPE_F16) {
                data_f16.resize(nelements);
                fin.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                data_f32.resize(nelements);
                for (int i = 0; i < nelements; ++i) {
                    data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                }
            } else {
                data_f32.resize(nelements);
                fin.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
            }
            ttype = qtype;
            std::vector<float> work;
            work.resize(nelements); // for quantization

            size_t cur_size = 0;
            std::vector<int64_t> hist_cur(1 << 4, 0);
            assert(ttype==GGML_TYPE_Q4_0 || ttype==GGML_TYPE_Q4_1 || ttype==GGML_TYPE_Q5_0 || ttype==GGML_TYPE_Q5_1
            || ttype==GGML_TYPE_Q8_0 || ttype==GGML_TYPE_Q2_K || ttype==GGML_TYPE_Q3_K || ttype==GGML_TYPE_Q4_K
            || ttype==GGML_TYPE_Q5_K || ttype==GGML_TYPE_Q6_K  );
// shared_ptr<OBS_WX<QBL_FLOAT>> quant = Quant_Factory::CreateQuanter<QBL_FLOAT>(Quant_Factory::MODEL::OPT_BRAIN,H,W,Q,wbits,0x0);
            // cur_size = ggml_quantize_chunk((ggml_type) ttype, data_f32.data(), work.data(), 0, nelements/ne[0], ne[0], hist_cur.data(), nullptr);
            // memcpy(reinterpret_cast<char *>(tensor->data),work.data(),cur_size);
            // tensor->type = (ggml_type)ttype;
            //dequantize_row_q4_0();            
                
            printf("size = %8.2f MB -> %8.2f MB | hist: ", nelements * sizeof(float)/1024.0/1024.0, cur_size/1024.0/1024.0);
            for (int i = 0; i < (int) hist_cur.size(); ++i) {
                hist_all[i] += hist_cur[i];
            }

            for (int i = 0; i < (int) hist_cur.size(); ++i) {
                printf("%5.3f ", hist_cur[i] / (float)nelements);
            }
            printf("\n");
            //++n_tensors;   continue;
        } 

        size_t bpe = 0;

        switch (ttype) {
            case 0: bpe = ggml_type_size(GGML_TYPE_F32);  break;
            case 1: bpe = ggml_type_size(GGML_TYPE_F16);  break;
            case 2: bpe = ggml_type_size(GGML_TYPE_Q4_0); assert(ne[0] % 64 == 0); break;
            case 3: bpe = ggml_type_size(GGML_TYPE_Q4_1); assert(ne[0] % 64 == 0); break;
            default:
                    {
                        fprintf(stderr, "%s: unknown ttype %d in model file\n", __func__, ttype);
                        return false;
                    }
        };

        if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
            fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                    __func__, name.data(), ggml_nbytes(tensor), (size_t) nelements*bpe);
            return false;
        }

        fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));    
        total_size += ggml_nbytes(tensor);
        if (++n_tensors % 8 == 0) {
            fprintf(stderr, ".");
            fflush(stdout);
        }
        // gg_print_tensor_("",tensor);"
        printf("\n");
        if(name.length()<GGML_MAX_NAME)
            ggml_set_name(tensor,name.c_str());         
        if (n_tensors == int(tensors.size())) { //So strange!!! more datas left
            break;
        }
    }

    if (n_tensors != int(tensors.size())) {
        fprintf(stderr, "%s: model file has %d tensors, but %d tensors were expected\n", __func__, n_tensors, (int) tensors.size());
        return false;
    }

    fprintf(stderr, " done\n");
    fprintf(stderr, "%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);

    return true;
}

inline void Gensor2float_(const hGensor w,float *A,int flag=0x0)   {
    size_t ne00 = ggml_nelements(w),nbyte = ggml_nbytes(w); 
    void *data_0 = w->data;
    enum ggml_type tp0 = w->type;
    void  *src0_row = (void *) ((char *) w->data );
    assert(ggml_is_quantized(w->type));
    switch(w->type){
        // case GGML_TYPE_F16:
        //     ggml_fp16_to_fp32_row((ggml_fp16_t*)w->data,A,nIn*nOut);
        //     break;
        case GGML_TYPE_F32:
            break;
        case GGML_TYPE_Q2_K:
            dequantize_row_q2_K((const block_q2_K*)src0_row, A, ne00);//-0.00318908691
            break;
        case GGML_TYPE_Q3_K:
            dequantize_row_q3_K((const block_q3_K*)src0_row, A, ne00);  
            break;
        case GGML_TYPE_Q4_K:
            dequantize_row_q4_K((const block_q4_K*)src0_row, A, ne00);  
            break;
        case GGML_TYPE_Q6_K:
            dequantize_row_q6_K((const block_q6_K*)src0_row, A, ne00);  
            break;
        case GGML_TYPE_Q8_0:
            dequantize_row_q8_0((const block_q8_0*)src0_row, A, ne00);  
            break;
        default:
            assert(0);
            // ggml_tensor_dequant(ctx0,w,GGML_TYPE_F32);       //memory leak@"float * wdata = malloc(sizeof(float)*ne00)" !!!
            // memcpy(A,w->data,sizeof(float)*ne00);
            // w->data = data_0;       w->type = tp0;
            break;
    }
}
inline float *Gensor2float(struct ggml_context * ctx0,const hGensor w,int flag=0x0)   {
    size_t ne00 = ggml_nelements(w),nbyte = ggml_nbytes(w); 
    void *data_0 = w->data;
    float *A=new float[ne00];
    Gensor2float_(w,A,flag);
    /*enum ggml_type tp0 = w->type;
    void  *src0_row = (void *) ((char *) w->data );
    assert(ggml_is_quantized(w->type));
    switch(w->type){
        // case GGML_TYPE_F16:
        //     ggml_fp16_to_fp32_row((ggml_fp16_t*)w->data,A,nIn*nOut);
        //     break;
        case GGML_TYPE_F32:
            break;
        case GGML_TYPE_Q2_K:
            dequantize_row_q2_K((const block_q2_K*)src0_row, A, ne00);//-0.00318908691
            break;
        case GGML_TYPE_Q3_K:
            dequantize_row_q3_K((const block_q3_K*)src0_row, A, ne00);  
            break;
        case GGML_TYPE_Q4_K:
            dequantize_row_q4_K((const block_q4_K*)src0_row, A, ne00);  
            break;
        case GGML_TYPE_Q6_K:
            dequantize_row_q6_K((const block_q6_K*)src0_row, A, ne00);  
            break;
        default:
            assert(0);
            // ggml_tensor_dequant(ctx0,w,GGML_TYPE_F32);       //memory leak@"float * wdata = malloc(sizeof(float)*ne00)" !!!
            // memcpy(A,w->data,sizeof(float)*ne00);
            // w->data = data_0;       w->type = tp0;
            break;
    }*/
    
    return A;
}


//must called after ggml_build_forward/backwrad_expand
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
}

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

inline struct ggml_tensor * add_to_f32(struct ggml_context * ctx,hGensor  a,hGensor  b) {
    if (ggml_is_quantized(a->type) || a->type == GGML_TYPE_F16) {
        return ggml_add_cast(ctx, a, b, GGML_TYPE_F32);
    } else if (a->type == GGML_TYPE_F32) {
        return ggml_add(ctx, a, b);
    } else {
        fprintf(stderr,"%s: ggml_add_cast on tensors with type '%s' is not yet supported.\n",__func__, ggml_type_name(a->type));
        exit(1);
    }
};

inline hGensor gg_axpy_f32(struct ggml_context * ctx,hGensor a,float alpha,hGensor b,float beta) {
    hGensor result = nullptr;
    ggml_type type = GGML_TYPE_F32;
    if (ggml_is_quantized(a->type) || a->type == GGML_TYPE_F16) {
        bool is_node = false;       //ggml_add_cast_impl
        // if (a->grad || b->grad) {
        //     GGML_ASSERT(ggml_are_same_shape(a, b));
        //     is_node = true;
        // }

        struct ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, a->ne);
        float abc[3] = {1,alpha,beta};
        memcpy(result->op_params, abc, sizeof(abc));        //ggml_set_op_params(result, abc, sizeof(abc));
        result->op   = GGML_OP_ADD;
        // result->grad = is_node ? ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, a->ne) : NULL;
        result->src[0] = a;   
        result->src[1] = b;

        return result;
    } else if (a->type == GGML_TYPE_F32) {
        result = ggml_add(ctx, ggml_scale(ctx,a,alpha), ggml_scale(ctx,b,beta));
    } else {
        fprintf(stderr,"%s: gg_axpy on tensors with type '%s' is not yet supported.\n",__func__, ggml_type_name(a->type));
        exit(1);
    }
    return result;
};

void _WANDB_log(double a);

int Gensor_loab(struct ggml_context * ctx0,hGensor w,int nHeavy,hGensor ga,hGensor gb,int flag=0x0);
int Gensor_SVD(struct ggml_context * ctx0,hGensor w,int nHeavy,hGensor U,hGensor D,hGensor V,int flag=0x0);

inline void _TIME(double fmillis) {
    if (fmillis < 1000.0f) {
        _INFO("%.1fms", (float) fmillis);
        return;
    }
    const int64_t one_sec = 1000, one_min = one_sec*60, one_hour = one_min*60, one_day = one_hour*24;

    int64_t millis  = (int64_t) fmillis;
    int64_t days    = millis/one_day;
    int64_t hours   = (millis - days*one_day)/one_hour;
    int64_t minutes = (millis - days*one_day - hours*one_hour)/one_min;
    int64_t seconds = (millis - days*one_day - hours*one_hour - minutes*one_min)/one_sec;

    // to print int64_t either cast to (long long int) or use macro PRId64 from <inttypes.h>
    if (days > 0) {
        _INFO("%lldd ", (long long int) days);
    }
    if(hours==0 && minutes==0){
        _INFO("%02ld", seconds);
    }else
        _INFO("%02lld:%02lld:%02lld", (long long int) hours, (long long int) minutes, (long long int) seconds);
}
float LOSS_cross_entropy_1(int n,const float*preP,int target,int&isMatch,int flag=0x0);
struct ggml_tensor * ggml_cross_entropy_loss_1(struct ggml_context * ctx,struct ggml_tensor * a, struct ggml_tensor * b);
int CHECK_SAME_TENSORS(const string& desc,const std::vector<hGensor>&arrA,const std::vector<hGensor>&arrB,int flag=0x0);
size_t F_SIZE(const std::string&fpath,FILE *fp0=NULL,int flag=0x0); 
struct ggml_context *InitCTX(size_t msize,int flag=0x0);

inline hGensor To4D(struct ggml_context * ctx_build,hGensor cur,int64_t n1,int64_t n2,int64_t n3,int64_t n4){
    cur = ggml_reshape_4d(ctx_build, cur, n1, n2,n3,n4);
    return cur;
}
inline hGensor Permute(struct ggml_context * ctx_,hGensor cur,int64_t n1,int64_t n2,int64_t n3,int64_t n4,bool isCont=true)    {
    hGensor q = ggml_permute(ctx_, cur, n1,n2,n3,n4);   
    gTN0(q,"%s.#",cur->name);     
    if(isCont)    {
        q = ggml_cont(ctx_,q);        
        gTN(q,"%s.#c",cur->name);           
    }
    return q;
}

ggml_cgraph * GG_dup_graph(ggml_context * ctx, ggml_cgraph *src);
hGensor GG_SCAL(struct ggml_context * ctx,struct ggml_tensor  * a,float s,int flag=0x0);
hGensor GG_map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor);
hGensor GradOf(struct ggml_cgraph *cgraph,hGensor node,int flag=0);

typedef struct ggml_tensor gensor;
typedef struct ggml_tensor *hGensor;
typedef std::map<std::string, struct ggml_tensor *> TENSORs;
typedef std::vector<int> SHAPE;
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

#ifdef __cplusplus
extern "C" {
#endif
    //
    // For TGraph
    //
        size_t ggml_hash_size(size_t min_sz);
        size_t ggml_graph_nbytes(size_t size, bool grads);
        struct ggml_object * ggml_new_object(struct ggml_context * ctx, enum ggml_object_type type, size_t size);        
        void * ggml_graph_compute_thread(void * data);
        void clear_numa_thread_affinity(void);
        int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads);
#ifdef __cplusplus
}
#endif