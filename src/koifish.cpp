
#include <vector>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <string>
#include "./Manifold/gLLM.hpp"
#include "./Manifold/VAE.hpp"
#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int main(int argc, char ** argv) {
    print_build_info();
    
    struct CLI_params params;
    if (!params.parse(argc, argv)) {
        return -1;
    }   
    assert(params.nabla > 0);     

    if (params.common.seed == LLAMA_DEFAULT_SEED) {
        params.common.seed = time(NULL);
    }
    printf("\n%s: seed: %u\n", __func__, params.common.seed);
    srand(params.common.seed);

    if(params.test=="GPT_work")
        return GPT_work(params);    
    if(params.test=="fish_1")
        return fish_1(params);   
    if(params.test=="GGUF_list")
        return GGUF_list(params);     
    if(params.test=="bubble")
        return Fish_bubble(params);   
    if(params.test=="tutor")
        return Tutor(params);   
    params.OnArch();    
    hFISH fish = nullptr;
    if(params.n_swarm>1)   {
        fish = Fish::MakeSwarm("Fish_",params,0x0);    
    }else {
        vector<hWIKI> wikis = WIKI::MakeInstance("",params,0x0);      
        if(wikis.size()==0){
            // _INFO("====== NO WIKI !!! ======\n");       return;
        }else{
            wikis[0]->CopyParams(params);      
        }   
        fish = Fish::MakeInstance("Fish_",params,wikis,Fish::ROLE_TYPE::COMMON,0x0);    
    } 
    if(fish && fish->isTrain())
        fish->Train( );

    return 0x0;    
}

extern "C" bool alloc_tensor_range(struct ggml_context * ctx,struct ggml_tensor * first, struct ggml_tensor * last,
        ggml_backend_buffer_type_t buft, size_t size,ggml_backend_buffer_t ** buffers, size_t * n_buffers);
size_t Fish::InitBackEnd(struct ggml_context *ctx,int flag){
    assert(back_data==nullptr);
    auto buft = ggml_backend_cpu_buffer_type();
    /*
        static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
         {
            ggml_backend_cpu_buffer_type_get_name,
             ggml_backend_cpu_buffer_type_alloc_buffer,
             ggml_backend_cpu_buffer_type_get_alignment,
            NULL, // defaults to SIZE_MAX
             NULL, // defaults to ggml_nbytes
            ggml_backend_cpu_buffer_type_is_host,
        },
         NULL,
    };
    */
    // back_data = ggml_backend_alloc_ctx_tensors_from_buft(ctx,type );
    assert(ggml_get_no_alloc(ctx) == true);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t max_size = ggml_backend_buft_get_max_size(buft);        //  SIZE_MAX

    ggml_backend_buffer_t * buffers = NULL;
    size_t n_buffers = 0;

    size_t cur_buf_size = 0;
    struct ggml_tensor * first = ggml_get_first_tensor(ctx);
    for (struct ggml_tensor * t = first; t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        size_t this_size = 0;
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }
        if(this_size>3096*1024*1024)        // huge tensor of "gate_exLogits_1" 524M        [ 32000  512  32  1 f32]
            assert(0);
        if (this_size > max_size) {
            fprintf(stderr, "%s: tensor %s is too large to fit in a %s buffer (tensor size: %zu, max buffer size: %zu)\n",__func__, t->name,
                    ggml_backend_buft_name(buft),this_size, max_size);
            for (size_t i = 0; i < n_buffers; i++) {
                ggml_backend_buffer_free(buffers[i]);
            }
            free(buffers);
            return NULL;
        }
        
        if ((cur_buf_size + this_size) > max_size) {
            // allocate tensors in the current buffer
            if (!alloc_tensor_range(ctx, first, t, buft, cur_buf_size, &buffers, &n_buffers)) {
                return NULL;
            }
            first = t;
            cur_buf_size = this_size;
        } else {
            cur_buf_size += this_size;
        }
    }

    // allocate remaining tensors
    if (cur_buf_size > 0) {
        if (!alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, &buffers, &n_buffers)) {
            return NULL;
        }
    }

    if (n_buffers == 0) {
#ifndef NDEBUG
        fprintf(stderr, "%s: all tensors in the context are already allocated\n", __func__);
#endif
        return NULL;
    }

    ggml_backend_buffer_t buffer;
    if (n_buffers == 1) {
        buffer = buffers[0];
    } else {
        assert(0);
        // buffer = ggml_backend_multi_buffer_alloc_buffer(buffers, n_buffers);
    }
    free(buffers);
    // return buffer;
    back_data = buffer;
    assert(back_data!=nullptr);     
    size_t sz=ggml_backend_buffer_get_size(back_data);          //buffer->size;
    double sG = sz*1.0/1.0e9;
    return sz;
}   