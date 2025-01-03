
#include <memory>
// struct WIKI;
// std::shared_ptr<WIKI> INST_LAMA(struct CLI_params& params,const std::string&path,int flag=0x0);

// cys  08_28
bool llama2params(struct llama_model * lmodel,struct CLI_params& cparam);
bool llama_get_params(struct llama_model * lmodel,struct llama_hparams& cparam);
bool llama_ctx_get_( struct llama_context * ctx,void **, int type);
bool llama_ctx_set_( struct llama_context * ctx,void *hData, int type);
struct llama_context * llama_ctx_reset_(struct llama_context * ctx,struct llama_model * model,struct llama_context_params   params);
void GG_set_(struct ggml_tensor * cur, const char * name,int il); 
//repeat a to same shape as b
struct ggml_tensor * _repeat(struct ggml_context * ctx,struct ggml_tensor * a,struct ggml_tensor * b);
struct ggml_tensor * mamba_build_layer(
        struct ggml_context * ctx,struct llama_context & lctx,  //  const llama_ubatch & batch,
        struct ggml_cgraph * graph,struct ggml_tensor * curT,struct ggml_tensor * inpL,int il,int n_layer, int n_tokens,int32_t kv_head=-1,int32_t n_kv=-1,int n_outputs=-1);         
struct ggml_tensor * moe_build_ffn(struct ggml_context * ctx,struct llama_context & lctx,
         struct ggml_tensor * cur,struct ggml_tensor * gate_inp,struct ggml_tensor * up_exps,struct ggml_tensor * gate_exps,struct ggml_tensor * down_exps,
                    int64_t   n_expert,int64_t   n_expert_used,bool   norm_w,bool   scale_w,float   w_scale,int   il);         
bool llama_model2vocb_( struct llama_model * model,void *hData, int type);
int _llama_build_graph(struct llama_model * model,struct ggml_cgraph **hgf,struct ggml_cgraph **hgb,int flag);
struct ggml_cgraph * _llama_raw_graph(llama_model * model,struct ggml_cgraph *gfx,const std::string&prompt,bool isOnline,int flag);
extern "C" void _pt_cys_(const char* title, struct ggml_tensor * t, int n);