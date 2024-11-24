/**
 *  Copyright 2023-2024 by Grusoft 
 * 
 *  \brief Segment Anything Model(https://github.com/facebookresearch/segment-anything/).hpp
 *  \author Yingshi Chen
 */
#include "GG_util.hpp"   
#include "Fish.hpp"   

/*
    sam_model_load: nEmbed      = 768
    sam_model_load: n_enc_layer      = 12
    sam_model_load: n_enc_head       = 12
    sam_model_load: n_enc_out_chans  = 256
    sam_model_load: n_pt_embd        = 4
    sam_model_load: ftype            = 1
    sam_model_load: qntvr            = 0
*/
struct sam_point {    float x;    float y;  };
struct sam_params {
    int32_t seed      = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    std::string model     = "models/sam-vit-b/ggml-model-f16.bin"; // model path
    std::string fname_inp = "img.jpg";
    std::string fname_out = "img.out";
    float   mask_threshold            = 0.f;
    float   iou_threshold             = 0.88f;
    float   stability_score_threshold = 0.95f;
    float   stability_score_offset    = 1.0f;
    float   eps                       = 1e-6f;
    float   eps_decoder_transformer   = 1e-5f;
    int engine=0x0;
    // sample image [680x453]
    sam_point pt = { 414.375f, 262.796875f, };  //   { 414.375f, 162.796875f, };

    bool isOnlySymbol = false;
};

struct NT_SAM : public NeLayer {    
    int nEmbed,nHead,head_dim;
    bool is_global_attn;
    LayerNormal norm1,norm2;
    hGensor rel_pos_w=nullptr,rel_pos_h=nullptr;
    SelfAttention attention;
    SLP in_proj;    // in_proj
    SLP proj;       // out_proj
    SLP mlp_lin1,mlp_lin2;    

    hGensor Forward(hFISH,int nEmbed,int nHead,int W,int H,hGensor cur,int flag=0x0);
    hGensor Build_(struct ggml_context * ctx0,hGensor inpL,float eps,
        int n_window_size,int n_enc_state,int n_enc_head_dim,int n_enc_head,int flag);  
    NT_SAM(hFISH ctx,const std::string&key_,const SHAPE& shape,bool is_global_,int flag=0x0);
};

struct SAM_encoder : public Fish { 
    int nEmbed,nHead,head_dim;
    int n_window_size,n_img_embd,n_patch_size,n_enc_out_chans;
    // std::vector<Transformer*> layers;
    
    SAM_encoder(struct ggml_context *ctx_, bool grad_,int nEmbed,int nLayer,int n_enc_head_dim,int _enc_out_chans,
            int n_pt_embd,int _img_embd,int _window_size,int _patch_size,int flag=0x0) 
        :   /*Fish("SAM_encoder",ctx_,grad_),*/nEmbed(nEmbed),head_dim(n_enc_head_dim),n_window_size(_window_size),n_img_embd(_img_embd),n_patch_size(_patch_size),
        n_enc_out_chans(_enc_out_chans)   {
        assert(0);  //Deprecated
        assert(nEmbed>=head_dim && head_dim>0);
        nHead = nEmbed/head_dim;
    }

    bool is_global_attn(int32_t layer) const    {
        std::vector<int> indices;
        switch (nEmbed) {
            case  768: indices = {  2,  5,  8, 11 };    break;
            case 1024: indices = {  5, 11, 17, 23 };    break;
            case 1280: indices = {  7, 15, 23, 31 };    break;
            default:
                {
                    fprintf(stderr, "%s: unsupported nEmbed = %d\n", __func__, nEmbed);
                } break;
        };

        for (const auto & idx : indices) {
            if (layer == idx) {
                return true;
            }
        }

        return false;
    }
    void AfterAddLayer(hLayer hLay,int flag=0x0)    override    {
        NT_SAM *hT = dynamic_cast<NT_SAM *>(hLay.get());
        if(hT!=nullptr){
     
        };
    } 
    void AfterAddNeuron(hNeuron hN,int flag=0x0) override{
        
    }

    hGensor pe,neck_conv_0,neck_conv_1;
    SLP proj;
    LayerNormal neck_norm_0,neck_norm_1;
    void Neck(const std::string&key_,const SHAPE& shape,int flag=0x0) override{
        //struct ggml_context * ctx = graph->ctx;
        pe = AddTensor(key_+".pos_embed",GGML_TYPE_F32,{nEmbed, n_img_embd, n_img_embd, 1},0x0);
        // pe = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, nEmbed, n_img_embd, n_img_embd, 1);
        // model.tensors["image_encoder.pos_embed"] = pe;
        proj.BuildX(key_+".patch_embed.proj",{n_patch_size, n_patch_size,           3, nEmbed},this,0x0);
            // proj_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_patch_size, n_patch_size,           3, nEmbed);
            // proj_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,            1,            1, nEmbed);
        neck_conv_0 = AddTensor(key_+".neck.0.weight",GGML_TYPE_F16,{1, 1, nEmbed,     n_enc_out_chans},0x0);
        neck_conv_1 = AddTensor(key_+".neck.2.weight",GGML_TYPE_F16,{3, 3, n_enc_out_chans, n_enc_out_chans},0x0);
            // neck_conv_0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, 1, nEmbed,     n_enc_out_chans);
            // neck_conv_1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, n_enc_out_chans, n_enc_out_chans);
                        // model.tensors["image_encoder.neck.0.weight"] = neck_conv_0;
            // model.tensors["image_encoder.neck.2.weight"] = neck_conv_1;
        neck_norm_0.BuildX(key_+".neck.1",{n_enc_out_chans},this,0x0);
            // neck_norm_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            // neck_norm_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
        neck_norm_1.BuildX(key_+".neck.3",{n_enc_out_chans},this,0x0);
            // neck_norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            // neck_norm_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
    }
};

struct sam_state {
    struct ggml_tensor * embd_img;
    struct ggml_tensor * output;
    struct ggml_tensor * low_res_masks;
    struct ggml_tensor * iou_predictions;
    //struct ggml_tensor * tmp_save = {};
    struct ggml_context * ctx;
    // buffer for `ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;
    // buffers to evaluate the model
    std::vector<uint8_t> buf_compute_img_enc;
    std::vector<uint8_t> buf_compute_fast;
    ggml_gallocr_t       allocr = {};
};

inline struct ggml_tensor* sam_layer_norm_2d(
                    struct ggml_context * ctx0,
                    struct ggml_tensor  * layer,
                    int                   n_channels,
                    struct ggml_tensor  * w,
                    struct ggml_tensor  * b,
                    float                 eps) {
    // LayerNorm2d
    // normalize along channel dimmension
    // TODO: better implementation
    layer = ggml_permute(ctx0,
                ggml_norm(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, layer, 1, 2, 0, 3)), eps),
                2, 0, 1, 3);

    layer = ggml_add(ctx0,
              ggml_mul(ctx0,
                  ggml_repeat(ctx0, ggml_reshape_3d(ctx0, w, 1, 1, n_channels), layer),
                  layer),
              ggml_repeat(ctx0, ggml_reshape_3d(ctx0, b, 1, 1, n_channels), layer));

    return layer;
}

struct SegmentAnything  : public Fish {
    // int nThread=1; 
    sam_params params;
    sam_state state;
    
    int32_t n_enc_state               = 768;
    int32_t n_enc_layer               = 12;
    int32_t n_enc_head                = 12;
    int32_t n_enc_out_chans           = 256;
    int32_t n_pt_embd                 = 4;
    int32_t n_dec_heads               = 8;
    int32_t ftype                     = 1;
    float   mask_threshold            = 0.f;
    float   iou_threshold             = 0.88f;
    float   stability_score_threshold = 0.95f;
    float   stability_score_offset    = 1.0f;
    float   eps                       = 1e-6f;
    float   eps_decoder_transformer   = 1e-5f;
    const int tfm_layers_count = 2;
    const int qkv_count = 3;
    const int norm_count = 4;
    const int n_hypernet_mpls_count = 4;

    int32_t n_img_size,n_patch_size,n_window_size,n_img_embd,n_enc_head_dim;
    shared_ptr<SAM_encoder> enc = nullptr;
    shared_ptr<Fish> nnDec = nullptr,enc_prompt=nullptr;

    std::vector<int> global_attn_indices={};
    
    SegmentAnything(sam_params& param_, int32_t img_size=1024,int32_t window_size=14,int32_t patch_size=16,int flag=0x0) 
        : n_img_size(img_size),n_patch_size(patch_size),n_window_size(window_size){
        params = param_;
        to_quant = {
            "attn.proj.weight",
            // "model/wte",
            // "model/lm_head",
            // "model/h.*/attn/c_attn/w",
            // "model/h.*/attn/c_proj/w",
            // "model/h.*/mlp/c_fc/w",
            // "model/h.*/mlp/c_proj/w",
        };
        n_img_embd = n_img_size / n_patch_size;   
        static size_t buf_size = 256u*1024*1024;
        struct ggml_init_params ggml_params = {buf_size,NULL,false, };
        state.ctx = ggml_init(ggml_params);
        state.embd_img = ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32,n_img_embd, n_img_embd, n_enc_out_chans);
        state.low_res_masks = ggml_new_tensor_3d(state.ctx, GGML_TYPE_F32,n_enc_out_chans, n_enc_out_chans, 3);
        state.iou_predictions = ggml_new_tensor_1d(state.ctx, GGML_TYPE_F32, 3);
        state.buf_compute_img_enc.resize(ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead());
        state.allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());   
    }

    size_t Size(int flag=0x0) override    {    
        ctx_size = 0;
        int n_enc_layer_local  = global_attn_indices.size();
        const int32_t n_enc_layer_global = n_enc_layer - n_enc_layer_local;
        // image encoder
        {
            ctx_size += n_enc_state*n_img_embd*n_img_embd*ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_state*3*n_patch_size*n_patch_size*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_state*ggml_type_size(GGML_TYPE_F32);

            ctx_size +=     n_enc_state*n_enc_out_chans*1*1*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_out_chans*n_enc_out_chans*3*3*ggml_type_size(GGML_TYPE_F16);

            ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);
        }
        // image encoder layers
        {
            ctx_size += n_enc_layer*n_enc_state*ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_enc_layer*n_enc_state*ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_layer_global*n_enc_head_dim*(2*n_img_embd - 1)*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_layer_global*n_enc_head_dim*(2*n_img_embd - 1)*ggml_type_size(GGML_TYPE_F16);

            ctx_size += n_enc_layer_local*n_enc_head_dim*(2*n_window_size - 1)*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_layer_local*n_enc_head_dim*(2*n_window_size - 1)*ggml_type_size(GGML_TYPE_F16);

            ctx_size += n_enc_layer*3*n_enc_state*n_enc_state*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_layer*3*n_enc_state*            ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_layer*n_enc_state*n_enc_state*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_layer*n_enc_state*            ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_layer*n_enc_state*ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_enc_layer*n_enc_state*ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_layer*4*n_enc_state*n_enc_state*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_layer*4*n_enc_state*            ggml_type_size(GGML_TYPE_F32);

            ctx_size += n_enc_layer*4*n_enc_state*n_enc_state*ggml_type_size(GGML_TYPE_F16);
            ctx_size += n_enc_layer*4*n_enc_state*            ggml_type_size(GGML_TYPE_F32);
        }

        ctx_size += (8 + 14*n_enc_layer)*ggml_tensor_overhead();

        // prompt encoder
        {
            ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F16); // 2*(n_enc_out_chans/2)

            ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);
            ctx_size += n_pt_embd*n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);
        }

        ctx_size += (2 + n_pt_embd)*ggml_tensor_overhead();

        // mask decoder
        {
            //transformer
            {
                // self_attn
                ctx_size += tfm_layers_count*qkv_count*n_enc_state*n_enc_state*ggml_type_size(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*qkv_count*n_enc_state*            ggml_type_size(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*n_enc_state*                      ggml_type_size(GGML_TYPE_F32);

                // all norms
                ctx_size += tfm_layers_count*norm_count*n_enc_state*ggml_type_size(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*norm_count*n_enc_state*ggml_type_size(GGML_TYPE_F32);

                // cross_attn_token_to_img
                ctx_size += tfm_layers_count*qkv_count*n_enc_state*(n_enc_state/2)*ggml_type_size(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*qkv_count*(n_enc_state/2)*            ggml_type_size(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*n_enc_state*                          ggml_type_size(GGML_TYPE_F32);

                // mlp
                ctx_size += tfm_layers_count*8*n_enc_out_chans*n_enc_out_chans*ggml_type_size(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*8*n_enc_out_chans*                ggml_type_size(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*n_enc_out_chans*8*n_enc_out_chans*ggml_type_size(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*n_enc_out_chans*                  ggml_type_size(GGML_TYPE_F32);

                // cross_attn_img_to_token
                ctx_size += tfm_layers_count*qkv_count*n_enc_state*(n_enc_state/2)*ggml_type_size(GGML_TYPE_F16);
                ctx_size += tfm_layers_count*qkv_count*(n_enc_state/2)*            ggml_type_size(GGML_TYPE_F32);
                ctx_size += tfm_layers_count*n_enc_state*                          ggml_type_size(GGML_TYPE_F32);

                // transformer_final_attn_token_to_img
                ctx_size += qkv_count*n_enc_state*(n_enc_state/2)*ggml_type_size(GGML_TYPE_F16);
                ctx_size += qkv_count*(n_enc_state/2)*            ggml_type_size(GGML_TYPE_F32);
                ctx_size += n_enc_state*                          ggml_type_size(GGML_TYPE_F32);

                // transformer_norm_final
                ctx_size += norm_count*n_enc_state*ggml_type_size(GGML_TYPE_F32);
                ctx_size += norm_count*n_enc_state*ggml_type_size(GGML_TYPE_F32);

                // output_upscaling
                ctx_size += n_enc_out_chans*n_img_embd*2*2*ggml_type_size(GGML_TYPE_F16);
                ctx_size += 3*n_img_embd*                  ggml_type_size(GGML_TYPE_F32);
                ctx_size += n_enc_out_chans*n_img_embd*(n_img_embd/2)*2*2*ggml_type_size(GGML_TYPE_F16);
                ctx_size += (n_img_embd/2)*                               ggml_type_size(GGML_TYPE_F32);

                ctx_size += n_hypernet_mpls_count*2*n_enc_out_chans*n_enc_out_chans*ggml_type_size(GGML_TYPE_F16);
                ctx_size += n_hypernet_mpls_count*2*n_enc_out_chans*                ggml_type_size(GGML_TYPE_F32);
                ctx_size += n_hypernet_mpls_count*n_enc_out_chans*(n_img_embd/2)*ggml_type_size(GGML_TYPE_F16);
                ctx_size += n_hypernet_mpls_count*(n_img_embd/2)*                ggml_type_size(GGML_TYPE_F32);

                // iou_prediction_head
                ctx_size += 2*n_enc_out_chans*n_enc_out_chans*ggml_type_size(GGML_TYPE_F16);
                ctx_size += 2*n_enc_out_chans*                ggml_type_size(GGML_TYPE_F32);
                ctx_size += n_pt_embd*n_enc_out_chans*ggml_type_size(GGML_TYPE_F16);
                ctx_size += n_pt_embd*                ggml_type_size(GGML_TYPE_F32);

                // iou_token_w
                ctx_size += n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);

                // mask_tokens_w
                ctx_size += n_pt_embd*n_enc_out_chans*ggml_type_size(GGML_TYPE_F32);
            }
        }
        fprintf(stderr, "%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));  //ctx size = 202.33 MB
        return ctx_size;
    }

    bool Load(const std::string&mode_path,const ggml_ftype qtype,int flag=0x0) {
        std::ifstream fin = GGML_Load(mode_path.c_str(),flag);
        if(!fin)    return false;
        fin.read((char *) &n_enc_state,     sizeof(n_enc_state));
        fin.read((char *) &n_enc_layer,     sizeof(n_enc_layer));
        fin.read((char *) &n_enc_head,      sizeof(n_enc_head));
        fin.read((char *) &n_enc_out_chans, sizeof(n_enc_out_chans));
        fin.read((char *) &n_pt_embd,       sizeof(n_pt_embd));
        fin.read((char *) &ftype,           sizeof(ftype));        
        assert(n_enc_state>0);        

        int32_t qntvr = ftype / GGML_QNT_VERSION_FACTOR;
        ftype %= GGML_QNT_VERSION_FACTOR;
        ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (ftype));
        if (wtype == GGML_TYPE_COUNT) {
            fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",__func__, mode_path.c_str(), ftype);
            return false;
        }
        n_enc_head_dim = n_enc_state / n_enc_head;
        switch (n_enc_state) {
            case  768: global_attn_indices = {  2,  5,  8, 11 };
            case 1024: global_attn_indices = {  5, 11, 17, 23 };
            case 1280: global_attn_indices = {  7, 15, 23, 31 };
            break;
            default:
                {
                    fprintf(stderr, "%s: unsupported nEmbed = %d\n", __func__, n_enc_state);
                } break;
        };
        Size();
        struct ggml_init_params params = {ctx_size,NULL,false,        };
        auto ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "%s: _init() failed\n", __func__);            return false;
        }

        enc = std::make_shared<SAM_encoder>(
            ctx,false,n_enc_state,n_enc_layer,n_enc_head_dim,n_enc_out_chans,n_pt_embd,n_img_embd,n_window_size,n_patch_size);
        childs.push_back(enc);
        enc->Neck("image_encoder",{},0x0);
        for (int i = 0; i < n_enc_layer; ++i) {    
            bool is_global_attn = enc->is_global_attn(i);            
            NT_SAM *hFormer = new NT_SAM(enc,"image_encoder.blocks." + std::to_string(i),{n_enc_state,n_enc_head_dim,n_img_embd,n_window_size},is_global_attn,0x0);  
            enc->AddLayer(std::shared_ptr<NeLayer>(hFormer));
        }
        // prompt encoder
        enc_prompt = nullptr;/*std::make_shared<Fish>("enc_prompt",ctx,0x0);*/        childs.push_back(enc_prompt);
        enc_prompt->AddTensor("prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",GGML_TYPE_F32,{n_enc_out_chans/2, 2});
        // enc.pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_enc_out_chans/2, 2);
        enc_prompt->AddTensor("prompt_encoder.not_a_point_embed.weight",GGML_TYPE_F32,{n_enc_out_chans});
        enc_prompt->AddTensor("prompt_encoder.no_mask_embed.weight",GGML_TYPE_F32,{n_enc_out_chans});
            // enc.not_a_pt_embd_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
            // enc.no_mask_embd_w  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);
        // enc.pt_embd.resize(n_pt_embd);
        for (int i = 0; i < n_pt_embd; i++) {
            enc_prompt->AddTensor("prompt_encoder.point_embeddings." + std::to_string(i) + ".weight",GGML_TYPE_F32,{n_enc_out_chans});
            // enc.pt_embd[i] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_enc_out_chans);            
        }
        // mask decoder
        nnDec = std::make_shared<Fish>("nnDec",ctx,0x0);        childs.push_back(nnDec);
        for (int i = 0; i < tfm_layers_count; ++i) {
            const auto prefix = "mask_decoder.transformer.layers." + std::to_string(i);
            nnDec->AddLayer(prefix,{
                    NP_("SelfAttention",".self_attn",{n_enc_out_chans, n_enc_out_chans, n_enc_out_chans}),
                    NP_("SelfAttention",".cross_attn_token_to_image",{n_enc_out_chans, n_enc_out_chans/2, n_enc_out_chans}),
                    NP_("LayerNormal",".norm1",{n_enc_out_chans}),
                    NP_("SelfAttention",".cross_attn_image_to_token",{n_enc_out_chans, n_enc_out_chans/2, n_enc_out_chans}),
                    NP_("LayerNormal",".norm2",{n_enc_out_chans}),
                    NP_("LayerNormal",".norm3",{n_enc_out_chans}),
                    NP_("LayerNormal",".norm4",{n_enc_out_chans}),
                    NP_("SLP",".mlp.lin1",{n_enc_out_chans, 8*n_enc_out_chans}),
                    NP_("SLP",".mlp.lin2",{8*n_enc_out_chans,n_enc_out_chans}),
                }
            );
        }
        nnDec->AddLayer("mask_decoder.transformer",{
                        NP_("SelfAttention",".final_attn_token_to_image",{n_enc_out_chans, n_enc_out_chans/2, n_enc_out_chans}),
                        NP_("LayerNormal",".norm_final_attn",{n_enc_out_chans}),
                    }
                ); 
        nnDec->AddTensor("mask_decoder.output_upscaling.0.weight",GGML_TYPE_F16, {2, 2, n_img_embd, n_enc_out_chans});
        nnDec->AddTensor("mask_decoder.output_upscaling.0.bias",GGML_TYPE_F32, {n_img_embd});
        nnDec->AddTensor("mask_decoder.output_upscaling.1.weight",GGML_TYPE_F32, {n_img_embd});
        nnDec->AddTensor("mask_decoder.output_upscaling.1.bias",GGML_TYPE_F32, {n_img_embd});
        nnDec->AddTensor("mask_decoder.output_upscaling.3.weight",GGML_TYPE_F16,  {2, 2, n_img_embd/2, n_img_embd});
        nnDec->AddTensor("mask_decoder.output_upscaling.3.bias",GGML_TYPE_F32, {n_img_embd/2});
        /*
            dec.output_upscaling_0_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 2, 2, n_img_embd, n_enc_out_chans);
            dec.output_upscaling_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd);
            dec.output_upscaling_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd);
            dec.output_upscaling_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd);
            dec.output_upscaling_3_w = ggml_new_tensor_4d(ctx, GGML_TYPE_F16,  2, 2, n_img_embd/2, n_img_embd);
            dec.output_upscaling_3_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_img_embd/2);
        */
        for (int i = 0; i < n_hypernet_mpls_count; ++i) {
            nnDec->AddLayer("mask_decoder.output_hypernetworks_mlps."+ std::to_string(i),{
                        NP_("SLP",".layers.0",{n_enc_out_chans, n_enc_out_chans}),
                        NP_("SLP",".layers.1",{n_enc_out_chans, n_enc_out_chans}),
                        NP_("SLP",".layers.2",{n_enc_out_chans, n_img_embd/2}),
                    }
                ); 
        }       
        nnDec->AddLayer("mask_decoder.iou_prediction_head",{
                        NP_("SLP",".layers.0",{n_enc_out_chans, n_enc_out_chans}),
                        NP_("SLP",".layers.1",{n_enc_out_chans, n_enc_out_chans}),
                        NP_("SLP",".layers.2",{n_enc_out_chans, n_pt_embd}),
                    }
                );     
        nnDec->AddTensor("mask_decoder.iou_token.weight",GGML_TYPE_F32,  {n_enc_out_chans, 1});
        nnDec->AddTensor("mask_decoder.mask_tokens.weight",GGML_TYPE_F32, {n_enc_out_chans, n_pt_embd}); 

        UpdateTensors();
        bool bRet = gg_load_weights(fin,qtype,gensors,to_quant,to_skip);// load weights      
        assert(bRet); 
        fin.close();
        return bRet;
    }

    void EncodeImage(int nx,int ny,const std::vector<float>& data,int flag=0x0)  {    //Build graph of encoder of image
        size_t sz = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
        state.buf_compute_img_enc.resize(sz);
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

        struct ggml_init_params ggml_params = {state.buf_compute_img_enc.size(),state.buf_compute_img_enc.data(),true,};
        struct ggml_context * ctx0   = ggml_init(ggml_params);
        // struct ggml_cgraph  * gf     = ggml_new_graph(ctx0);
        
        struct ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_img_size, n_img_size, 3, 1);
        ggml_set_name(inp, "inp");
        ggml_set_input(inp);

        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L392
        struct ggml_tensor * cur = ggml_conv_2d_sk_p0(ctx0, enc->proj.w, inp);
        cur = ggml_add_inplace(ctx0,cur,ggml_repeat(ctx0, enc->proj.b, cur));
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L394
        // keep in F32
        cur = ggml_cont(ctx0,ggml_permute(ctx0, cur, 1, 2, 0, 3));
        // convert to F16
        //cur = ggml_cpy(ctx0,
        //        ggml_permute(ctx0, cur, 1, 2, 0, 3),
        //        ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, n_enc_state, n_img_embd, n_img_embd));
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L108-L109
        cur = ggml_add_inplace(ctx0, cur, enc->pe);

        struct ggml_tensor * inpL = cur;
        assert(enc->layers.size()==n_enc_layer);
        for (int il = 0; il < n_enc_layer; ++il) {
            NT_SAM * hLay = dynamic_cast<NT_SAM *>(enc->layers[il].get());            
            inpL = hLay->Build_(ctx0,inpL,eps,n_window_size,n_enc_state,n_enc_head_dim,n_enc_head,0x0);
        }

        cur = ggml_cont(ctx0, ggml_permute(ctx0, inpL, 2, 0, 1, 3));
        cur = ggml_conv_2d_sk_p0(ctx0, enc->neck_conv_0, cur);
        cur = sam_layer_norm_2d(ctx0, cur, n_enc_out_chans, enc->neck_norm_0.w, enc->neck_norm_0.b, eps);
        cur = ggml_conv_2d_s1_ph(ctx0, enc->neck_conv_1, cur);
        cur = sam_layer_norm_2d(ctx0, cur, n_enc_out_chans, enc->neck_norm_1.w, enc->neck_norm_1.b, eps);
        cur = ggml_cpy(ctx0, cur, state.embd_img);
        out_node = cur;         
        gg_print_tensor_("embd_img_0", state.embd_img);
        in_node = state.embd_img;           //???
        Build(ctx0,state.allocr,params.isOnlySymbol);
        // hForwTG->print();
        ggml_free(ctx0);
        SetInput(nx,ny,data);
        hGensor inp1 = nullptr;     //hForwTG->get_tensor("inp");
        gg_print_tensor_("",inp1);
        hForwTG->compute_helper(params.n_threads,0x0);     
/*
T873:leaf_178: [ 64  64  256  1 f32] sum=12757.2 data=[-0.994997-0.963888] rZ=0%
        {-0.05100 -0.06349 -0.07116 -0.06840 -0.06826 -0.06972 -0.07148 -0.07088 -0.06774 -0.05427 ...
        0.01573 0.01771 0.02244 0.01670 0.01755 0.01667 0.01797 0.02044 0.02100 0.03389 }
    //openblas
     {-0.05101 -0.06349 -0.07116 -0.06837 -0.06826 -0.06974 -0.07149 -0.07087 -0.06775 -0.05427 -0.06469 -0.06627 -0.06987 -0.06701 -0.07403 -0.07661 -0.07799 -0.07747 -0.07566 -0.07484 -0.07647 -0.07696 -0.07568 -0.07464 -0.07195 -0.06934 -0.06083 -0.05930 -0.07135 -0.07290 -0.07332 -0.07152 ...
            0.01642 0.01482 0.01381 0.01452 0.01562 0.01324 0.01643 0.01636 0.01630 0.01915 0.01828 0.01951 0.02027 0.01687 0.01412 0.01521 0.01351 0.01504 0.01480 0.01463 0.01543 0.01752 
        0.01574 0.01769 0.02241 0.01670 0.01753 0.01665 0.01794 0.02047 0.02100 0.03392 }
*/
        gg_print_tensor_("embd_img", state.embd_img,32);
        hGensor inp0 = GetGensor("inp");  /**/    
    }
};
typedef shared_ptr<SegmentAnything> hSegmentAnything;