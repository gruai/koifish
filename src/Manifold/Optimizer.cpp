#include "Fish.hpp"
#include "../ggex/GG_util.hpp"
#include "../LLAMA/common/common.h"

 Optimizer::Optimizer(Fish *g_,struct train_params_common& params_,int flag) : train_params(params_),gang(g_) {
    InitOpt(train_params,flag); 
    val_loader.type = DataLoader::TYPE::DT_EVAL;
    
    if(0){
        const char* train_tokens = "./src/llmc//tinyshakespeare/tiny_shakespeare_train.bin";
        const char* val_tokens = "./src/llmc/tinyshakespeare/tiny_shakespeare_val.bin";
        int B = train_params.n_batch,T =train_params.n_ctx;
        train_loader.init(train_tokens, B, T, 0, 1, 1);
        val_loader.init(val_tokens, B, T, 0, 1, 0);
        _INFO("%s train dataset num_batches: %zu\n", __func__,train_loader.tokens.size() / (B*T));
        _INFO("%s val dataset num_batches: %zu\n", __func__,val_loader.tokens.size() / (B*T));        
    }
}

hGensor Optimizer::hLoss()             {    
    assert(gang!=nullptr);
    GGML_ASSERT(ggml_is_scalar(gang->loss));  return gang->loss;          
}

hGensor Optimizer::hTargetProbs()      {   return gang->target_probs;  }
hGensor Optimizer::hPreLogits()       {   return gang->preLogits;     }

/*
// ggml_compute_forward_cross_entropy_loss_f32
    float max = -INFINITY;
    ggml_vec_max_f32(nc, &max, s0);
    ggml_float sum = ggml_vec_soft_max_f32(nc, st, s0, max);        assert(sum > 0.0);
    sum = (1.0 - eps) / sum;        //eps = 1e-9;
    // avoid log(0) by rescaling from [0..1] to [eps..1]
    ggml_vec_scale_f32(nc, st, sum);
    ggml_vec_add1_f32(nc, st, st, eps);
    ggml_vec_log_f32(nc, st, st);
    ggml_vec_mul_f32(nc, st, st, s1);

    float st_sum = 0;
    ggml_vec_sum_f32(nc, &st_sum, st);
    sums[ith] += st_sum;
*/
bool Optimizer::OnLogits(int flag)   {   
    auto pTarget = hTargetProbs();
    size_t nz = ggml_nelements(pTarget),nToken=pTarget->ne[0],n_ctx=pTarget->ne[1]; 

    auto pLogits = hPreLogits();
    assert(pLogits!=nullptr); 
    assert(pLogits->type==GGML_TYPE_F32);
    
    float *p = (float*)(pLogits->data),sum=0;       //target_probs->data
    
    for(int k=0;k<n_ctx;k++){
        sum = 0;
        for(int i=0;i<nToken;i++)    
            sum+=p[i];        
        // assert(fabs(sum-1.0)<1.0e-6);
    }

    return true;
}

bool Optimizer::GradAccumulation(float&fx,int np,struct train_opt_callback_data *callback_data,struct ggml_cplan *cplan,int flag)   {    
    fx = 0;
    float * g  = (float *)opt->adam.g->data;
    auto loss = hLoss();
    float *fLoss = (float*)(loss->data),*fLossG = (float*)(loss->grad->data);
    ggml_set_zero(opt->adam.g);   
    assert(gb!=nullptr);
    float sched = opt->params.adam.sched;        
    const int n_accum = MAX(1, opt->params.n_gradient_accumulation);
    const float accum_norm = 1.0f / (float) n_accum;
    bool cancel = false;
    struct train_state *train = callback_data->train;
    for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
        // int64_t used_samples = gang->update_batch(callback_data,train->shuffle_next_sample,callback_data->params);
        int64_t used_samples = train_loader.update_batch(train->shuffle_next_sample,gang);
        train->train_samples += used_samples;
        train->shuffle_next_sample += used_samples;

        if(one_step(train,train_loader,accum_step, &sched)){
            return false;
        }

        ggml_set_f32      (loss->grad, 1.0f);
        ggml_graph_compute(gb, cplan);
        OnLogits();
        ggml_opt_acc_grad(np, opt_ps, g, accum_norm);        
        fx += ggml_get_f32_1d(loss, 0);
    }
    fx *= accum_norm;
    return true;
}

bool isGensor(hGensor gensor,vector<string> keys,int flag=0x0){
    string name = gensor->name;
    for(auto key:keys){
        if( name.find(key) != std::string::npos ) 
            return true;
    }
    return false;
}
    /**
     * 1. LARS/LAMB  trus_ratio - ratio between the norm of the layer weights and norm of gradients update
     */
void Optimizer::AdamW(int nx,int np,CLI_params& hparams,int flag)  {
    struct ggml_opt_params params = opt->params;
    float sched = params.adam.sched;
    const float alpha = params.adam.alpha;
    const float decay = params.adam.decay * alpha;
    const float beta1 = params.adam.beta1;
    const float beta2 = params.adam.beta2;
    const float eps   = params.adam.eps;
    const float gclip = params.adam.gclip;
    const int decay_min_ndim = params.adam.decay_min_ndim;
    float gnorm = 1.0f;
    float * g  = (float *)opt->adam.g->data;  // gradients
    float * m  = (float *)opt->adam.m->data;  // first moment
    float * v  = (float *)opt->adam.v->data;  // second moment
    float * pf = params.past > 0 ? (float *)opt->adam.pf->data : NULL; // past function values
    float fx = 0;

    if (gclip > 0.0f) { //Gradient clipping maybe superior than L2 Regularization in terms of gaining results and easiness in implementation
        // gradient clipping
        ggml_float sum = 0.0;
        for (int64_t i = 0; i < nx; ++i) {
            sum += (ggml_float)(g[i]*g[i]);
        }
        ggml_float norm = sqrt(sum);                    
        if (norm > (ggml_float) gclip) {
            gnorm = (float) ((ggml_float) gclip / norm);
        } 
    }
    
    const float beta1h = alpha*sched/(1.0f - powf(beta1, opt->iter));                
    const float beta2h =        1.0f/(1.0f - powf(beta2, opt->iter));
    int64_t i = 0;
    zmuv_0 = DBL_MAX,zmuv_1 = 0.0;
    for (int p = 0; p < np; ++p) {
        const int64_t ne = ggml_nelements(opt_ps[p]);
        // assert(opt_ps[p]->flags==0x0);       ???
        bool isZMUV = isGensor(opt_ps[p],{"ffn_gate.weight","ffn_down.weight","ffn_up.weight"}) && hparams.ZMUV_ratio>0;     //ne>=hparams.n_ctx*hparams.n_embd
        const float p_decay = ((ggml_n_dims(opt_ps[p]) >= decay_min_ndim) ? decay : 0.0f) * sched;
        float wnorm=0,wmean=0,norm=0,trust_ratio;
        for (int64_t j = 0; j < ne; ++j) {
            float x  = ggml_get_f32_1d(opt_ps[p], j);       
            wnorm += x*x;           wmean +=x;
        }
        wnorm = sqrt(wnorm);        wmean /= ne;
        double sigma = wnorm*wnorm/ne - wmean*wmean;
        if(hparams.lars_ratio>0)   {   //lars/lamb                        
            for (int64_t j = i; j < i+ne; ++j) {
                norm += g[j]*g[j];
            }
            trust_ratio = wnorm/sqrt(norm+eps);
            trust_ratio = std::min(trust_ratio,hparams.lars_ratio);
            gnorm = trust_ratio;
        }
        double normX = 0.0,meanX=0;
        for (int64_t j = 0; j < ne; ++j) {
            float x  = ggml_get_f32_1d(opt_ps[p], j);
            float g_ = isZMUV ? g[i] : g[i]*gnorm;
            m[i] = m[i]*beta1 +    g_*(1.0f - beta1);   //linear interpolations of the gradients 
            v[i] = v[i]*beta2 + g_*g_*(1.0f - beta2);   //linear interpolations of their variances
            float mh = m[i]*beta1h;     //debiasing 
            float vh = v[i]*beta2h;
            vh = sqrtf(vh) + eps;
            x  = x*(1.0f - p_decay) - mh/vh;        //ormalize step(mh) by its standard deviation(vh)
            if(isZMUV){
                //x -= (sigma-1.0)*(x-wmean)/ne;
                //ggml_compute_forward_rms_norm_f32(
                x -= hparams.ZMUV_ratio*(wnorm-1.0)/(wnorm)*x;
                normX += x*x;       meanX+=x;
            }                        
            ggml_set_f32_1d(opt_ps[p], j, x);
            ++i;
        }
        if(isZMUV){
            meanX = meanX/ne;
            // normX = sqrt(normX/ne-meanX*meanX);
            normX = sqrt(normX);
            zmuv_0 = min(zmuv_0,normX),zmuv_1 = max(zmuv_1,normX);
        }
    }
}

enum ggml_opt_result Optimizer::ggml_train(struct ggml_context * ctx, hGensor loss_,hGensor target_,
    struct ggml_cgraph * gf_,struct ggml_cgraph * gb_,CLI_params& hparams)    {
    struct train_opt_callback_data *callback_data = &(train_loader.callback_data);
    // assert(loss==nullptr && target_probs==nullptr);
    // loss = loss_;       target_probs = target_;
    // preLogits = gang->preLogits;
    // callback_data->target_probs = target_probs;
    tokens_input = callback_data->tokens_input;
    
    gf=gf_;      gb=gb_;

    enum ggml_opt_result result = GGML_OPT_RESULT_DID_NOT_CONVERGE;
    struct ggml_opt_params params = opt->params;
   
    bool isAccuX = true,   cancel = false;      //Converge much faster!!! see Lay5_accux_.info
    auto loss = hLoss( );
    float *fLoss = (float*)(loss->data),*fLossG = (float*)(loss->grad->data),val_loss;
    
    // float *target = (float*)(data->target_probs->data);
    _INFO("Optimizer::%s: GradAccumu=%d rZMUV=%g rLARS=%g...... \n\n", __func__,(int)isAccuX,hparams.ZMUV_ratio,hparams.lars_ratio );
    if(0){
        result = ggml_opt_resume_g(ctx, opt, loss, gf, gb, &train_opt_callback, callback_data);
    }else{
        //struct ggml_tensor * opt_ps[GGML_MAX_PARAMS];  these will store the parameters we want to optimize
        int np = 0,i;
        int64_t nx = 0;
        for (int i = 0; i < gf->n_nodes; ++i) {
            if (gf->nodes[i]->flags & GGML_TENSOR_FLAG_PARAM) {
                GGML_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);
                GGML_ASSERT(np < GGML_MAX_PARAMS);
                opt_ps[np++] = gf->nodes[i];
                _INFO("%4d(op=%d)\t", np, gf->nodes[i]->grad->op );
                gg_print_tensor_("",gf->nodes[i],0);
                nx += ggml_nelements(gf->nodes[i]);
            }
        }            
        if ((opt->params.type != params.type) || (opt->nx != nx) || (opt->params.past != params.past)) {
            int iter = opt->iter;
            ggml_opt_init(opt->ctx, opt, params, nx);
            opt->iter = iter;
        }
        float sched = params.adam.sched;
        const int n_accum = MAX(1, params.n_gradient_accumulation);
        const float accum_norm = 1.0f / (float) n_accum;

        float * g  = (float *)opt->adam.g->data;  // gradients
        float * m  = (float *)opt->adam.m->data;  // first moment
        float * v  = (float *)opt->adam.v->data;  // second moment
        float * pf = params.past > 0 ? (float *)opt->adam.pf->data : NULL; // past function values
        float fx = 0;
        // cplan = ggml_graph_plan(gb, params.n_threads);
        // struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, cplan.work_size);
        // cplan.work_data = (uint8_t *)ctx->mem_buffer + obj->offs;
        auto *cplan = &(gang->gb_plan);
        if(isAccuX){
            if( !GradAccumulation(fx,np,callback_data,cplan,0x0))
                return GGML_OPT_RESULT_CANCEL; 
        }else   { 
            ggml_set_zero(opt->adam.g);            
            for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
                if(one_step(callback_data->train,train_loader,accum_step, &sched)){
                        return GGML_OPT_RESULT_CANCEL;
                    }
                ggml_set_f32      (loss->grad, 1.0f);
                ggml_graph_compute(gb, cplan);
                ggml_opt_acc_grad(np, opt_ps, g, accum_norm);
                fx += ggml_get_f32_1d(loss, 0);
            }
            fx *= accum_norm;
        }

        opt->adam.fx_prev = fx;
        opt->adam.fx_best = opt->adam.fx_prev;
        if (pf) {
            pf[opt->iter % params.past] = opt->adam.fx_prev;
        }

        opt->loss_before = opt->adam.fx_prev;
        opt->loss_after  = opt->adam.fx_prev;

        // initialize
        if (opt->just_initialized) {
            opt->adam.n_no_improvement = 0;
            opt->just_initialized = false;
        }

        float * fx_best = &opt->adam.fx_best;
        float * fx_prev = &opt->adam.fx_prev;
        int * n_no_improvement = &opt->adam.n_no_improvement;

        int iter0 = opt->iter;

        // run the optimizer
        for (int t = 0; t < params.adam.n_iter; ++t) {
            opt->iter = iter0 + t + 1;
            GGML_PRINT_DEBUG  ("=== iter %d ===\n", t);
            GGML_PRINT_DEBUG  ("loss      = %10.6f\n", ggml_get_f32_1d(loss, 0));
            GGML_PRINT_DEBUG_5("df/dx0 = %10.6f\n", ggml_get_f32_1d(opt_ps[0]->grad, 0));
            GGML_PRINT_DEBUG_5("df/dx1 = %10.6f\n", ggml_get_f32_1d(opt_ps[1]->grad, 0));
            for (int i = 0; i < np; ++i) {
                GGML_PRINT_DEBUG("param %d: %10.6f, g = %10.6f\n", i,ggml_get_f32_1d(opt_ps[i], 0), ggml_get_f32_1d(opt_ps[i]->grad, 0));
            }
            const int64_t t_start_wall = ggml_time_us(),t_start_cpu = ggml_cycles();
            AdamW(nx,np,hparams,0x0);   
            //gradient is useless at this stage  
            if (train_params.eval_every>0 && t % train_params.eval_every == 0) {
                val_loss = Evaluate(val_loader,t);  
            }        
            if( train_params.gpt_every>0 && t%train_params.gpt_every==0 )   {
                gang->GenSentence(1);   
            }
   
            if(isAccuX){
                if( !GradAccumulation(fx,np,callback_data,cplan,0x0))
                    return GGML_OPT_RESULT_CANCEL; 
            }else   { 
                fx = 0;
                ggml_set_zero(opt->adam.g);
                for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
                    if(one_step(callback_data->train,train_loader,accum_step, &sched)){
                        return GGML_OPT_RESULT_CANCEL;
                    }
                    // ggml_graph_reset  (gf);
                    ggml_set_f32      (loss->grad, 1.0f);
                    ggml_graph_compute(gb, cplan);
                    ggml_opt_acc_grad(np, opt_ps, g, accum_norm);
                    fx += ggml_get_f32_1d(loss, 0);
                }
                fx *= accum_norm;
            }

            opt->loss_after = fx;
            UpdateLoss(opt->iter,sched,opt->loss_after);        //scheduler->Append(loss);
            if(gang->hDistler!=nullptr) 
                gang->hDistler->UpdateSigma(opt->iter);
            // check convergence
            if (fabsf(fx - fx_prev[0])/fx < params.adam.eps_f) {
                GGML_PRINT_DEBUG("converged\n");                result = GGML_OPT_RESULT_OK;
            }
            // delta-based convergence test
            if (pf != NULL) {
                // need at least params.past iterations to start checking for convergence
                if (params.past <= iter0 + t) {
                    const float rate = (pf[(iter0 + t)%params.past] - fx)/fx;
                    if (fabsf(rate) < params.delta) {
                        result = GGML_OPT_RESULT_OK;
                    }
                }
                pf[(iter0 + t)%params.past] = fx;
            }
            // check for improvement
            if (params.max_no_improvement > 0) {
                if (fx_best[0] > fx) {
                    fx_best[0] = fx;
                    n_no_improvement[0] = 0;
                } else {
                    ++n_no_improvement[0];

                    if (n_no_improvement[0] >= params.max_no_improvement) {
                        result = GGML_OPT_RESULT_OK;
                    }
                }
            }
            
            fx_prev[0] = fx;
            const int64_t t_end_cpu = ggml_cycles();
            GGML_PRINT_DEBUG("time iter:      %5.3f s\n", ((float)(t_end_cpu - t_start_cpu))/CLOCKS_PER_SEC);
            const int64_t t_end_wall = ggml_time_us();
            GGML_PRINT_DEBUG("wall time iter: %5.3f s\n", (t_end_wall - t_start_wall)/1e6);
            
        }

        result = GGML_OPT_RESULT_DID_NOT_CONVERGE;
    }
    return result;
}

void Fish::Train(  int flag )   {
    
}

float Optimizer::Compute(std::vector<llama_token>&tokens,bool onlyForward,int flag){
    // llama_token bos = llama_token_bos(llama_get_model(lctx)),eos = llama_token_eos(llama_get_model(lctx))
    float * g  = (float *)opt->adam.g->data;
    auto loss = hLoss( );
    float *fLoss = (float*)(loss->data),*fLossG = (float*)(loss->grad->data);
    ggml_set_zero(opt->adam.g);   
    auto cplan =&(gang->gb_plan);
    llama_token token;
    ggml_set_i32_nd(tokens_input, 0, 0, 0, 0, bos);

    size_t i,len=tokens.size(),ldB=tokens_input->ne[0],nB0=tokens_input->ne[1],nSamp = len/ldB;
    // if(nSamp<nB0){   //useless should rebuild graph and plan
    //     tokens_input->ne[1] = nSamp;
    // }
    assert(len>0 && len<=tokens_input->ne[0]);
    for(i=0;i<len;i++){
        token = tokens[i];
        ggml_set_i32_nd(tokens_input, (int) (i + 1), (int) 0, 0, 0, token);
    }
    if(onlyForward)
        ggml_graph_compute(gf, cplan);   
    else
        ggml_graph_compute(gb, cplan);   /**/
    // if(nSamp<nB0){
    //     tokens_input->ne[1] = nB0;
    // }
    return *fLoss;
}

/*
    REF:    /home/cys/Github/llama.cpp-master/examples/batched/batched.cpp
*/
float Optimizer::Evaluate(DataLoader&loader,int iter,int flag){  
    if( val_loader.num_batches==0) 
        return 0; 
    _INFO("[eval] " );   
    GST_TIC(tic);     
    
    auto loss = hLoss();           
    float *fLoss = (float*)(loss->data),sum=0;
    double l2,delta_max=0,delta_=0,a;
    int i,ldT=tokens_input->ne[0],nB=0,step=max(loader.num_batches/10,1);
    
    llama_token tMost = (llama_token) (loader.n_vocab - 1);
    // loader.reset();
    // std::vector<llama_token> tokens;
    const float *wLog = nullptr;
    if(gang->exLogits!=nullptr){
        assert(gang->exLogits->type==GGML_TYPE_F32);//ggml_float
        wLog = (const float *)(gang->exLogits->data); 
    }

    for (i = 0; i < loader.num_batches; i+=step) {
        loader.update_batch(i,gang);    
        if(wLog!=nullptr)    {
            size_t nz=ggml_nelements(gang->exLogits),j;
            for (l2 = 0,j = 0; j < nz; j++    ) {
                a = wLog[j];            l2 += a*a;              
            }
            l2 = sqrt(l2)/nz;            
            delta_max = max(delta_max,l2);      delta_+=l2;        
        }
        // ggml_graph_compute(gb, cplan);     
        ggml_graph_compute(gf, &(gang->gb_plan));     
        sum += *fLoss;
        nB++;
        break;
    }
    sum /= nB;
    if(wLog==nullptr) 
        _INFO("\t Loss@Evaluation=%f T=%gs ======\n", sum,GST_TOC(tic) );
    else
        _INFO("\t Loss@Evaluation=%f delta=%g(%.5g) T=%gs ======\n", sum,delta_max,delta_/nB,GST_TOC(tic) );
    return sum;
}




std::string shuffle_samples_X(
        const std::string & rng_state,size_t* shuffled_offs,
        size_t            * shuffled_begins,
        size_t            * shuffled_sizes,
        const size_t      * begins,
        const size_t      * sizes,
        size_t              count) {
    if (count == 0) return rng_state;

    std::mt19937 rng;
    mt19937_set_state(rng, rng_state);

    // sort indices by random value for each index
    std::vector<size_t> idcs;
    {
        std::vector<unsigned> rnd;
        idcs.resize(count);
        rnd.resize(count);
        for (unsigned i=0; i<count; ++i) {
            idcs[i] = i;
            rnd[i]  = rng();
        }

        std::sort(idcs.begin(), idcs.end(), [&rnd](size_t a, size_t b){
            // stable sort for reproducibility
            return (rnd[a] == rnd[b]) ? (a < b) : (rnd[a] < rnd[b]);
        });
    }

    // create random offsets
    for (unsigned i=0; i<count; ++i) {
        shuffled_offs[i] = (size_t) ((sizes[idcs[i]] - 1) * ((double) rng() / (double) (rng.max()-1)));
    }

    // reorder begins and sizes by sorted indices
    for (unsigned i=0; i<count; ++i) {
        shuffled_begins[i] = begins[idcs[i]];
    }

    for (unsigned i=0; i<count; ++i) {
        shuffled_sizes[i] = sizes[idcs[i]];
    }

    return mt19937_get_state(rng);
}

void Optimizer::Init_CallbackData(struct llama_context * lctx,struct train_params_common& train_params,hGensor  tokens_input,int flag) {
    app_ctx = lctx;

    struct train_opt_callback_data& opt_cb_data = train_loader.callback_data;
    opt_cb_data.params                 = &(train_params); //hparams.common
    opt_cb_data.train                  = train;
    opt_cb_data.save_cb                = nullptr;   //&save_train_;
    opt_cb_data.save_data              = nullptr;   //&save_data;
    opt_cb_data.lctx                   = lctx;
    opt_cb_data.last_save_iter         = opt->iter;
    // opt_cb_data.tokens_data            = train_tokens.data();
    // opt_cb_data.tokens_size            = train_tokens.size();
    // opt_cb_data.samples_begin          = train_samples_begin.data();
    // opt_cb_data.samples_size           = train_samples_size.data();
    // opt_cb_data.samples_count          = train_samples_size.size();
    train_loader.train = train;         val_loader.train = train;
    train_loader.hOPT = this;           val_loader.hOPT = this;

    // opt_cb_data.shuffled_samples_offs  = train_shuffled_samples_offs.data();
    // opt_cb_data.shuffled_samples_begin = train_shuffled_samples_begin.data();
    // opt_cb_data.shuffled_samples_size  = train_shuffled_samples_size.data();

    opt_cb_data.tokens_input           = tokens_input;
    opt_cb_data.target_probs           = hTargetProbs();
    first_iter             = opt->iter;
    opt_cb_data.first_epoch            = train->train_epochs;
    opt_cb_data.iter_at_last_epoch     = -1;
    opt_cb_data.last_time              = ggml_time_ms();
    opt_cb_data.millis_per_iter        = 0.0;
}

bool Optimizer::one_step(struct train_state *train,DataLoader&loader, int accum_step, float *sched, int flag)    {
    // LearningCurve(0x0);
    struct train_params_common *params = &(train_params);   //data->params;
    // struct train_state *train = data->train;
    struct ggml_opt_context *opt = train->opt;
    int n_batch = params->n_batch;
    int n_ctx = params->n_ctx;

    if (accum_step == 0)        {
        // time measurement
        int64_t now = ggml_time_ms();
        if (now > last_time && opt->iter > first_iter)            {
            double dt = (double)(now - last_time);
            if (millis_per_iter == 0.0)                {
                millis_per_iter = dt;
            }                else                {
                const double gain = 0.7;
                millis_per_iter = millis_per_iter * (1.0 - gain) + dt * gain;
            }
        }
        double remaining_millis = 0.0;
        if (millis_per_iter > 0.0)            {
            const int n_iter = params->adam_n_iter;
            const int done_iter = opt->iter - first_iter;
            const int remaining_iter = n_iter - done_iter;
            remaining_millis = remaining_iter * millis_per_iter;
        }
        // file saving
        /*const bool save_now = (params->save_every > 0) && (opt->iter - last_save_iter >= params->save_every);
        if (save_now)            {                
            int new_iters = opt->iter - last_save_iter;
            train->train_its += new_iters;
            train->train_tokens += new_iters * opt->params.n_gradient_accumulation * n_batch * n_ctx;
            SaveTrain(&save_data, train);
            last_save_iter = opt->iter;
        }*/

        // exclude file saving from time measurement, by measuring last_time after saving
        last_time = ggml_time_ms();
        *sched = learning_schedule(opt->iter, params->warmup, params->cos_decay_steps, params->adam_alpha, params->adam_min_alpha,
                                    params->cos_decay_min, params->cos_decay_restart, params->enable_restart);
        int impr_plot = -(int)(1 + (opt->loss_before - opt->loss_after) * 10.0f + 0.5f);
        if (impr_plot > 0)
            impr_plot = 0;
        if (std::isnan(opt->loss_before) || std::isnan(opt->loss_after))
            impr_plot = 0;
        _INFO("[train] iter=%6d sample=%zu/%zu sched=%.3f loss=%f ",
                opt->iter, std::min(1 + train->shuffle_next_sample, train->shuffle_sample_count), train->shuffle_sample_count,
                *sched, opt->loss_after); //,hOPT->zmuv_0,hOPT->zmuv_1
        if (millis_per_iter > 0)            {
            _INFO(" dt=");          _TIME(millis_per_iter);
            _INFO(" eta=");         _TIME(remaining_millis);
        }
        float improvement = opt->loss_before - opt->loss_after;
        const float plot_scale = 10.0f;
        int bar_len = (int)(1 + improvement * plot_scale + 0.5);
        _INFO(" |");
        for (int i = 0; i < bar_len; ++i)            {
            _INFO("-");
        }
        _INFO(">");            _INFO("\n");
    }

    if (train->shuffle_next_sample >= train->shuffle_sample_count)
    {
        ++train->train_epochs;
        _INFO("%s: reshuffle samples. completed epochs: %llu\n", __func__, (long long unsigned)train->train_epochs);
        // note: we may have used some samples from the current shuffling more than once
        train->shuffle_rng_state_current = train->shuffle_rng_state_next;
        // train->shuffle_rng_state_next = shuffle_samples(
        //     train->shuffle_rng_state_current,data->shuffled_samples_offs,data->shuffled_samples_begin,
        //     data->shuffled_samples_size,data->samples_begin,data->samples_size,data->samples_count);
        
        loader.Shuffle();           //SAMP_0816
        // train->shuffle_rng_state_next = shuffle_samples(
        //     train->shuffle_rng_state_current,loader.shuffled_samples_offs.data(),
        //     loader.shuffled_samples_begin.data(),loader.shuffled_samples_size.data(),
        //     loader.samp_begin.data(),loader.samp_size.data(),loader.samp_size.size());
        
        train->shuffle_next_sample = 0;
    }

    const bool last_epoch_reached = (params->n_epochs > 0 && (int64_t)train->train_epochs - first_epoch >= params->n_epochs);
    if (last_epoch_reached)    {
        // allow optimization iteration at last epoch to be completed before canceling
        if (iter_at_last_epoch < 0)        {
            iter_at_last_epoch = opt->iter;
        }
        else if (opt->iter > iter_at_last_epoch)        {
            return true;
        }
    }
    return false;
}