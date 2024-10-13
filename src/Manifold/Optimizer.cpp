#include "gLLM.hpp"
#include "Dictionary.hpp"
#include "../ggex/GG_util.hpp"

 Optimizer::Optimizer(LLaMeta *g_,struct train_params_common& params_,int flag) : train_params(params_),gang(g_) {
    // InitOpt(train_params,flag); 
    val_loader.type = SampLoader::TYPE::DT_EVAL;
    if(gang->isTrain())  {
        train_loader.Init(g_);
        
    }else{
       
    }
    val_loader.Init(g_); 
}

hGensor Optimizer::hLoss()             {    
    assert(gang!=nullptr);
    if(gang->loss!=nullptr)
        GGML_ASSERT(ggml_is_scalar(gang->loss));  
    return gang->loss;          
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

bool OPT_Adam::BatchGrad(float&fx,struct train_opt_callback_data *cb0,struct ggml_cplan *cplan,int flag)   {    
    fx = 0;
    float * g  = (float *)grad->data;
    auto loss = hLoss();
    float *fLoss = (float*)(loss->data),*fLossG = (float*)(loss->grad->data);
    ggml_set_zero(grad);   
    struct ggml_cgraph *_gf=gang->ForwarGraph(),*_gb=gang->BackwardGraph();
    assert(_gb!=nullptr);
      
    const int n_accum = MAX(1, train_params.n_gradient_accumulation);
    const float accum_norm = 1.0f / (float) n_accum;
    bool cancel = false;
    struct train_state *train = nullptr;    //callback_data->train;
    for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
        int64_t used_samples = train_loader.update_batch(-1,gang);
        train_samples += used_samples;      
        if(one_step(train,train_loader,accum_step, &sched)){
            return false;
        }

        ggml_set_f32      (loss->grad, 1.0f);
        ggml_graph_compute(_gb, cplan);
        OnLogits();
        ggml_opt_acc_grad(opt_ps.size(), opt_ps.data(), g, accum_norm);        
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

float Optimizer::gClip(int nx,CLI_params& hparams,int flag)  {
    // struct ggml_opt_params params = opt->params;
    float * g  = (float *)grad->data;  // gradients
    // const float gclip = params.adam.gclip;
    float gnorm = 1.0f;     
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
    return gnorm;
}

/**
 * 1. LARS/LAMB  trus_ratio - ratio between the norm of the layer weights and norm of gradients update
 */
void Optimizer::UpdateParams(float gnorm,int nx,CLI_params& hparams,int flag)  {
    /*struct ggml_opt_params params = opt->params;
    float sched = params.adam.sched;
    const float alpha = params.adam.alpha;
    const float decay = params.adam.decay * alpha;
    const float beta1 = params.adam.beta1;
    const float beta2 = params.adam.beta2;
    const float eps   = params.adam.eps;    
    const float gclip = params.adam.gclip;
    const int decay_min_ndim = params.adam.decay_min_ndim;
    
    float * g  = (float *)g->data;  // gradients
    float * m  = (float *)m->data;  // first moment
    float * v  = (float *)v->data;  // second moment
    float * pf = params.past > 0 ? (float *)pf->data : NULL; // past function values
    float fx = 0;
    
    const float beta1h = alpha*sched/(1.0f - powf(beta1, opt->iter));                
    const float beta2h =        1.0f/(1.0f - powf(beta2, opt->iter));
    int64_t i = 0;
    zmuv_0 = DBL_MAX,zmuv_1 = 0.0;
    for (int p = 0; p < opt_ps.size(); ++p) {
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
    }   */
}

void OPT_Adam::UpdateParams(float gnorm,int nx,CLI_params& hparams,int flag)  {
    if(!isAdaptiveSched)        //params.adam.sched;   
        sched = 1.0;  
    float * g  = (float *)grad->data;  // gradients
    float * m  = (float *)gm->data;  // first moment
    float * v  = (float *)gv->data;  // second moment
    float * pf = past > 0 ? (float *)gpf->data : NULL; // past function values
    float fx = 0;
   
    beta1h = alpha*sched/(1.0f - powf(beta1, iter));                
    beta2h =        1.0f/(1.0f - powf(beta2, iter));
    int64_t i = 0;
    zmuv_0 = DBL_MAX,zmuv_1 = 0.0;
    for (auto hP : opt_ps) {
        p_decay = ((ggml_n_dims(hP) >= decay_min_ndim) ? decay : 0.0f) * sched;
        const int64_t ne = ggml_nelements(hP);
        UpdateTensorParam(hP,m+i,v+i,g+i,gnorm);
        i += ne;       
    }  
}

void OPT_Adam::UpdateTensorParam(hGensor hP,float *m,float *v,float *g,float gnorm){ 
    double normX = 0.0,meanX=0;
    const int64_t ne = ggml_nelements(hP);
    for (int64_t j = 0; j < ne; ++j,m++,v++,g++) {
        float x  = ggml_get_f32_1d(hP, j);
        // float g0 = isZMUV ? *g : *g*gnorm;
        float g0 = *g*gnorm;
        *m = *m*beta1 +    g0*(1.0f - beta1);   //linear interpolations of the gradients 
        *v = *v*beta2 + g0*g0*(1.0f - beta2);   //linear interpolations of their variances
        float mh = *m*beta1h;     //debiasing 
        float vh = *v*beta2h;
        vh = sqrtf(vh) + eps;
        x  = x*(1.0f - p_decay) - mh/vh;        //ormalize step(mh) by its standard deviation(vh)                     
        ggml_set_f32_1d(hP, j, x);
    }
}

void OPT_AdamMiniV::UpdateTensorParam(hGensor hP,float *m,float *v0,float *g,float gnorm){       
    bool isEmbed = true;
    if(isEmbed) {
        OPT_Adam::UpdateTensorParam(hP,m,v0,g,gnorm);
        return;
    }    

    const int64_t ne = ggml_nelements(hP);
    float v_hat = 0;
    // *v = *v*beta2 + g0*g0*(1.0f - beta2);   //linear interpolations of their variances
    for (int64_t j = 0; j < ne; ++j,m++,g++) {
        float x  = ggml_get_f32_1d(hP, j);
        // float g0 = isZMUV ? *g : *g*gnorm;
        float g0 = *g*gnorm;
        *m = *m*beta1 +    g0*(1.0f - beta1);   //linear interpolations of the gradients 
        float mh = *m*beta1h;     //debiasing 
        float vh = v_hat*beta2h;
        vh = sqrtf(vh) + eps;
        x  = x*(1.0f - p_decay) - mh/vh;        //ormalize step(mh) by its standard deviation(vh)                     
        ggml_set_f32_1d(hP, j, x);
    }
}


enum ggml_opt_result Optimizer::Search(struct ggml_context * ctx, hGensor loss_,hGensor target_,CLI_params& hparams)    {
    // struct train_opt_callback_data *callback_data = &(train_loader.callback_data);
    // assert(loss==nullptr && target_probs==nullptr);
    // loss = loss_;       target_probs = target_;
    // preLogits = gang->preLogits;
    // callback_data->target_probs = target_probs;
    // tokens_input = callback_data->tokens_input;
    
    struct ggml_cgraph *_gf=gang->ForwarGraph(),*_gb=gang->BackwardGraph();

    enum ggml_opt_result result = GGML_OPT_RESULT_DID_NOT_CONVERGE;
    // struct ggml_opt_params params = opt->params;
   
    bool cancel = false;      //isAccuX = true
    auto loss = hLoss( );
    float *fLoss = (float*)(loss->data),*fLossG = (float*)(loss->grad->data),val_loss;
    
    // float *target = (float*)(data->target_probs->data);
    _INFO("Optimizer::%s: Accumulation=%d AdaptiveSched=%d rZMUV=%g rLARS=%g...... \n\n", __func__,
        train_params.n_gradient_accumulation,(int)isAdaptiveSched,hparams.ZMUV_ratio,hparams.lars_ratio );
    if(0){  //only for old version
        // result = ggml_opt_resume_g(ctx, opt, loss, gf, gb, &train_opt_callback, callback_data);
    }else{        
        /*if ((opt->params.type != params.type) || (opt->nx != nx) || (opt->params.past != params.past)) {   //resume
            int iter = opt->iter;
            ggml_opt_init(opt->ctx, opt, params, nx);
            opt->iter = iter;
        }*/
           
        float fx = 0;
        auto *cplan = &(gang->gb_plan);

        if( !BatchGrad(fx,nullptr,cplan,0x0))
            return GGML_OPT_RESULT_CANCEL;         

        fx_prev.push_back(fx);        fx_best.push_back(fx);   
        loss_before = fx;        loss_after  = fx;

        // initialize
        if (just_initialized) {
            n_no_improvement = 0;
            just_initialized = false;
        }

        int iter0 = 0;  //opt->iter;
        // run the optimizer
        for (int t = 0; t < train_params.adam_n_iter; ++t) {
            iter = iter0 + t + 1;
            /*GGML_PRINT_DEBUG  ("=== iter %d ===\n", t);
            GGML_PRINT_DEBUG  ("loss      = %10.6f\n", ggml_get_f32_1d(loss, 0));
            GGML_PRINT_DEBUG_5("df/dx0 = %10.6f\n", ggml_get_f32_1d(opt_ps[0]->grad, 0));
            GGML_PRINT_DEBUG_5("df/dx1 = %10.6f\n", ggml_get_f32_1d(opt_ps[1]->grad, 0));
            for (auto hT : opt_ps) {
                GGML_PRINT_DEBUG("param %d: %10.6f, g = %10.6f\n", i,ggml_get_f32_1d(hT, 0), ggml_get_f32_1d(hT->grad, 0));
            }*/
            const int64_t t_start_wall = ggml_time_us(),t_start_cpu = ggml_cycles();
            float gnorm = gClip(nParams,hparams);
            UpdateParams(gnorm,nParams,hparams,0x0);   
            // AdamMiniV(gnorm,nx,hparams,0x0);   
            //gradient is useless at this stage  
            if (hparams.eval_every>0 && t % hparams.eval_every == 0) {
                val_loss = Evaluate(val_loader,t);  
            }        
            if( hparams.gpt_every>0 && t%hparams.gpt_every==0 )   {
                gang->GenSentence(1);   
            }   
            
            if( !BatchGrad(fx,nullptr,cplan,0x0))
                return GGML_OPT_RESULT_CANCEL;            

            loss_after = fx;
            UpdateLoss(iter,fx);        //scheduler->Append(loss);
            if(gang->hDistler!=nullptr) 
                gang->hDistler->UpdateSigma(iter);
            // check convergence
            if (fabsf(fx - fx_prev[0])/fx < train_params.adam_eps_f) {
                GGML_PRINT_DEBUG("converged\n");                result = GGML_OPT_RESULT_OK;
            }
            // delta-based convergence test
            /*if (pf != NULL) {
                // need at least params.past iterations to start checking for convergence
                if (params.past <= iter0 + t) {
                    const float rate = (pf[(iter0 + t)%params.past] - fx)/fx;
                    if (fabsf(rate) < params.delta) {
                        result = GGML_OPT_RESULT_OK;
                    }
                }
                pf[(iter0 + t)%params.past] = fx;
            }*/
            // check for improvement
            if (train_params.opt_max_no_improvement > 0) {
                if (fx_best[0] > fx) {
                    fx_best[0] = fx;
                    n_no_improvement = 0;
                } else {
                    ++n_no_improvement;

                    if (n_no_improvement >= train_params.opt_max_no_improvement) {
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
    struct ggml_cgraph *_gf=gang->ForwarGraph(),*_gb=gang->BackwardGraph();
    assert(_gf!=nullptr);
    auto loss = hLoss( );
    float *fLoss = (float*)(loss->data),*fLossG = onlyForward ? nullptr:(float*)(loss->grad->data);    
    auto cplan =&(gang->gb_plan);
    llama_token token;
    auto tokens_input = gang->Input( );
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
        ggml_graph_compute(_gf, cplan);   
    else{
        float * g  = (float *)grad->data;
        ggml_set_zero(grad);  
        ggml_graph_compute(_gb, cplan);   /**/
    }
        
    // if(nSamp<nB0){
    //     tokens_input->ne[1] = nB0;
    // }
    return *fLoss;
}

/*
    REF:    /home/cys/Github/llama.cpp-master/examples/batched/batched.cpp
*/
float Optimizer::Evaluate(SampLoader&loader,int iter,int flag){  
    if( val_loader.num_batches==0) 
        return 0; 
    if(iter!=-666)   _INFO("[eval] " );   
    GST_TIC(tic);     
    struct ggml_cgraph *_gf=gang->ForwarGraph();
    auto loss = hLoss();     
    double l2,delta_max=0,delta_=0,a,sum=0;
    auto tokens_input = gang->Input( ); //,ldT=tokens_input->ne[0]
    int i,nB=0,step=max(loader.num_batches/10,1);
    size_t nz=0,j;
    llama_token tMost = (llama_token) (loader.n_vocab - 1);
    
    const float *wLog = nullptr;
    for (i = 0; i < loader.num_batches; i+=step) {
        if(tokens_input!=nullptr)   //in some debug mode, tokens_input maybe null
            loader.update_batch(i,gang);    
        if(wLog!=nullptr)    {
            for (l2 = 0,j = 0; j < nz; j++    ) {
                a = wLog[j];            l2 += a*a;              
            }
            l2 = sqrt(l2)/nz;            
            delta_max = max(delta_max,l2);      delta_+=l2;        
        }
        // ggml_graph_compute(gb, cplan);     
        ggml_graph_compute(_gf, &(gang->gb_plan));     //((float*)hPreLogits()->data)[0]
        sum += loss==nullptr ? 0 : ((float*)(loss->data))[0];         //float *fLoss = (float*)(loss->data)
        nB++;
        break;
    }
    if(iter==-666)    {   //hack
        return i;
    }
    float last = lcEval.Last(),aloss = sum/nB;  //[eval]   Loss@Evaluation=7.302641 T=0.232s ======
    lcEval.Add(aloss);
    float delta=last-aloss,best=lcEval.Best();   
    bool isOverfit = delta<0 && abs(aloss-best)>best/10;       
    if(isOverfit)   {
        _INFO(" !OVERFIT! ");
    }
    _INFO(" Loss@EvalSet=%f(%.2g) best=%f(eval_%d) E2T=%.3g T=%gs ======\n",aloss,delta,best,lcEval.best_id,
        aloss-lcTrain.Last(),GST_TOC(tic) );

    if(wLog==nullptr) {     }        
    else
        _INFO("\t Loss@Evaluation=%f delta=%g(%.5g) T=%gs ======\n", aloss,delta_max,delta_/nB,GST_TOC(tic) );
    
    return aloss;
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



bool Optimizer::one_step(struct train_state *trainst0,SampLoader&loader, int accum_step, float *sched, int flag)    {
    // LossCurve(0x0);
    struct train_params_common *params = &(train_params);   //data->params;
    // struct train_state *train = data->train;
    // struct ggml_opt_context *opt = train->opt;
    int n_batch = params->n_batch;
    int n_ctx = params->n_ctx;

    if (accum_step == 0)        {
        // time measurement
        int64_t now = ggml_time_ms();
        if (now > last_time && iter > first_iter)            {
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
            const int done_iter = iter - first_iter;
            const int remaining_iter = n_iter - done_iter;
            remaining_millis = remaining_iter * millis_per_iter;
        }        

        // exclude file saving from time measurement, by measuring last_time after saving
        last_time = ggml_time_ms();
        *sched = learning_schedule(iter, params->warmup, params->cos_decay_steps, params->adam_alpha, params->adam_min_alpha,
                                    params->cos_decay_min, params->cos_decay_restart, params->enable_restart);
        int impr_plot = -(int)(1 + (loss_before - loss_after) * 10.0f + 0.5f);
        if (impr_plot > 0)
            impr_plot = 0;
        if (std::isnan(loss_before) || std::isnan(loss_after))
            impr_plot = 0;
        _INFO("[train]_%-6d\tsample@%zu/%zu sched=%.3f loss=%f ",
                iter, std::min(1 + train_loader.next_sample, train_loader.shuffle_sample_count), train_loader.shuffle_sample_count,
                *sched, loss_after); //,zmuv_0,zmuv_1
        lcTrain.Add(loss_after);
        if (millis_per_iter > 0)            {
            _INFO(" dt=");          _TIME(millis_per_iter);
            _INFO(" eta=");         _TIME(remaining_millis);
        }
        float improvement = loss_before - loss_after;
        _INFO("\n");
    }

    if (train_loader.next_sample >=train_loader.shuffle_sample_count)    {
        ++train_epochs;
        _INFO("%s: reshuffle samples. completed epochs: %llu\n", __func__, train_epochs);
        // note: we may have used some samples from the current shuffling more than once
        train_loader.shuffle_rng_state_current =train_loader.shuffle_rng_state_next;
        // train->shuffle_rng_state_next = shuffle_samples(
        //     train->shuffle_rng_state_current,data->shuffled_samples_offs,data->shuffled_samples_begin,
        //     data->shuffled_samples_size,data->samples_begin,data->samples_size,data->samples_count);
        
        loader.Shuffle();           //SAMP_0816
        // train->shuffle_rng_state_next = shuffle_samples(
        //     train->shuffle_rng_state_current,loader.shuffled_samples_offs.data(),
        //     loader.shuffled_samples_begin.data(),loader.shuffled_samples_size.data(),
        //     loader.samp_begin.data(),loader.samp_size.data(),loader.samp_size.size());
        
        train_loader.next_sample = 0;
    }

    const bool last_epoch_reached = (params->n_epochs > 0 && train_epochs - first_epoch >= params->n_epochs);
    if (last_epoch_reached)    {
        // allow optimization iteration at last epoch to be completed before canceling
        if (iter_at_last_epoch < 0)        {
            iter_at_last_epoch = iter;
        }
        else if (iter > iter_at_last_epoch)        {
            return true;
        }
    }
    return false;
}

void Optimizer::BeforeTrain(struct llama_context * lctx,struct train_params_common& train_params,hGensor tokens_,int flag) {    
    first_iter             = iter;
    // train_loader.hOPT = this;           val_loader.hOPT = this;
    // assert(tokens_!=nullptr);
    // tokens_input = tokens_;

    opt_ps = gang->optParams;
    nMostParam = 0;
    for (auto ps : opt_ps) {            
        nMostParam += ggml_nelements(ps);
    } 
    assert(nMostParam>=nParams);

    // max_epoch = train_params.adam_n_iter
}

void Optimizer::Prepare(size_t nx_,int flag){
    // assert(nx_>=0);
    iter = 0;
    nParams = nx_;         
    just_initialized = true;
    if(nParams>0)   {
        struct ggml_init_params ctx_opt_params;    
        ctx_opt_params.mem_size = GGML_MEM_ALIGN*3 + ggml_tensor_overhead()*3 + ggml_type_size(GGML_TYPE_F32)*nParams*3;
        if (past > 0) {
            ctx_opt_params.mem_size += GGML_MEM_ALIGN + ggml_tensor_overhead() + ggml_type_size(GGML_TYPE_F32)*past;
        }    
        ctx_opt_params.mem_buffer = NULL;
        ctx_opt_params.no_alloc   = false;
        _ctx = ggml_init(ctx_opt_params);    
        if(gang->isTrain()) {
            grad = ggml_new_tensor_1d(_ctx, GGML_TYPE_F32, nParams);
        }
        train_loader.Prepare(this);
    }
    
    val_loader.Prepare(this);
}

bool Optimizer::PrepareData( CLI_params& hparams,int flag )   {   
    GST_TIC(tic);   

    bool isLoadOK = false;  
    string spTrain = hparams.serial_path+".train",spEval = hparams.serial_path+".eval";
    // auto& tokens = hTokenset->tokens;
    
    if(1)   {
        if( train_loader.Serialize(spTrain,false) 
            && val_loader.Serialize(spEval,false)){
                if(train_loader.len()>0){
                    // hDict->nUniqueToken = train_loader.n_unique_tokens; 
                    // _INFO("%s: nTrain=%zu nEval=%zu batch_sample=%s T=%.3g\n", __func__, train_loader.len(),val_loader.len(),GST_TOC(tic));
                    isLoadOK = true;
                }
        }            
    }
    if(!isLoadOK) {
        hDataToken hTokenset=train_loader.tokens;
        assert(hTokenset!=nullptr);
           
        std::vector<size_t> samples_begin,samples_size;
        // auto train_params = hparams.common;
        size_t nUnique = hTokenset->nUnique,nVocab=hTokenset->nVocab;
        // int n_ctx_tokens = hparams.n_ctx;
        if( hTokenset->InitSamps(hparams.common.n_ctx,samples_begin,samples_size)){

        }else{
            _INFO("%s: NULL Samps!!!    batch_sample=%s nTrain=%zu nEval=%zu T=%.3g\n", __func__, train_loader.batch_sample.c_str(),
                train_loader.len(),val_loader.len(),GST_TOC(tic));      
            return false;
        }        
  
        train_loader.SetSamples(nVocab,hTokenset,samples_begin,samples_size,true,hparams);    
        val_loader.SetSamples(nVocab,hTokenset,samples_begin,samples_size,false,hparams);
        
        // assert(val_loader.n_unique_tokens <= nUnique && train_loader.n_unique_tokens <= nUnique);
        // val_loader.n_unique_tokens = nUnique;
        // train_loader.n_unique_tokens = nUnique;
        shuffle_samples_hash = train_loader.shuffle_samples_hash;
        train_loader.Serialize(spTrain,true);
        // train_loader.Serialize(spTrain,false);      //only for debug
        val_loader.Serialize(spEval,true);
    }
    
    // GGML_ASSERT(train_samples_begin.size() == train_samples_size.size());
    _INFO("%s: batch_sample=%s nTrain=%zu nEval=%zu T=%.3g\n", __func__, train_loader.batch_sample.c_str(),
        train_loader.len(),val_loader.len(),GST_TOC(tic));        
    return true;
}

void OPT_Adam::Prepare(size_t nx_,int flag){
    Optimizer::Prepare(nx_,flag);
    if(gang->isTrain()){
        assert( grad!=nullptr );
        gm  = ggml_new_tensor_1d(_ctx, GGML_TYPE_F32, nParams);
        gv  = ggml_new_tensor_1d(_ctx, GGML_TYPE_F32, nParams);
        gpf = past > 0 ? ggml_new_tensor_1d(_ctx, GGML_TYPE_F32, past) : NULL;
        ggml_set_zero(gm);
        ggml_set_zero(gv);
        if (gpf) {
            ggml_set_zero(gpf);
        }
    }
}

OPT_Adam::OPT_Adam(LLaMeta *g_,struct train_params_common& params_,int flag)
    : Optimizer(g_,params_,flag)    {

    sched              = 1.0f;
    alpha              = train_params.adam_alpha;
    decay              = train_params.adam_decay * alpha;
    decay_min_ndim     = train_params.adam_decay_min_ndim;
    beta1              = train_params.adam_beta1;
    beta2              = train_params.adam_beta2;
    gclip              = train_params.adam_gclip;
    eps_f              = train_params.adam_eps_f;
}

void OPT_Adam::Dump(int typ){
    Optimizer::Dump(typ);
    _INFO("%s:  sched=%.4g ADAM(lr=%g,%g,[%g-%g]) decay=%g(%d)\n", __func__, sched,alpha,decay,beta1,beta2,
        decay,decay_min_ndim);
}