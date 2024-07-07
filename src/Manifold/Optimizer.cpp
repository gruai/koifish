#include "Ganglia.hpp"
#include "../ggex/GG_util.hpp"
#include "../LLAMA/common/common.h"

bool Optimizer::GradAccumulation(float&fx,int np,struct train_opt_callback_data *callback_data,struct ggml_cplan *cplan,int flag)   {    
    fx = 0;
    float * g  = (float *)opt->adam.g->data;
    ggml_set_zero(opt->adam.g);   
    assert(gb!=nullptr);
    float sched = opt->params.adam.sched;        
    const int n_accum = MAX(1, opt->params.n_gradient_accumulation);
    const float accum_norm = 1.0f / (float) n_accum;
    bool cancel = false;
    for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
        if(gang->one_step(callback_data,accum_step, &sched)){
            return false;
        }
        // if (callback) {
        //     callback(callback_data, accum_step, &sched, &cancel);
        //     if (cancel) {
        //         return false;      
        //     }
        // }
        // ggml_graph_reset  (gf);
        ggml_set_f32      (loss->grad, 1.0f);
        ggml_graph_compute(gb, cplan);
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

enum ggml_opt_result Optimizer::ggml_train(struct ggml_context * ctx, struct train_opt_callback_data *callback_data, hGensor loss_,hGensor target_,
    struct ggml_cgraph * gf_,struct ggml_cgraph * gb_,CLI_params& hparams)    {
    assert(loss==nullptr && target_probs==nullptr);
    loss = loss_;       target_probs = target_;
    callback_data->target_probs = target_probs;
    gf=gf_;      gb=gb_;

    enum ggml_opt_result result = GGML_OPT_RESULT_DID_NOT_CONVERGE;
    struct ggml_opt_params params = opt->params;
    GGML_ASSERT(ggml_is_scalar(loss));
    bool isAccuX = true,   cancel = false;      //Converge much faster!!! see Lay5_accux_.info
    float *fLoss = (float*)(loss->data),*fLossG = (float*)(loss->grad->data);
    // float *target = (float*)(data->target_probs->data);
    _INFO("Optimizer::%s: GradAccumulation=%d rZMUV=%g rLARS=%g...... \n\n", __func__,(int)isAccuX,hparams.ZMUV_ratio,hparams.lars_ratio );
    if(0){
        result = ggml_opt_resume_g(ctx, opt, loss, gf, gb, &train_opt_callback, callback_data);
    }else{
        //struct ggml_tensor * opt_ps[GGML_MAX_PARAMS];  these will store the parameters we want to optimize
        int np = 0;
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

        // constants
        float sched = params.adam.sched;
        const float alpha = params.adam.alpha;
        const float decay = params.adam.decay * alpha;
        const float beta1 = params.adam.beta1;
        const float beta2 = params.adam.beta2;
        const float eps   = params.adam.eps;
        const float gclip = params.adam.gclip;
        const int decay_min_ndim = params.adam.decay_min_ndim;
        const int n_accum = MAX(1, params.n_gradient_accumulation);
        const float accum_norm = 1.0f / (float) n_accum;

        float * g  = (float *)opt->adam.g->data;  // gradients
        float * m  = (float *)opt->adam.m->data;  // first moment
        float * v  = (float *)opt->adam.v->data;  // second moment
        float * pf = params.past > 0 ? (float *)opt->adam.pf->data : NULL; // past function values
        float fx = 0;
        struct ggml_cplan cplan = ggml_graph_plan(gb, params.n_threads);
        struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_WORK_BUFFER, cplan.work_size);
        cplan.work_data = (uint8_t *)ctx->mem_buffer + obj->offs;

        if(isAccuX){
            if( !GradAccumulation(fx,np,callback_data,&cplan,0x0))
                return GGML_OPT_RESULT_CANCEL; 
        }else   { 
            ggml_set_zero(opt->adam.g);            
            for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
                if(gang->one_step(callback_data,accum_step, &sched)){
                        return GGML_OPT_RESULT_CANCEL;
                    }
                ggml_set_f32      (loss->grad, 1.0f);
                ggml_graph_compute(gb, &cplan);
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
            {
                float gnorm = 1.0f;
                /**
                 * 1. LARS/LAMB  trus_ratio - ratio between the norm of the layer weights and norm of gradients update
                 */
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

            if(isAccuX){
                if( !GradAccumulation(fx,np,callback_data,&cplan,0x0))
                    return GGML_OPT_RESULT_CANCEL; 
            }else   { 
                fx = 0;
                ggml_set_zero(opt->adam.g);
                for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
                    if(gang->one_step(callback_data,accum_step, &sched)){
                        return GGML_OPT_RESULT_CANCEL;
                    }
                    // ggml_graph_reset  (gf);
                    ggml_set_f32      (loss->grad, 1.0f);
                    ggml_graph_compute(gb, &cplan);
                    ggml_opt_acc_grad(np, opt_ps, g, accum_norm);
                    fx += ggml_get_f32_1d(loss, 0);
                }
                fx *= accum_norm;
            }

            opt->loss_after = fx;
            UpdateLoss(opt->iter,sched,opt->loss_after);
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

void Ganglia::Train(  int flag )   {
    
}

int64_t Ganglia::update_batch(struct train_opt_callback_data *data,int next_id,struct train_params_common *params) {
    struct llama_context * lctx=data->lctx;
    struct ggml_tensor   * tokens_input=data->tokens_input;
    struct ggml_tensor   * target_probs=data->target_probs;
    int64_t                example_id=next_id;
    const size_t         * samples_offs=data->shuffled_samples_offs;
    const size_t         * samples_begin=data->shuffled_samples_begin;
    const size_t         * samples_size=data->shuffled_samples_size;
          size_t           samples_count=data->samples_count;
    const llama_token    * train_data=data->tokens_data;
    size_t                 n_train_data=data->tokens_size;
    bool                   separate_with_eos=params->separate_with_eos;
    bool                   separate_with_bos=params->separate_with_bos;
    bool                   fill_with_next_samples=params->fill_with_next_samples;
    bool                   sample_random_offsets=params->sample_random_offsets;
    GGML_ASSERT(samples_count > 0);
    GGML_ASSERT(ggml_is_matrix(tokens_input));
    GGML_ASSERT(ggml_is_3d(target_probs));
    int64_t n_vocab  = target_probs->ne[0],n_tokens = tokens_input->ne[0],n_batch  = tokens_input->ne[1];
    GGML_ASSERT(n_vocab  == target_probs->ne[0]);
    GGML_ASSERT(n_tokens == target_probs->ne[1]);
    GGML_ASSERT(n_batch  == target_probs->ne[2]);

    int64_t used_samples = 0;
    ggml_set_f32(target_probs, 0.0f);
    llama_token bos = llama_token_bos(llama_get_model(lctx));
    llama_token eos = llama_token_eos(llama_get_model(lctx));
    std::string sBos = llama_token_to_piece(lctx, bos),sEos = llama_token_to_piece(lctx, eos);
    
    // LLAMA_LOG_INFO("%s: example_id=%d n_batch=%d n_train_samples=%zu\n", __func__, example_id, n_batch, n_train_samples);
    _INFO("\t%s::%ld nSampe=(%ld/%ld)",__func__,example_id, samples_count,n_train_data);
    for (int k=0; k<n_batch; ++k) {
        // LLAMA_LOG_INFO("%s: batch %d\n", __func__, k);
        std::vector<int32_t> tok_ids;
        size_t sample_idx   = (example_id + used_samples) % samples_count;
        size_t sample_offs  = sample_random_offsets ? samples_offs[sample_idx] : 0;
        size_t sample_begin = samples_begin[sample_idx];
        size_t sample_size  = samples_size[sample_idx];
        ++used_samples;
        assert(sample_offs==0);
        
        // LLAMA_LOG_INFO("%s: sample_idx=%zu sample=%zu\n", __func__, sample_idx, sample);
        GGML_ASSERT(sample_begin+sample_size-1 < n_train_data);
        std::string sentence="";
        ggml_set_i32_nd(tokens_input, 0, k, 0, 0, bos);
        bool sample_separation_eos = !separate_with_eos;
        bool sample_separation_bos = !separate_with_bos;
        for (int64_t i=0; i<n_tokens; ++i) {
            llama_token token = eos;
            if (sample_offs >= sample_size && fill_with_next_samples) { //true only arg == "--fill-with-next-samples"
                if (!sample_separation_eos) {
                    // insert eos token to separate samples
                    sample_separation_eos = true;
                } else if (!sample_separation_bos) {
                    // insert bos token to separate samples
                    sample_separation_bos = true;
                    token = bos;
                } else {
                    // sample separation is done, continue with next sample
                    sample_separation_eos = !separate_with_eos;
                    sample_separation_bos = !separate_with_bos;
                    sample_offs  = 0;
                    sample_idx   = (example_id + used_samples) % samples_count;
                    sample_begin = samples_begin[sample_idx];
                    sample_size  = samples_size[sample_idx];
                    ++used_samples;
                }
            }
            // note: no else-if here
            if (sample_offs < sample_size) {
                token = clamp(train_data[sample_begin+sample_offs], 0, (llama_token) (n_vocab - 1));
                ++sample_offs;
            }
            ggml_set_f32_nd(target_probs,  token, (int) i, (int) k, 0, +1.0f);
            tok_ids.push_back(token);
            
            if (i+1<n_tokens) {
                ggml_set_i32_nd(tokens_input, (int) (i + 1), (int) k, 0, 0, token);
                sentence = llama_token_to_piece(lctx, token);
                // _INFO("%s,",sentence.c_str());
            }
        }
        _INFO(" %ld@\"%s...\"",sample_begin,sentence.c_str());     //sample_size
        if(wiki!=nullptr && wiki->logits!=nullptr){
            assert(target_probs->type == GGML_TYPE_F32);
            wiki->Decode(tok_ids,0x0);
            auto g=wiki->logits;   // wiki->logits = ggml_new_tensor_3d(ctx_input, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);
            int ld0=g->nb[0],ld1=g->nb[1],ld2=g->nb[2],ld3=g->nb[3];          
            assert(ld0==4); 
            float *logits = wiki->logits_out;   //n_vocab,nToken,
            // target=(float*)((char *)(target_probs->data)+i*ld1+k*ld2);
            assert(sizeof(float)*n_vocab*n_tokens==ld2);    
            memcpy(g->data+k*ld2,logits,sizeof(float)*n_vocab*n_tokens);                
            for (int64_t i=0; i<n_tokens; ++i) {
                /*llama_token token = tok_ids[i];
                for(int j=0;j<n_vocab;j++,logits++){                    
                    // target[j]=1.0f-*logits;
                    // void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
                    // ggml_set_f32_nd(target_probs, j, (int) i, (int) k, 0, +1.0f-*logits);
                }*/
                    
            }
        }        
    }_INFO("\n");

    return used_samples;
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
