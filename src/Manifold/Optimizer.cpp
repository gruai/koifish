/**
 *  Copyright 2023-2024 by Grusoft 
 * 
 *  \brief A collection of neurons
 *  \author Yingshi Chen
 */

#include "Ganglia.hpp"
#include "../ggex/GG_util.hpp"

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

enum ggml_opt_result Optimizer::ggml_train(struct ggml_context * ctx, struct train_opt_callback_data *callback_data, hGensor loss_,hGensor target_,
    struct ggml_cgraph * gf_,struct ggml_cgraph * gb_)    {
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
    _INFO("Optimizer::%s: GradAccumulation=%d ...... \n", __func__,(int)isAccuX );
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
                for (int p = 0; p < np; ++p) {
                    const int64_t ne = ggml_nelements(opt_ps[p]);
                    const float p_decay = ((ggml_n_dims(opt_ps[p]) >= decay_min_ndim) ? decay : 0.0f) * sched;
                    for (int64_t j = 0; j < ne; ++j) {
                        float x  = ggml_get_f32_1d(opt_ps[p], j);
                        float g_ = g[i]*gnorm;
                        m[i] = m[i]*beta1 +    g_*(1.0f - beta1);
                        v[i] = v[i]*beta2 + g_*g_*(1.0f - beta2);
                        float mh = m[i]*beta1h;
                        float vh = v[i]*beta2h;
                        vh = sqrtf(vh) + eps;
                        x  = x*(1.0f - p_decay) - mh/vh;
                        ggml_set_f32_1d(opt_ps[p], j, x);
                        ++i;
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