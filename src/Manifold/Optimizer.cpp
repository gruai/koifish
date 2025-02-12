/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *  
 *  Optimizer
 * 
 *  \brief Optimizer
 *  \author Yingshi Chen
 */
#include "Optimizer.hpp"
#include "gLLM.hpp"
#include "Dictionary.hpp"
#include "../ggex/GG_util.hpp"

int tpFuseCu = 1;
struct train_params_ Optimizer::TrainParams()    {   
    return _fish->hparams.common;   
}

Optimizer::Optimizer(NLP_AutoRegressive *g_, CLI_params& hparams,int flag) : _fish(g_) {    
    tpSign = hparams.Get({"train","optimizatioin","sign"},0,false);
    string method = hparams.Get({"train","optimizatioin","method"},string("adamw"),false);
    auto train_params=TrainParams();
    /*
        Although many people think "ADAMw is much better than SGD for attention models"  https://arxiv.org/pdf/2310.01082
        But may litter slower. For example:   jModel_SGD_v.info
    */
    tpGD = method=="adamw" ? ADAMw : method=="sgdv" ? SGD_v : method=="hsgd" ? SGD_HYBRID : ADAMw;

    nGradAccum =  std::max(1, train_params.n_gradient_accumulation);
    isGlobalGrad = nGradAccum>1;        // Nearly same alloc grad or not
    train_loader = std::make_shared<SampLoader>(_fish,"Train",false);
    val_loader = std::make_shared<SampLoader>(_fish,"Eval",false);
    val_loader->type = SampLoader::TYPE::DT_EVAL;
    if(_fish->isTrain())  {
        // train_loader->Init(g_,"Train");
        
    }else{
       
    }
}

hGensor Optimizer::hLoss()             {    
    assert(_fish!=nullptr);
    if(_fish->loss!=nullptr){
        // auto a = _fish->loss->Item();
        // assert(ggml_is_scalar(G(_fish->loss)));  
    }
        
    return _fish->loss;          
}

hGensor Optimizer::hTargetProbs()      {   return _fish->target_probs;  }
hGensor Optimizer::hPreLogits()       {   return _fish->preLogits;     }

hGensor Optimizer::GradOf(hGensor node,int flag){
    auto cgraph = _fish->GetBackRaw();
    return ::GradOf(cgraph,node,flag);    
}

/* Xoshiro256PlusSIMD       https://github.com/stephanfr/Xoshiro256PlusSIMD/blob/main/README.md
 * https://www.lomont.org/papers/2008/Lomont_PRNG_2008.pdf
 */
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
    size_t nz = tELEM(pTarget),nToken=pTarget->ne[0],n_ctx=pTarget->ne[1]; 

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

static void ggml_opt_acc_grad(int np, hGensor  const ps[], float * g, float scale) {
    int64_t i = 0;
    for (int p = 0; p < np; ++p) {
        const int64_t ne = tELEM(ps[p]) ;
        // TODO: add function to get all elements at once
        for (int64_t j = 0; j < ne; ++j) {
            assert(0);
            // g[i++] += ggml_get_f32_1d(ps[p]->grad, j) * scale;
        }
    }
}

bool cuClear(std::vector<hGTensor> tensors,int flag);
bool OPT_Adam::BatchGrad(int iter,float&fx,int flag)   {    
    fx = 0;
    auto loss = hLoss();
    float *fLoss = (float*)(loss->data),*g = nullptr,accum_norm = 1.0f/(float) nGradAccum;
    OutCLS* cls = _fish->GetNeuron<OutCLS>("OutCLS",0);
    cls->hLoader = train_loader;
    if(grad!=nullptr){
        ZERO_(grad);   
        g = (float *)grad->data;
    }
    bool bench=false;
    // struct ggml_cgraph *_gf=_fish->GetForwRaw(),*_gb=_fish->GetBackRaw();
    // assert(_gb!=nullptr);
    for (int accum_step = 0; accum_step < 1/*nGradAccum*/; ++accum_step) {
        // auto now = ggml_time_ms();        
        int64_t nSamp = train_loader->UpdateBatch(-1,_fish);
        if(nSamp==0)        {
            _WARN("<%s> Failed to get next batch!!!\n",__func__);
            return false;
        }
        
        train_samples += nSamp;      
        if(AfterLoadBatch(accum_step)){
            return false;
        }   
        
#ifdef _TENSOR_CUD_
        cuClear(opt_ps,0x0);    
        // double a = tNormOf(opt_ps,0x0);
        GraphCompute(train_loader,_fish->hBackTG);
#else
        auto grad = GradOf(loss);
        tSET(grad,1.0f);    //ggml_set_f32      (grad, 1.0f);
        if(bench){ //  only for performance benchmark
            GST_TIC(t0);
            // ggml_graph_comp0(_gf,0x0); 
            // ggml_graph_comp0(_gb,0x0); 
            _INFO("gb_compute %s T=%.3g","",GST_TOC(t0));  
            exit(-666);
        }else{
            GraphCompute(train_loader,_fish->hBackTG);
        }           
        
        OnLogits();
        if(isGlobalGrad){
            ggml_opt_acc_grad(opt_ps.size(), opt_ps.data(), g, accum_norm); 
        }else{  //  g[i++] += ggml_get_f32_1d(ps[p]->grad, j) * scale;
            
        } 
        fx += tGET(loss,0);  
        UpdateTrainLoss(-1,fx);
#endif               
       
        if(accum_step==0){
            // sched = UpdateSchedule(flag);
        }
            
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

int Optimizer::SignStochastic(int nx,CLI_params& hparams,int flag){    
    if(tpSign<=0)
        return tpSign;
    if(grad==nullptr){
        for (auto hP : opt_ps) {
            size_t ne = tELEM(hP);
            float *g =(float*)(GradOf(hP)->data);
            for (int64_t i = 0; i < ne; ++i) {
                g[i] = g[i]>0?1:-1;
            }    
        }  
    }else{
        float * g  = (float *)grad->data;  
        double sum=0.0,norm=0;
        for (int64_t i = 0; i < nx; ++i) {
            g[i] = g[i]>0?1:-1; //signed
            sum += (g[i]*g[i]);
        }
        norm = sqrt(sum);
        assert(norm<FLT_MAX);         
    }
    
    return 0x0;  
}

/**
 * 1. LARS/LAMB  trus_ratio - ratio between the norm of the layer weights and norm of gradients update
 */
void Optimizer::UpdateParams(int nx,CLI_params& hparams,int flag)  {    
}

void OPT_Adam::UpdateParams(int nx,CLI_params& hparams,int flag)  {
    floatX *g = nullptr;
    
    float clip = 0.f, fx = 0,sum,sched;
    g2_sum = 0;
    if(!isAdaptiveSched)        //params.adam.sched;   
        sched = 1.0;  
#ifdef _TENSOR_CUD_
    // sum = tNormOf(opt_ps,0x0);        //may have bug
    // g_step = sqrt(sum/nParams);
    if(DEBUG.train_hyperparams==1){
        adam.decay = 0;      sched = 1.0;
    }
#else
#endif
    if(grad!=nullptr){
        g = (floatX *)grad->data;  // gradients
        clip = gClip(nParams,g,nullptr);        
    }
    auto now = ggml_time_ms();    
    // float beta1_correction = 1.0f - powf(beta1, t);
    // float beta2_correction = 1.0f - powf(beta2, t);   
    beta1h =        sched/(1.0f - powf(adam.beta1, iter));                
    beta2h =        1.0f/(1.0f - powf(adam.beta2, iter));
    size_t i = 0;
    zmuv_0 = DBL_MAX,zmuv_1 = 0.0;
    if(iter==1)
        _INFO("clip=%g(%g) lr=%g beta1=%g , beta2=%g,  eps=%g , weight_decay=%g\n",1.0e-6,0.0,adam.alpha,adam.beta1, adam.beta2, adam.eps,adam.decay);
    for (auto hP : opt_ps) {
        p_decay = ((tDIM(hP) >= adam.decay_min_ndim) ? adam.decay : 0.0f) * sched;
        const int64_t ne = tELEM(hP);
        UpdateTensorParam(hP,i,g==nullptr?nullptr:g+i,clip);
        i += ne;       
    }
    tData = ggml_time_ms()-now;
    double a = sqrt(g2_sum),off=0;
#ifdef _TENSOR_CUD_
    off = a - g_step;
    assert(a);
#endif
    g_step = a;
}

float Optimizer::gClip(int ne,floatX *g,hGensor hP,int flag)  {
    float clip = 1.0f,a,a1=-FLT_MAX;     
    double sum=0.0;

    for (int64_t i = 0; i < ne; ++i,++g) {
        a = T2Float(g); 
        sum += a*a;     a1=std::max(fabs(a),a1);
    }

    g_ll = sum;             g2_sum += g_ll;
    double norm = sqrt(sum),avg = sqrt(sum/ne);      
    if (gclip > 0.0f) { //Gradient clipping maybe superior than L2 Regularization in terms of gaining results and easiness in implementation
        // gradient clipping
        assert(norm<FLT_MAX);         
        if (norm >gclip) {
            clip = (float) (gclip / norm);
        } 
    }
    assert(clip>0);
   
    if(hP!=nullptr){
        if(fabs(a1)<1.0e-10){
            _INFO("\t|g|=0@%s!",hP->name);
        }
        if(isnan(avg)){
            _INFO("\tNAN |g|@%s!!",hP->name);
        }        
    }

    return clip;
}

inline bool isStrMatch(const string& target,const vector<string>&words){
    for(auto w : words){
        if(target.find(w) != std::string::npos)
            return true;
    }
    return false;
}

double OPT_Adam::UpdateTensorParam(hGensor hP,size_t offset,floatX *gX,float clip){ 
    // assert(gimap.find(hP)!=gimap.end());
    float alpha=adam.alpha,beta1=adam.beta1,beta2=adam.beta2,eps=adam.eps;
    auto& im = _fish->GetGensorInfo(hP); //gimap[hP];
    float *m=im.gm; //==nullptr?nullptr:(float*)(im.gm->data);
    float *v=im.gv; //==nullptr?nullptr:(float*)(im.gv->data);        //first&second moment
     
    const int64_t ne = tELEM(hP);
    floatX *paramX = (floatX*)(hP->data),*paramX0,*gX0;
    float mh,vh,g0,x,x0,x00;
#ifdef _TENSOR_CUD_
    paramX = (floatX*)_tmp;     gX=paramX+hP->szData;
    paramX0 = paramX,           gX0 = gX;
    hP->SerialGP(paramX,gX,true);           x00 = T2Float(paramX);     
    if(DEBUG.train_hyperparams==1){
        clip = 1.0e-6;        
    }else{
        clip = gClip(ne,gX,hP); 
    }
        
#else
    if(gX==nullptr){       
        gX = fGrad(hP);
        clip = gClip(ne,gX,hP);        
    }else{

    }
    assert(hP->type==GGML_TYPE_F32);
#endif    
    
    GD_METHOD tpCurGD = tpGD;
    if(tpGD==SGD_HYBRID){
        tpCurGD = im.isAdam ? ADAMw : SGD;
    }
    switch(tpCurGD){
    case SGD:       //  converge very slow  !!!
        for (int64_t j = 0; j < ne; ++j,v++,gX++,paramX++) {
            g0 = *gX*clip;        x = T2Float(paramX);  
            *paramX = x-alpha*(g0);
        }  
        break;
    case SGD_v:     // why v is so important?
        for (int64_t j = 0; j < ne; ++j,v++,gX++,paramX++) {
            g0 = *gX*clip;        x = T2Float(paramX);  
            *v = *v*beta2 + g0*g0*(1.0f - beta2);
            vh = sqrtf(*v*beta2h) + eps;
            *paramX = x-alpha*(g0)/vh;               //  beta1h = learning rate
        }  
        break;    
    case SGD_blk_v:   {  // why v is so important?
        double vb = *v*beta2+(g_ll/ne)*(1.0f - beta2);      // *v = *v*beta2 + g0*g0*(1.0f - beta2);
        *v=vb;      v++;
        vh = sqrtf(vb*beta2h) + eps;
        for (int64_t j = 0; j < ne; ++j,gX++,paramX++) {
            g0 = *gX*clip;      x = T2Float(paramX);  
            *paramX = x-alpha*(g0)/vh;               //  beta1h = learning rate
        }  
        }
        break;
    case ADAMw:        
    default:
        for (int64_t j = 0; j < ne; ++j,m++,v++,gX++,paramX++) {
            float g=T2Float(gX);
            x0 = x  = T2Float(paramX);  
            // float g0 = isZMUV ? *gX : *gX*clip;
            g0 = g*clip;        
            *m = *m*beta1 +    g0*(1.0f - beta1);   //linear interpolations of the gradients 
            *v = *v*beta2 + g0*g0*(1.0f - beta2);   //linear interpolations of their variances
            mh = *m*beta1h;     //debiasing 
            vh = *v*beta2h;
            vh = sqrtf(vh) + eps;
            x  = x*(1.0f - p_decay) - alpha*mh/vh;        //ormalize step(mh) by its standard deviation(vh)                     
            *paramX = x; //ggml_set_f32_1d(hP, j, x);
            //  !!!update our low precision version of the parameters using stochastic rounding!!!
        }    
        break;
    case ADAMw_cuda:{
            //float grad = grad_scale * (float)grads_memory[idx];
            //float m = m_memory[idx],v = v_memory[idx];
            // m = lerp(grad, m, beta1);                   m_memory[idx] = m;
            // v = lerp(grad * grad, v, beta2);            v_memory[idx] = v;
            // m /= beta1_correction;  // m_hat
            // v /= beta2_correction;  // v_hat
            // fetch the old value of this parameter as a float, from either source
            // float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];
            // update this parameter
            // float param = old_param - (learning_rate * (m / (sqrtf(v) + eps) + weight_decay * old_param));
            // stochastic_rounding(param, &params_memory[idx], seed);
    }
        break;
    
    } 
#ifdef _TENSOR_CUD_
    x = T2Float(paramX0);               g0 = x-x00;
    hP->SerialGP(paramX0,gX0,false);
    // hP->SerialGP(paramX0,gX0,true);    x  = T2Float(paramX0);  
#endif    
    return 0.0;
}

static string GD_NAME[]={
    "ADAMw","SGD","SGD_v","SGD_blk_v","SGD_HYBRID",   
};

void Optimizer::UpdateTrainLoss(int x,float loss,int flag){
    int step = iter;
    if(step>1){
        float last = scheduler->Last();
        isStopImprove = step>0 ? (loss>last*1.1) : false;   
        if(isStopImprove){
            // _INFO("%s_%d: StopImprove\n", __func__, iter);
        }       
        fx_prev[0] = loss;                
    }else if(step==0){
        fx_prev.push_back(loss);        fx_best.push_back(loss);   //   only 1 element  ???
        loss_before = loss;             
    }
    loss_after  = loss;
    scheduler->Append(loss);

    isConverge = false; /*fabsf(loss_after - fx_prev[0])/loss_after < train_params.adam.eps_loss*/
    // check for improvement
        /*if (train_params.opt_max_no_improvement > 0) {
            if (fx_best[0] > fx) {
                fx_best[0] = fx;
                n_no_improvement = 0;
            } else {
                ++n_no_improvement;

                if (n_no_improvement >= train_params.opt_max_no_improvement) {
                    result = OK;
                }
            }
        } */       
        // fx_prev[0] = fx;
}

int RAW_update(std::vector<hGTensor>& tensors,ADAM_params_ adam, float learning_rate,float& grad_norm, int t,int alg, int flag);
Optimizer::RESULT Optimizer::Search(struct ggml_context * ctx, hGensor loss_,hGensor target_,CLI_params& hparams)    {
    hEDS = _fish->hEDS;              assert(hEDS!=nullptr);
    auto train_params = TrainParams();    
    
    last_time = ggml_time_ms();
    Optimizer::RESULT result = DID_NOT_CONVERGE;   
    bool cancel = false;      
    string suf,pref;
    
    _INFO("\nOptimizer::%s@<%s> %s device=[%s] \n", __func__,_fish->hBackTG->name.c_str(),
        _fish->isLoadCheckpoint?hparams.save.checkpoint_in.c_str():"",
        hEDS->__repr__(suf,pref,0).c_str());
    _INFO("\t Accumulation=%d AdaptiveSched=%d GRAP=%p rZMUV=%g rLARS=%g \n",nGradAccum,(int)isAdaptiveSched,grad,hparams.ZMUV_ratio,hparams.lars_ratio );
        // tpGD=SGD_HYBRID;    //ADAMw      SGD_v    SGD_HYBRID        SGD_blk_v
    _INFO("\tDECENT=%d(%s) SIGN=%d tpFuseCu=%d\n\n",tpGD,GD_NAME[tpGD].c_str(),tpSign,tpFuseCu);
    DEBUG.Dump(0);

    float a=0, val_loss=0,grad_norm=0;    
    if( !BatchGrad(0,a,0x0) )       //  warmup
        return CANCEL;    

    if (just_initialized) {
        n_no_improvement = 0;
        just_initialized = false;
    }
    // g_dump_level = 0;
    int iter0 = 0;  //opt->iter;
    for (int t = 0; t < train_params.adam.n_iter; ++t) {
        iter = iter0 + t + 1;        
        // const int64_t t_start_wall = ggml_time_us(),t_start_cpu = ggml_cycles();
        float lr = train_params.LearningRate();
#ifdef _TENSOR_CUD_
        lr = scheduler->LearningRate(iter);
#else
        lr = scheduler->LearningRate(iter);
        // lr = learning_schedule(iter, _params.warmup, _params.cos_decay_steps, _params.adam.alpha, _params.adam.min_alpha,_params.cos_decay_min, _params.cos_decay_restart, _params.enable_restart);
#endif
#ifdef _TENSOR_CUD_
        RAW_update(opt_ps,train_params.adam,lr, grad_norm, iter,1,0);
        last_lr = lr;       g_step = grad_norm; //sqrt(grad_norm*grad_norm/nParams);
#else         
        SignStochastic(nParams,hparams);    
        UpdateParams(nParams,hparams,0x0);     
#endif   
        UpdateLossCurve(0x0);    
        // AdamMiniV(clip,nx,hparams,0x0);   
        //gradient is useless at this stage
        if (train_params.save_every>0 && t % train_params.save_every == 0) {  
            _fish->SaveTrain(""); 
        }
        if (hparams.common.eval_every>0 && t % hparams.common.eval_every == 0) {
            val_loss = Evaluate(val_loader,iter);  
        }    
#ifdef _TENSOR_CUD_
#else        
        if( hparams.common.gpt_every>0 && t%hparams.common.gpt_every==0 )   {
            _fish->GenSentence(1);   
        }  
#endif          
        if( !BatchGrad(iter,a,0x0))
            return CANCEL; 
      
        if(_fish->hDistler!=nullptr) 
            _fish->hDistler->UpdateSigma(iter);
        // check convergence
        if (isConverge) {
            _INFO("[search] Converged!!!\n");                result = OK;
        }

        result = DID_NOT_CONVERGE;
    }
    return result;
}

void Fish::Train(  int flag )   {
    
}

void RAW_forward(Fish *fish,int flag);
float RAW_backward(Fish *fish,const int* hostInput,int accum_steps,bool,int flag);
double Optimizer::GraphCompute(hSampLoader hLoader,hTGraph hTG, int flag){
    // return 0.0; //only for debug
    int64_t now = ggml_time_ms();
    int nThread = TrainParams().n_threads,no=0,nAccum=TrainParams().n_gradient_accumulation;
    bool bench = false;
    string suffix,prefix;
    bool isOnlyEvaluate = !hTG->isBackward;
    float mean_loss = 0.f;
    OutCLS* cls = _fish->GetNeuron<OutCLS>("OutCLS",0);
#ifdef _TENSOR_CUD_    
    const int* hostInput = (int*)hLoader->hostBatch->data;        //(int*)train_loader->hostBatch->data;
    isBackward = false;
    RAW_forward(_fish,flag);    //0x1001
    isBackward = true;
    RAW_backward(_fish,hostInput,nAccum,isOnlyEvaluate,flag);
    mean_loss = hLoader->hTokens->LossOnResult(hLoader,cls);
    if(isOnlyEvaluate){
        return mean_loss;
    }else
        UpdateTrainLoss(-1,mean_loss);   
    /*for(auto it : _fish->gensors.infos){
        auto node = it.first;
        if(!BIT_TEST(node->flags,GTensor::F_TOX)){
            int debug=0x0;
            assert(node->data==nullptr);            
        }
    }*/
    /*isBackward = false;
    hGensor cur = _fish->Input(); 
    for(auto nn : _fish->neurons){   
        if(nn->isGang())    continue;
        if(no==5){
            int debug = 1;
        }
        _INFO("%d\tF@%s\n",no++,nn->__repr__(suffix,prefix).c_str());            
        cur = nn->Interact(nullptr,cur);        
        // cuLiteTest(GTensor::B,GTensor::T,GTensor::C);           //only for debug
    }
    isBackward = true;
    for (auto it = _fish->neurons.rbegin(); it != _fish->neurons.rend(); ++it)    {
        hNeuron nn = *it;
        if(nn->isGang())    continue;
        _INFO("%d\tB@%s\n",no,nn->__repr__(suffix,prefix).c_str());            
        cur = nn->Interact(nullptr,cur); 
    }*/
#else
    auto cgraph = hTG->raw();
    
    struct ggml_cgraph *_gf=_fish->GetForwRaw(),*_gb=_fish->GetBackRaw();
    assert(cgraph==_gf || cgraph==_gb);
    
    if(bench){ //  only for performance benchmark
        GST_TIC(t0);
        CHILD_1218_GRAD //ggml_graph_comp0(_gf,0x0); 
        // ggml_graph_comp0(_gb,0x0); 
        _INFO("gb_compute %s T=%.3g","",GST_TOC(t0));  
        exit(-666);
    }
    if(hEDS->isOnlyCPU()){
        auto *cplan = &(_fish->gb_plan);
        auto status = ggml_graph_compute(cgraph, cplan);
        return status==GGML_STATUS_SUCCESS;
    }
    
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(lctx.backend_metal)) {
        ggml_backend_metal_set_n_cb(lctx.backend_metal, n_threads);
    }
#endif
    hEDS->SetThread(nThread);

    auto status = ggml_backend_sched_graph_compute_async(hEDS->sched0, cgraph);
    if(status!=GGML_STATUS_SUCCESS)
        return false;
#endif
    tX = ggml_time_ms()-now;
    return 0.0;
}

/*
    REF:    /home/cys/Github/llama.cpp-master/examples/batched/batched.cpp
*/
float Optimizer::Evaluate(hSampLoader loader,int iter,int flag){  
    if( loader->num_batches==0 ) 
    {    assert(0);         return 0;          }
    assert(loader->num_batches>0);
    if(iter!=-666)   _INFO("[eval] " );   
    GST_TIC(tic);   
    OutCLS* cls = _fish->GetNeuron<OutCLS>("OutCLS",0);  
    cls->hLoader = loader;
    auto loss = hLoss();     
    double l2,delta_max=0,delta_=0,a,mean_loss=0,ee;
    auto tokens_input = _fish->Input( ); 
    int i,nB=0,step=max(loader->num_batches/10,1);      //smaple to reduce eval time
    size_t nz=0,j;
    llama_token tMost = (llama_token) (_fish->nClass() - 1);
    hSAMP samp = nullptr;
    const float *wLog = nullptr;
    loader->next_sample = 0;        // fix this to keep same acc on each experiment
    for (i = 0; i < loader->num_batches; i+=step) {        
        if(tokens_input!=nullptr)   {//in some debug mode, tokens_input maybe null
            loader->UpdateBatch(i,_fish);
            samp = loader->cur_samps[i];
        } 
        if(wLog!=nullptr)    {
            for (l2 = 0,j = 0; j < nz; j++    ) {
                a = wLog[j];            l2 += a*a;              
            }
            l2 = sqrt(l2)/nz;            
            delta_max = max(delta_max,l2);      delta_+=l2;        
        }
        mean_loss += GraphCompute(loader,_fish->hForwTG);
        nB++;
        if(_fish->hparams.isOnlyGPT){
            return ee;
        }     
#ifdef _TENSOR_CUD_
#else   
        a = ((float*)hPreLogits()->data)[0];        //  -6.60046101     -4.3040733  
        ee=loader->DecodeVerify(samp,tokens_input,_fish->preLogits);
        mean_loss += loss==nullptr ? 0 : ((float*)(loss->data))[0];         //float *fLoss = (float*)(loss->data)
        break;
#endif              
    }
    mean_loss /= nB;
    if(iter==-666)    {   //hack
        return i;
    }

    float last = lcEval.Last();  //[eval]   Loss@Evaluation=7.302641 T=0.232s ======
    lcEval.Add(mean_loss);
    float delta=last-mean_loss,best=lcEval.Best();   
    bool isOverfit = delta<0 && abs(mean_loss-best)>best/10;       
    // if(isOverfit)   {
    //     _INFO(" !OVERFIT! ");
    // }
    _INFO(" Loss@\"%s\"=%.3f(%.2g) best=%.4f(eval_%d) E2T=%.3g T=%gs x=%.3g\n",loader->sTokenSet().c_str(), mean_loss,delta,best,lcEval.best_id,
        mean_loss-lcTrain.Last(),GST_TOC(tic),ee );

    if(wLog==nullptr) {     }        
    else
        _INFO("\t Loss@Evaluation=%f delta=%g(%.5g) T=%gs\n", mean_loss,delta_max,delta_/nB,GST_TOC(tic) );
    string sX="_loss="+std::to_string(mean_loss);    
    // if(delta>0)     
    //     _fish->SaveTrain(sX);  

    return mean_loss;
}

float cosine_decay(int64_t step, int64_t decay_steps, float minimum) {
    if (step > decay_steps) {
        step = decay_steps;
    }
    const float cosine_decay = 0.50f*(1.0f + cosf(3.14159265359f*step/decay_steps));
    const float decay = (1 - minimum)*cosine_decay + minimum;
    return decay;
}

float cosine_decay_restart(int64_t step, int64_t decay_steps, float minimum, float restart_step_mult) {
    while (step > decay_steps) {
        step -= decay_steps;
        decay_steps = (int64_t) (restart_step_mult * decay_steps);
    }
    return cosine_decay(step, decay_steps, minimum);
}

float learning_schedule(
    int64_t step,
    int64_t warmup_steps,
    int64_t cos_decay_steps,
    float   learning_rate,
    float   overall_minimum,
    float   cos_decay_minimum,
    float   cos_decay_restart_step_mult,
    bool    enable_restart) {

    float result =
        (step < warmup_steps)
            ? (float) step / (float) warmup_steps
            : enable_restart
                ? cosine_decay_restart(
                    step - warmup_steps,
                    cos_decay_steps,
                    cos_decay_minimum,
                    cos_decay_restart_step_mult)
                : cosine_decay(
                    step,
                    cos_decay_steps,
                    cos_decay_minimum);

    float min = overall_minimum / learning_rate;
    result = min + result * (1.0f - min);
    return result;
}

float Optimizer::UpdateLossCurve(int flag){
    struct train_params_ _params = TrainParams();   
    int n_batch = _params.n_batch,n_ctx = _params.n_ctx;
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
        const int n_iter = _params.adam.n_iter,done_iter = iter - first_iter,remaining_iter = n_iter - done_iter;
        remaining_millis = remaining_iter * millis_per_iter;
    }        

    last_time = ggml_time_ms();
    
    int impr_plot = -(int)(1 + (loss_before - loss_after) * 10.0f + 0.5f);
    if (impr_plot > 0)
        impr_plot = 0;
    if (std::isnan(loss_before) || std::isnan(loss_after))
        impr_plot = 0;
    lcTrain.Add(loss_after);
    if((iter-1)%_params.dump_every==0 || isDumpOnce){
        isDumpOnce = false;
        _INFO("[train]_%-6d loss=%f |g|=%g\tlr=%.2e | %s ",iter,loss_after,g_step,last_lr,train_loader->IterInfo().c_str()); //,zmuv_0,zmuv_1    
        if (millis_per_iter > 0)            {
            _TIME_INFO(" dt=",millis_per_iter);  _TIME_INFO(" tData=",tData);    _TIME_INFO(" tX=",tX);        _TIME_INFO(" eta=",remaining_millis);
        }
        size_t tokens_processed = _fish->hparams.nTokensPerGrad();   //(size_t) * B * T * grad_accum_steps;
        float tokens_per_second = tokens_processed / millis_per_iter * 1000.0f;
        ema_tps = iter==1 ? tokens_per_second : 0.95f * ema_tps + 0.05f * tokens_per_second;
        _INFO(" | %.1fK token/s",ema_tps/1000.0);     _INFO("\n");
    }
    float improvement = loss_before - loss_after;
    return improvement;
}
// TODO 
bool Optimizer::AfterLoadBatch(int accum_step, int flag)    {
    // LossCurve(0x0);
    auto _params = TrainParams();   
    int n_batch = _params.n_batch,n_ctx = _params.n_ctx;

    if (accum_step == 0)        {
        // *sched = UpdateSchedule(flag);
    }
    if(train_loader->isNextEpoch(train_epochs)) ++train_epochs;
    /*if (train_loader->next_sample >=train_loader->shuffle_sample_count)    {
        ++train_epochs;
        _INFO("%s: reshuffle samples. completed epochs: %llu\n", __func__, train_epochs);
        // note: we may have used some samples from the current shuffling more than once
        train_loader->shuffle_rng_state_current =train_loader->shuffle_rng_state_next;
        // train->shuffle_rng_state_next = shuffle_samples(
        //     train->shuffle_rng_state_current,data->shuffled_samples_offs,data->shuffled_samples_begin,
        //     data->shuffled_samples_size,data->samples_begin,data->samples_size,data->samples_count);
        
        train_loader->Shuffle();           //SAMP_0816
        // train->shuffle_rng_state_next = shuffle_samples(
        //     train->shuffle_rng_state_current,loader->shuffled_samples_offs.data(),
        //     loader->shuffled_samples_begin.data(),loader->shuffled_samples_size.data(),
        //     loader->samp_begin.data(),loader->samp_size.data(),loader->samp_size.size());
        
        train_loader->next_sample = 0;
    }*/

    const bool last_epoch_reached = (_params.n_epochs > 0 && train_epochs - first_epoch >= _params.n_epochs);
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

void Optimizer::BeforeTrain(struct train_params_& train_params,hGensor tokens_,int flag) {    
    first_iter             = iter;
    // train_loader->hOPT = this;           val_loader->hOPT = this;
    // assert(tokens_!=nullptr);
    // tokens_input = tokens_;
    train_params.adam.n_parameters = nParams;    
#ifdef _TENSOR_CUD_
    train_params.adam.decay = 0.1;      //very high for GPT2 Model
#endif
    scheduler = std::make_shared<DiscreteSchedule>(train_params);
    opt_ps = _fish->optParams;
    nMostParam = 0;
    size_t offset = 0x0;
    for (auto ps : opt_ps) {            
        nMostParam += tELEM(ps);
#ifdef _TENSOR_CUD_
        ps->offset = offset;
        offset += ps->nByte();
#endif
    } 
    // assert(nMostParam>=nParams);
    assert(train_params.adam.n_iter>0);
    // max_epoch = train_params.adam.n_iter
}

size_t TGraph::Prepare4Train(struct ggml_context *ctx_,GD_METHOD tpGD,int flag){
    hOptimizer hOpt = hFish->hOPT;          assert(hOpt!=nullptr);
    size_t nP=0,nz=0,nzAll=0,id=0;
    for(auto& gi : hFish->gensors.infos){
        auto gensor = gi.first;
        int nParam = (int)tELEM(gensor);        
        if(strcmp(gensor->name,"output.weight")==0){
            int xxx = 0;
        }
        // auto& im = gi.second;
        id++;
        if(!(gensor->flags & GGML_TENSOR_FLAG_PARAM) )
            continue;
        nzAll += nParam;
        if(tpGD == GD_METHOD::SGD_HYBRID){
            gi.second.isAdam = isStrMatch(gensor->name,hOpt->adam_filter);
        }
            
        if(!gi.second.isAdam)
            continue;
        
        nP++;       nz+=nParam;
        gi.second.gm = new float[nParam]();
        gi.second.gv = new float[nParam]();
        gi.second.gpf = new float[nParam]();          
    }
    
    _INFO("[TGraph::%s] AdamTensor=(%d,%g) filter={",__func__,nP,nz*1.0/nzAll);
    for(auto f : hOpt->adam_filter){
        _INFO("\"%s\" ",f.c_str());
    }
    _INFO("}\n");
    
    return nz;
}

void Optimizer::Prepare(size_t nx_,int flag){
    // assert(nx_>=0);
    iter = 0;
    nParams = nx_;         
    just_initialized = true;
    double s=0;
    size_t nz = 0;
    NLP_AutoRegressive *dolphin = dynamic_cast<NLP_AutoRegressive*>(_fish);
    if(nParams>0)   {
        struct ggml_init_params ctx_opt_params;    
        ctx_opt_params.mem_size = GGML_MEM_ALIGN*3 + ggml_tensor_overhead()*3 + ggml_type_size(GGML_TYPE_F32)*nParams*3;
        if (past > 0) {
            ctx_opt_params.mem_size += GGML_MEM_ALIGN + ggml_tensor_overhead() + ggml_type_size(GGML_TYPE_F32)*past;
        }    
        ctx_opt_params.mem_buffer = NULL;
        ctx_opt_params.no_alloc   = false;
        _ctx = ggml_init(ctx_opt_params);    
        if(_fish->isTrain()) {
            if(isGlobalGrad){
                assert(nParams<INT_MAX);
                grad = TENSO(_ctx, GGML_TYPE_F32, {(int)(nParams)});
            }
                
            nz = _fish->hBackTG->Prepare4Train(_ctx,tpGD);
            s = nz*1.0/nParams;
            // gimap = _fish->hBackTG->gimap;
        }
        // train_loader->Init(dolphin,"Train",false);     //train_loader->Prepare(this);
    }
#ifdef _TENSOR_CUD_
    _tmp = new float[nParams]();
#else
#endif    
    // val_loader->Init(dolphin,"Eval",false);            //val_loader->Prepare(this);
}

bool Optimizer::PrepareData( CLI_params& hparams,int flag )   {   
    GST_TIC(tic);   
    train_loader->Prepare(_fish->tsTrain);                
    val_loader->Prepare(_fish->tsEval);
    bool isLoadOK = false;  
    string root=_fish->tsTrain->serial_root,spTrain = root+".train",spEval = root+".eval";
    if(root.empty()){ 
        train_loader->Shuffle();
        return true;
    }
    // auto& tokens = hTokenset->tokens;
    
    if(1)   {
        if( train_loader->Serialize(spTrain,false) 
            && val_loader->Serialize(spEval,false)){
                if(train_loader->len()>0){
                    // hDict->nUniqueToken = train_loader->n_unique_tokens; 
                    // _INFO("%s: nTrain=%zu nEval=%zu tpBatchSample=%s T=%.3g\n", __func__, train_loader->len(),val_loader->len(),GST_TOC(tic));
                    isLoadOK = true;
                }
        }            
    }
    if(!isLoadOK) {
        hDataToken hTokenset=train_loader->hTokens;
        assert(hTokenset!=nullptr);
           
        std::vector<size_t> samples_begin,samples_size;
        // auto train_params = hparams.common;
        size_t nUnique = hTokenset->nUnique,nVocab=hTokenset->nVocab;
        // int n_ctx_tokens = hparams.n_ctx;
        if( hTokenset->InitSamps(hparams.common.n_ctx,samples_begin,samples_size)){

        }else{
            _INFO("%s: NULL Samps!!!    tpBatchSample=%s nTrain=%zu nEval=%zu T=%.3g\n", __func__, train_loader->tpBatchSample.c_str(),
                train_loader->len(),val_loader->len(),GST_TOC(tic));      
            return false;
        }        
  
        train_loader->SetSamples(samples_begin,samples_size,true,hparams);    
        val_loader->SetSamples(samples_begin,samples_size,false,hparams);
        
        // assert(val_loader->n_unique_tokens <= nUnique && train_loader->n_unique_tokens <= nUnique);
        // val_loader->n_unique_tokens = nUnique;
        // train_loader->n_unique_tokens = nUnique;
        shuffle_samples_hash = train_loader->shuffle_samples_hash;
        train_loader->Serialize(spTrain,true);
        // train_loader->Serialize(spTrain,false);      //only for debug
        val_loader->Serialize(spEval,true);
    }
    
    // assert(train_samples_begin.size() == train_samples_size.size());
    _INFO("%s: tpBatchSample=%s nTrain=%zu nEval=%zu T=%.3g\n", __func__, train_loader->tpBatchSample.c_str(),
        train_loader->len(),val_loader->len(),GST_TOC(tic));        
    return true;
}

void OPT_Adam::Prepare(size_t nx_,int flag){
    Optimizer::Prepare(nx_,flag);
    if(_fish->isTrain()){
        if( grad!=nullptr ){

        }else{

        }
        /*gm  = TENSO(_ctx, GGML_TYPE_F32, nParams);
        gv  = TENSO(_ctx, GGML_TYPE_F32, nParams);
        gpf = past > 0 ? TENSO(_ctx, GGML_TYPE_F32, past) : NULL;
        float * v  = (float *)gv->data;   // gradients
        float * m  = (float *)gm->data;     // first moment
        ZERO_(gm);
        ZERO_(gv);
        if (gpf) {
            ZERO_(gpf);
        }*/
       /*_fish->hBackTG->Prepare(_ctx);
        */
    }
}

OPT_Adam::OPT_Adam(NLP_AutoRegressive *g_,CLI_params& params_,int flag)
    : Optimizer(g_,params_,flag)    {    
    auto train_params = TrainParams();
    //  0.9f, 0.95f, 1e-8f      decay=0.1
    adam = train_params.adam;
    // sched              = 1.0f;
}

void OPT_Adam::Dump(int typ){
    Optimizer::Dump(typ);
    // if(NOT_DUMP())  return;
     _INFO("[OPT_Adam]\tsRESI=%g",TrainParams().residual_scale);
    adam.Dump(typ);
    
}