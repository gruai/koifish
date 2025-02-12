#include "Distillation.hpp"
#include "../Manifold/Fish.hpp"
#include "../Manifold/Optimizer.hpp"
void Distillation::UpdateSigma( int step,int flag){
    if(alg!=SIGMA)
        return;

    float delta = 1.0/100.0/2;   //scheduler->GetSigma(step);
    if(step>0)
        alpha -= delta;      
    if(hFish->hOPT->isStopImproving()){
        // alpha = scheduler->Last()+delta;      hard to converge to zero
        alpha = scheduler->Last();   
    }     
    if(alpha<0)     {
        _WARN("%s Invalid alpha=%g!\n",__func__,alpha);
        alpha=0.0;
    }
    if(alpha>1.0)     {
        _WARN("%s Invalid alpha=%g!\n",__func__,alpha);
        alpha=1.0;
    }
   
    
    float abc[3] = {1,alpha,1-alpha};   //gensor
    switch(alg){
    case SIGMA:
        _INFO("Distillation::%s: sigma=[%g,%g]\n", __func__, alpha,1-alpha);
        for(auto gensor :gensors){
            memcpy(gensor->op_params, abc, sizeof(abc)); 
        }
        break;
    default:
        break;
    }
    scheduler->Append(alpha);
    return;
}