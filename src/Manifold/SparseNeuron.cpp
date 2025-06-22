/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *  
 *  \brief Neurons with sparsing activation/weight
 *  \author Yingshi Chen
 */
#include <memory>
#include "Fish.hpp"
#include "Optimizer.hpp"
#include "../g_stddef.hpp"
#include "HotPicker.hpp"
#ifdef _USE_GBDT_
    #include "../GBDT/data_fold/Histogram.hpp"
    using namespace Grusoft;
#endif
#include "../lenda/kernel/SVD.hpp"
#include "../Utils/GST_rander.hpp"

CS_Picker::CS_Picker(hFISH hFish,int flag){
    int nEmbed=hFish->config.nEmbed();      //
    dim=hFish->config.n_ff();
    hot=new int[dim*2]();
	for(int i=0;i<dim;i++)	hot[i] = 1;
	dTemp = new float[dim+nEmbed*dim];   
    
    T_hot = 0.2;        T_zero=1.0e-3;
}

//  Picker should much fast than dot!
double CS_Picker::tPick =0.0;

int CS_Picker::Update(int level,float *hb,int flag){    
    return -1;    
    double t0 = GST_us();
    if(level>0)
        return nLastHot;
	int nz=0,i=0,nHot=std::max((int)(dim*T_hot),16),nEx=nHot,id;
    nEx = ((nHot*2)/16)*16-nHot;
    // nEx = ((nHot)/16)*16-nHot;
	float *tmp= dTemp,prev=FLT_MAX,a;
    int *map = hot+dim;
    // for(i=0; i<dim; i++)    hot[i]=1;   return dim;

	for(i=0; i<dim; i++){
		if(hot[i]==0){
			// assert(hb[i]==0);	continue;
		}
        hot[i] = 0;
		if(fabs(hb[i])<T_zero)		continue;		
        // if(hb[i]<T_zero)		continue;		
		map[nz] = i;        tmp[nz++] = fabs(hb[i]);      
	}
    if(nz<nHot)
        return -1;
#ifdef _USE_GBDT_
    vector<tpSAMP_ID> idx;
	sort_indexes(nz,tmp,idx);
    for(i=0;i<nHot;i++){
        id = map[idx[nz-1-i]];   
        hot[id] = 1;
        a = fabs(hb[id]);     assert(prev>=a);
        prev = a;
    }
    i=0;
	while(i<nEx)	{
		id = rand()%dim;	
        if(hot[id]==0){
            hot[id] = 1;    nHot++;     i++;
        }
	}
    assert(nHot % 16==0);
    
    nLastHot = nHot;
    if(isMerge){
        for(nHot=0,i=0;i<dim;i++){
            if(hot[i]==0)       continue;
            hot[nHot++] = i;
        }
        assert(nHot==nLastHot);        
    }
#endif
    tPick += GST_us()-t0;
    return nHot;
}

HotPicker::HotPicker(SparseNeuron *n,int flag){
    name = n->name;
    // config.num_trees = 256;    
}

string HotPicker::__repr__( string& suffix,string& prefix,int flag){
    char buf[5012]="\0";
    const char*tab=prefix.c_str();    
    sprintf(buf+strlen(buf),"sparse_%s","GBDT");    
    if(flag>0)
        _INFO("%s",buf);     
    return buf;  
}

int HotPicker::Predict(int nPoint,floatI *data,int *hot,int flag){
    return 0x0;
}

bool HotPicker::SerialModel(const std::string&sPath,bool isSave,int flag){
    return false;
}

int HotPicker::Train(int flag){
#ifdef _USE_GBDT_
    string title=name+"_GBDT";
    ExploreDA *edaX = new ExploreDA(config,title, flag);
    hTrainData = std::make_shared<FeatsOnFold>(config, edaX, title, flag) ;       //  from X,Y
    size_t nSamp_ = arrX.size();
    hTrainData->InitMost(nSamp_);
    if(hTrainData==nullptr)
        return -1;

    int nTree = config.num_trees;
    hGBRT = std::make_shared<GBRT>(hTrainData.get(),nullptr, 0.333, BoostingForest::CLASIFY, nTree);
    hGBRT->Train("",0,0);
    SerialModel("",true);
#endif
    return 0x0;
}
int HotPicker::Eval(int flag){
    return 0x0;
}

SparseNeuron::SparseNeuron(const std::string &key_,JSON::const_iterator jit, Fish *hG_, int flag)
    : GeNeuron(key_,jit, hG_, flag){
    if(BIT_TEST(flag,F_HOTPICK)){
        isSparse = true;
    }
    if(isSparse)    {
        method = hG_->config.model.sparse.method;
        hPicker = std::make_shared<HotPicker>(this);
    }
}

void SparseNeuron::SetEmbed(TokenEmbed* embd_,int type,int flag){
    assert(embd_!=nullptr);
    subw = embd_;
    samp_type = type;
    if(samp_type==0){
        w->SetRefer(embd_->w);  
    }else{
        w->SetRefer(embd_->wInv);  
    }
    // SetRefer(embd_);
}

// TODO: Weighted sampling
void SparseNeuron::UpdateSamps(int seed,int flag){
    assert(hSamps!=nullptr);
    float *weight = nullptr;    // TODO: Weighted sampling
    int nVocab = hFish->nClass(),nSample=hSamps->size();
    // int *samps=new int[nSample];
    // samp_1 = nVocab;    
    // hSamps->SerialGP(samps,nullptr,sizeof(int)*nSample,false);  //29156,    22663,      34659
    // delete[] samps;
    std::vector<int> samps;
    Grusoft::GRander rander(seed);
    if(1){  // nearly same
        hSampLoader sloader = hFish->GetOptimizer()->train_loader;
        assert(sloader!=nullptr);
        sloader->PickSomeTokens(rander,nSample,samps);
    }else{
        
        samps = rander.kSampleInN(nSample, nVocab);
    }
    assert(samps.size()==nSample);
    hSamps->SerialGP(samps.data(),nullptr,sizeof(int)*nSample,false);  


}

/*
    EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation
*/
bool SparseNeuron::InitSVD(int flag){
    assert(hSVD==nullptr);
    int nIn=w->ne[1],nOut=w->ne[0],rank=min(256,min(nIn,nOut)/5);
    rank = (int)(rank/16)*16;       assert(rank>=16);
    size_t i,nz=w->size();      assert(nz==nIn*nOut);
    float *A=new float[nIn*nOut],tol_=0;    //1.0e-3
    f8e5m2_t *src = (f8e5m2_t*)(w->data);
    for(i=0;i<nz;i++)   
        A[i] = T2Float(src+i);  //fp8_to_float(src[i]);
    hSVD=std::make_shared<LoSVD<float>>(name,A,nIn,nOut,rank,tol_,typNUMBER::F32); 
    if(!hSVD->Build( ))  {
        compression = SKIP;
    }else{
        if(compression==SVD_a)  {   //keep same graph
            float *approx = hSVD->Approx( );
        }else{  
                               
        }      

    }
    delete[] A;
    return true;
}

bool SparseNeuron::Sparsing(int flag) {
    if(hPicker==nullptr)    return false;
    int iRet = hPicker->Train(flag);
    return iRet;
};

bool SparseNeuron::GetHotIndex(int nPoint,floatI *data,int *hot,int flag){
    if(hPicker==nullptr)    return false;
    hPicker->Predict(nPoint,data,hot);
    return true;
}

bool SparseNeuron::OnData(hGTensor X,hGTensor Y,int *hot,int flag){
    if(hPicker==nullptr)    return false;
    if(method==1)   {
        hPicker->arrX.push_back(X);
        hPicker->arrY.push_back(Y);        
    }else if(method==-1){
        int i,dim=Y->shape[0];
    }

    return true;
}

void Fish::Sparsing(int flag){
    for(auto neuron : neurons){
        if(!neuron->isSparse)   continue;
        neuron->Sparsing(flag);
    }
}

