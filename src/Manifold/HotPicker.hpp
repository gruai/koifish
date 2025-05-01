/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT 
 *
 *  \brief Train a model to predict hot element
 *  \author Yingshi Chen
 */
#include "../g_float.hpp"
#ifdef _USE_GBDT_
    #include "../GBDT/tree/GBRT.hpp"
    #include "../GBDT/data_fold/DataFold.hpp"
#endif

class SparseNeuron;
class HotPicker : public CS_Picker{
protected:
    string name;
    std::vector<hGTensor> arrX,arrY;
#ifdef _USE_GBDT_
    LiteBOM_Config config;    
    hTabularData hTrainData = nullptr;
    shared_ptr<Grusoft::GBRT> hGBRT = nullptr;
#endif
    SparseNeuron *neuron;

public:
    HotPicker(SparseNeuron *n,int flag=0x0);
    virtual int Train(int flag=0x0);
    virtual int Eval(int flag=0x0);
    virtual int Predict(int nPoint,floatI *data,int *hot,int flag=0x0);

    virtual bool SerialModel(const std::string&sPath,bool isSave=false,int flag=0x0);
    virtual string __repr__( string& suffix,string& prefix,int flag);
friend class SparseNeuron;
};
typedef shared_ptr<HotPicker> hHotPicker;

