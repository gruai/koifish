/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Collect some features of neuron
 *      1) For decision model
 *      2) Train a model to predict hot element
 *
 *  \brief
 *  \author Yingshi Chen
 */
#include "../g_float.hpp"
#ifdef __USE_GBDT__
#include "../GBDT/data_fold/DataFold.hpp"
#include "../GBDT/tree/GBRT.hpp"
#endif

class SparseNeuron;
class Neuron;
class GTensor;
class Grusoft::FeatsOnFold;

// Collect some features of neuron
class FeatPicker {
   protected:
    string name, sKey;
    string sFeatDesc;
    bool isMerge = false;
    int dimX = -1, nLastHot = -1;
    int dimY      = 1;  // number of targets
    int nPickSamp = 0, nMostSamp = 0;
    float T_hot = 0.2, T_zero = 1.0e-3;
    // row-major order: (3, 2) means "3 samps in a row(feature), 2 features"
    shared_ptr<GTensor> raw_feat;
    virtual bool CleanFeats(int flag=0x0);    //shrink, checkNA
    std::vector<shared_ptr<GTensor>> arrX, arrY;
    //only 1 train/validation/test
    std::shared_ptr<Grusoft::FeatsOnFold> hTrainData = nullptr, hValData = nullptr, hTestData = nullptr;
    
#ifdef __USE_GBDT__
    LiteBOM_Config config;
    shared_ptr<Grusoft::GBRT> hGBRT = nullptr;
#endif

    Neuron* neuron = nullptr;

   public:
    int* hot     = nullptr;
    float* dTemp = nullptr;
    static double tPick;  //  Picker should much fast than dot!
    FeatPicker(const std::string& , int dimX_, int dimY_ = 1, int flag = 0x0);
    FeatPicker(shared_ptr<Fish> hFish, int flag = 0x0);
    //  uses the first layer’s attention output to predict the sparsity pattern for the entire model
    virtual ~FeatPicker() {
        FREE_a(hot);
        FREE_a(dTemp);
    }

    bool BeforeCollate(hDataToken coral, int flag=0x0);

    // dimX(number of feats)+dimY(number of targets)
    int dim(int flag = 0x0) { return dimX + dimY; }
    virtual bool empty(int flag = 0x0) const { return nPickSamp <= 0; }
    virtual std::shared_ptr<Grusoft::FeatsOnFold> Coral2Tabular(hDataToken coral, hFISH backbone, MODEL_ARCH model, int flagD, int flag = 0x0);
    int Update(int level, float* hb, int flag = 0x0);
    virtual int Train(int flag = 0x0);
    virtual int Eval(int flag = 0x0) { NOT_IMPLEMENTED; }
    virtual int Predict(int nPoint, floatI* data, int* hot, int flag = 0x0) { NOT_IMPLEMENTED; }
    virtual bool SerialFeatures(Fish* hFish, const std::filesystem::path& path, bool isSave = false, int flag = 0x0);
    virtual string __repr__(string& suffix, string& prefix, int flag) { NOT_IMPLEMENTED; }
    friend class Fish;
    friend class Maisui;
    friend class GeNeuron;
    friend class SparseNeuron;
};
typedef shared_ptr<FeatPicker> hFeatPicker;

class HotPicker : public FeatPicker {
   protected:
   public:
    HotPicker(SparseNeuron* n, int flag = 0x0);
    int Train(int flag = 0x0) override;
    int Eval(int flag = 0x0) override;
    int Predict(int nPoint, floatI* data, int* hot, int flag = 0x0) override;

    virtual bool SerialModel(const std::string& sPath, bool isSave = false, int flag = 0x0);
    string __repr__(string& suffix, string& prefix, int flag) override;
    friend class SparseNeuron;
};
typedef shared_ptr<HotPicker> hHotPicker;
