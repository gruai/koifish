/**
 *  SPDX-FileCopyrightText: 2024-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#pragma once

#include <math.h>
#include <stddef.h>

#include <algorithm>
#include <vector>

#include "../CLI_params.hpp"
#include "../Utils/GST_obj.hpp"
#include "../Utils/GST_rander.hpp"
#include "../Utils/GST_util.hpp"

// #include "./GVMAT.h"

class GeQuant;
class GeNeuron;
class GTensor;
class Fish;
typedef shared_ptr<GeQuant> hQUANT;
namespace Grusoft {
class Distribution;
class FeatVector;
class FeatsOnFold;
};  // namespace Grusoft

class GeQuant : public std::enable_shared_from_this<GeQuant>  {
   protected:
    struct _q_sweep {
        NORMAL_MODE normal = NORMAL_MODE::NO_NORMAL;
        int bits           = 4;
        int mi             = 0;  // min impurity on GBDT

        _q_sweep() {}
        _q_sweep(NORMAL_MODE n, int b, int mi_ = 0) : normal(n), bits(b), mi(mi_) {}
        virtual string ToString(int format = 0x0) {
            string info = std::to_string(bits) + (normal != 0 ? "_normal" : "") + (mi != 0 ? "_mini" : "");
            return info;
        }

        size_t szQuant = 0x0;
    };
    _q_sweep best_;
    int group_size = 0;

    virtual void Flattern() {}
    //  1.  key embeddings - In initial layers, no significant outliers are observed. However, in the deeper layers, few channels (approximately four) exhibit
    //  visibly larger magnitudes (outliers),
    int nOutlier      = 128;
    GeNeuron* hNeuron = nullptr;
    std::string name;
    Grusoft::GRander rander;
    uint32_t seed;
    bool isCPU      = false;
    int nMostLoop   = 16;  // 8
    float impurity  = 0;
    float imbalance = 0, T_imbal = 1.1f;
    float err_0 = FLT_MAX, err_1 = 0;
    std::vector<double> split_thrshs;
    bool isCheckErr = false;

    // Sinkhorn-Normalized Quantization for 2D matrix         minimize matrix imbalance - alternatingly divide rows and columns by their current standard
    float SinkNormal(shared_ptr<GTensor> hTensor, const void* cpuData, floatGama* curGama, int flag = 0x0);
    float Normal_ROW01(shared_ptr<GTensor> hTensor, void* cpuData, floatGama* curGama, int flag = 0x0);
    hBITARR quant_data = nullptr, best_quant = nullptr;
    // nzMost is the number of elements defined in params.spMost
    size_t szMostQuant = 0x0, nzMost = 0x0, nMostGama = 0;
    floatGama *gama = nullptr, *best_gama = nullptr;
    typNUMBER tpGama = typNUMBER::BF16;  //  using floatGama = floatX;
    virtual float Core(shared_ptr<GTensor> tensor, const void* cpuData, floatGama* curGama, int flag = 0x0) { throw "GeQuant::Core is ...."; }
    virtual float _row_lut(int row, shared_ptr<GTensor> hTensor, const floatX* dat, float* tmpRow, int flag);

   public:
    QUANT_CARD params;
    //   Some layer(first N layers) is more challenging to quantize and requires a higher number of bits
    int bits = -1;
    double maxq, maxshrink, mse, norm, grid;
    float *scale = nullptr, *zero = nullptr;
    bool trits, perchannel = false, sym = false;

    static hQUANT MakeInstance(GeNeuron* hNeuron, const std::string& nam_, QUANT_CARD& params, std::vector<GeNeuron*> neurons, int flag);

    GeQuant() {}
    GeQuant(const std::string& nam_, void* hN, QUANT_CARD& params, int flag = 0x0);
    GeQuant(int bits_, bool perchannel_ = false, bool sym_ = true, bool mse_ = false, double norm_ = 2.4, int grid_ = 100, double maxshrink_ = .8,
            bool trits_ = false)
        : bits(bits_), trits(trits_), perchannel(perchannel_), sym(sym_) {
        maxq = pow(2.0, bits) - 1;
        if (trits)
            maxq = -1;
    }
    virtual ~GeQuant();

    virtual int ExTensor(shared_ptr<GTensor> hBase, std::vector<shared_ptr<GTensor>>&aux, int flag = 0x0);
    virtual bool isRTN() { return params.type == QUANT_MODE::RTN || params.type == QUANT_MODE::RTNf || params.type == QUANT_MODE::RTN_ZS; }
    virtual typNUMBER bit2typ(int flag = 0x0);
    virtual float RTN(shared_ptr<GTensor> tensor, const void* cpuData, int flag = 0x0);
    virtual float RT_NormalF(shared_ptr<GTensor> tensor, const void* cpuData, int flag = 0x0);
    virtual float LowBit(shared_ptr<GTensor> tensor, const void* cpuData, int flag = 0x0);
    virtual float AfterLowBit(shared_ptr<GTensor> tensor, const void* cpuData, int flag = 0x0);
    //  virtual float HighBit(shared_ptr<GTensor> tensor, int flag = 0x0) { throw "GeQuant::HighBit is ...."; }

    virtual float Update(shared_ptr<GTensor> tensor, int flag = 0x0) { throw "GeQuant::Update is ...."; }

    size_t szGama(int flag = 0x0);

    friend class GTensor;
    friend class Fish;
};
using QUANT_FACTORY = std::map<size_t, hQUANT>;

template <typename T>
struct Quantizer : public GeQuant {
    Quantizer(const std::string& nam_, void* hN, QUANT_CARD& param_, int flag = 0x0) : GeQuant(nam_, hN, param_, flag) {}

    /* onlys support shape==2
    virtual void Init(hGMAT x, bool weight = false) {
        shape = x->Shape();
        assert(shape.size() == 2);
        int i, ld = shape[0];
        double *xmax = new double[ld](), *xmin = new double[ld]();
        x->Range(xmin, xmax, perchannel ? 1 : 0);
        scale = new T[ld];
        zero  = new T[ld];

        if (maxq < 0) {
            for (i = 0; i < ld; i++) {
                scale[i] = xmin[i];
                zero[i]  = xmax[i];
            }
        } else {
            for (i = 0; i < ld; i++) {
                scale[i] = (xmax[i] - xmin[i]) / maxq;
                // On CPU tensors rounds half-to-even and on GPU tensor rounds away-from-zero !
                zero[i] = round(-xmin[i] / scale[i]);  // torch.round(-xmin / self.scale)
            }
            double T_zero = (maxq + 1) / 2;
            if (sym) {
                for (i = 0; i < ld; i++) {
                    zero[i] = T_zero;  // torch.full_like(self.scale, (self.maxq + 1) / 2)
                }
            }
        }
        if (GST_util::dump > 0) {
            GST_util::print("+ %s x=[%g-%g] scale=[%g,%g] zero=[%g,%g]\n", __func__, xmin[0], xmax[0], scale[0], scale[ld - 1], zero[0], zero[ld - 1]);
            if (ld < 16) {
                for (i = 0; i < ld; i++) printf("%g ", scale[i]);
                printf("\n");
                for (i = 0; i < ld; i++) printf("%g ", zero[i]);
                printf("\n");
            }
        }

        delete[] xmax;
        delete[] xmin;
    }*/

    virtual float Update(int len, T* col, T* q, int type, T d, T* err, int ld, int flag = 0x0) {
        float loss = 0;
        if (maxq < 0) {
            // return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
            return loss;
        }
        double a = 0, w;
        // T *col=val+no,*q=val+no,*err=hERR->val+no;
        for (int i = 0; i < len; i++, col += ld) {
            if (i == 3)  // only for debug
                i = 3;
            w = *col;
            a = round(float(*col) / scale[i] + zero[i]);  // 10., 13., 10.,
            if (a < 0)
                a = 0;
            if (a > maxq)
                a = maxq;
            *q = scale[i] * (a - zero[i]);  // 0.0618,  0.1926,  0.0604, -0.0526,  0.0000,  0.1382, -0.1075, -0.0445,
            w -= float(*q);
            *err = w / (float)d;                  // err1 = (w - q) / d
            loss += w * w / (float)d / (float)d;  // Losses1[:, i] = (w - q) ** 2 / d ** 2
            q += ld;
            err += ld;
        }
        // GST_util::print();
        return loss;
    }
};

template <typename T>
class Q_Cluster : public Quantizer<T> {
   protected:
    float clip_min = 0.001, clip_max = 1000, eps = 1.0e-6;
    float *log_mu1 = nullptr, *log_mu2 = nullptr;
    int nCluster = 16;
    virtual float Update_KMeans(shared_ptr<GTensor> tensor, int flag = 0x0);

   public:
    // Q_Cluster(const std::string& nam_, void* hN, int flag = 0x0) : Quantizer<T>(nam_, hN, flag) {
    //     nCluster = 8;  // 2 >> bits;
    // }
    // float Update(shared_ptr<GTensor> tensor, int flag = 0x0) override;
};

struct DORT_wrap;
template <typename T>
class Q_Impurity : public Quantizer<T> {
   protected:
    //  Data on random forest
    DORT_wrap* dort = nullptr;

    float clip_min = 0.001, clip_max = 1000, eps = 1.0e-6;
    float *log_mu1 = nullptr, *log_mu2 = nullptr;

    // virtual float MI_Tree(Grusoft::FeatsOnFold* hData_, const Grusoft::FeatVector*, int flag = 0x0);
    virtual float LowBit_GBDT(shared_ptr<GTensor> tensor, const void* cpuData, int flag = 0x0);
    float Core(shared_ptr<GTensor> tensor, const void* cpuData, floatGama* curGama, int flag = 0x0) override;

   public:
    Q_Impurity(const std::string& nam_, void* hN, QUANT_CARD& params, int flag = 0x0);
    // float LowBit(shared_ptr<GTensor> tensor, const void* cpuData, int flag = 0x0) override;
    virtual ~Q_Impurity();
    //  float Update(shared_ptr<GTensor> tensor, int flag = 0x0) override;
};

/**
 *   AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
 * 
 *  1. To eliminate runtime transposes, use transposition of qweight (quantized weight) relative to the original weight!
 */
template <typename T>
class Q_AWQ : public Quantizer<T> {
   protected:   

   public:
    Q_AWQ(const std::string& nam_, void* hN, QUANT_CARD& params, int flag = 0x0);
    float AfterLowBit(shared_ptr<GTensor> tensor, const void* cpuData, int flag = 0x0) override;
    virtual ~Q_AWQ() {  }
};

/**
 *   Signbit quantization on a Johnson-Lindenstrauss (JL) transform
 */
template <typename T, typename Tproj>
class Q_JL : public GeQuant {
   protected:
    bool isOrthogonal;
    //  n_size * group_size = seq_len
    int n_size;
    int seq_len, batch_size, head_size, emb_dim;
    shared_ptr<GTensor> hProj = nullptr, key_quant = nullptr, key_outlier_quant = nullptr, outlier_norms = nullptr;
    Tproj* JL = nullptr;  //   a random projection that would preserves the inner products

    virtual int InitProject(int flag);
    virtual float Update(int flag) {
        float loss = 0;
        return loss;
    }
    virtual float Score(int flag) { return 0.0; }

   public:
    // Q_JL(const std::string& nam_, SHAPE shape, int nOutlier, GeNeuron* hN, int flag);
    float Update(shared_ptr<GTensor> tensor, int flag = 0x0) override;
    virtual ~Q_JL() { FREE_a(JL); }
};

