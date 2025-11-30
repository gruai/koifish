/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Che
 */

#include "GeQuant.hpp"

#include "../GBDT/data_fold/Distribution.hpp"
#include "../GBDT/python/pyMORT_DLL.h"
#include "../Manifold/Fish.hpp"
#include "../Manifold/Neuron.hpp"
#include "GTensor.hpp"
using namespace Grusoft;

hQUANT GeQuant::MakeInstance(GeNeuron* hNeuron, const std::string& nam_, QUANT_CARD& params, std::vector<hGensor> tensors, int flag) {
    hQUANT hQuant = nullptr;
    params.type   = params.TypeOf(nam_);
    if (params.type == NO_QUANT)
        return hQuant;

    QUANT_FACTORY& quants = hNeuron->GetFish()->quants;
    auto key              = params.Hash(params, typeid(hNeuron), nam_);
    if (quants.find(key) != quants.end()) {
        hQuant = quants[key];
    } else {
        switch (params.type) {
            case MIQ:
                hQuant = std::make_shared<Q_Impurity<float>>(nam_ + "_quant", hNeuron, params, 0x0);
                break;
            case KV_JL: {
                SelfAttention* qkv = dynamic_cast<SelfAttention*>(hNeuron);
                assert(qkv != nullptr);
                // hQuant = std::make_shared<Q_JL<float,float>>(nam_+"_quant",SHAPE({head_dim,head_dim*2}), 256,this,0x0);
            } break;
            default:
                break;
        }
        quants[key] = hQuant;
    }
    // hFish->quants[] = hQuant;
    if (hQuant != nullptr) {
        for (auto t : tensors) {
            assert(t->hQuant == nullptr);
            t->hQuant = hQuant;
        }
    }
    return hQuant;
}

GeQuant::GeQuant(const std::string& nam_, void* hBase, QUANT_CARD& param_, int flag) : name(nam_), params(param_) {
    hNeuron = (GeNeuron*)(hBase);
    std::hash<std::string> hasher;
    size_t hash = hasher(name);  // Returns a size_t (unsigned integer)
    rander.Init(hash);
    bits = params.bits;
    assert(params.isValid());
    tpGama = TYPE_<floatGama>();

    // T_imbal = 1.0f;
    nzMost = 1;
    for (auto n : params.spMost) nzMost *= n;
    if (nzMost > 1) {
        assert(nzMost * bits % 8 == 0);
        szQuant    = nzMost * bits / 8;
        quant_data = new BIT_8[szQuant];
    }
}

typNUMBER GeQuant::tpQuant(int flag) {
    switch (bits) {
        case 4:
            return typNUMBER::Q4;
        case 3:
            return typNUMBER::Q3;
        case 2:
            return typNUMBER::Q2;
        defaut:
            assert(0);
            break;
    }
    return typNUMBER::Q4;
}

GeQuant::~GeQuant() { FREE_a(quant_data); }

/*
template <typename T, typename Tproj>
Q_JL<T, Tproj>::Q_JL(const std::string& nam_, SHAPE shape_, int nLier, GeNeuron* hN, int flag) : GeQuant(nam_, shape_, hN, flag) {
    nOutlier           = nLier;
    SelfAttention* qkv = dynamic_cast<SelfAttention*>(hN);
    auto config        = qkv->GetFish()->config;
    assert(qkv != nullptr);
    emb_dim    = qkv->head_dim;
    group_size = 32;
    seq_len    = config.chat_sampler.seq_len;
    assert(seq_len % group_size == 0);
    seq_len = n_size = seq_len / group_size;
    head_size        = qkv->n_head;
    batch_size       = 1, head_size, n_size;
    InitProject(flag);
}*/

/**
 * 1.


     def init_rot_dir(self):
        rot_matrices = []
        num_chunks = (self.dim[1] + self.dim[0] - 1) // self.dim[0]
        for i in range(num_chunks):
            start_idx = i * self.dim[0]
            end_idx = (i + 1) * self.dim[0]
            q, _ = torch.linalg.qr(self.proj_dir[:, start_idx:end_idx], mode='reduced')
            rot_matrices.append(q)
        return torch.cat(rot_matrices, dim=-1) * math.sqrt(self.dim[0])

    def compose_rand_hadamard_transform(self):
        H = torch.from_numpy(hadamard(self.dim[0], dtype=float) / math.sqrt(self.dim[0])).to(self.device)
        HD = (H * (2. * torch.randint(0, 2, (self.dim[0],), device=self.device) - 1.)).to(self.proj_dir_score.dtype)
        return torch.einsum('dn,dm-> mn', self.proj_dir_score, HD)
 */
template <typename T, typename Tproj>
int Q_JL<T, Tproj>::InitProject(int flag) {
    //  torch.randn(self.dim, generator=rng, dtype=torch.float32, device=self.device)
    SHAPE shape = this->params.spMost;
    hProj       = GT(hNeuron->GetFish(), typNUMBER::BF16, shape, 0x0, name + ".JL");  //  hN->hFish
    hProj->Alloc();
    JL   = TO<Tproj>(hProj);
    seed = rander.RandU32();  // !is different with hProj->param_seed
    //  a standard normal distribution (mean=0, std=1).
    hProj->tpInit = INIT_WEIGHT::GAUSSIAN_NORMAL;
    hProj->InitParam(0x0);
    SUM::nInitParam--;  // JL project matrix may also be params in later version
    // hProj->Print(hProj->name,0,-1);
    int batch = 1, sketch_dim = hProj->shape[0], hash_dim = sketch_dim / 8;
    int outlier_sketch_dim = nOutlier, outlier_hash_dim = outlier_sketch_dim / 8;
    // auto key_quant = torch::zeros({batch, head, n, group_size, hash_dim}, options).contiguous();
    key_quant = GT(hNeuron->GetFish(), typNUMBER::BF16, SHAPE{batch, head_size, n_size, group_size, hash_dim}, 0x0, name + ".quant");
    // auto key_outlier_quant = torch::zeros({batch, head, n, group_size, outlier_hash_dim}, options).contiguous();
    key_outlier_quant = GT(hNeuron->GetFish(), typNUMBER::BF16, SHAPE{batch, head_size, n_size, group_size, outlier_hash_dim}, 0x0, name + ".lier_quant");
    // auto outlier_norms = torch::zeros({batch, head, n, group_size}, options_outlier_norm).contiguous();
    outlier_norms = GT(hNeuron->GetFish(), typNUMBER::BF16, SHAPE{batch, head_size, n_size, group_size}, 0x0, name + ".lier_norm");

    return 0x0;
}

// Explicit instantiation for specific types
template class Q_JL<float, float>;
template class Q_JL<bf16, bf16>;

#include "../GBDT/tree/GBRT.hpp"
using namespace Grusoft;

template <typename T>
Q_Impurity<T>::Q_Impurity(const std::string& nam_, void* hN, QUANT_CARD& params, int flag) : Quantizer<T>(nam_, hN, params, flag) {
    assert(this->bits > 0 && this->bits <= 8);
    this->isSinkNormal = true;
    // this->params.type  = this->bits==2 ? QUANT_MODE::RTN_2 : QUANT_MODE::RTN_4;
    // if (dort == nullptr)
    //     dort = static_cast<DORT_wrap*>(LiteMORT_init(nullptr, 0, nullptr, 0x0));
    // LiteBOM_Config& config = dort->config;
    // config.feat_quanti     = 1024;
    // config.verbose         = 1000;
    // config.objective       = "quant";
}

float GeQuant::AfterLowBit(shared_ptr<GTensor> hTensor, void* srcData, int flag) {
    SUM::nQuantTensor++;
    SUM::szQuantBits += 0;
    hTensor->rc_normal = this->isSinkNormal;
    double e, e_0, e_1, err = 0, err_1 = -DBL_MAX, err_0 = DBL_MAX, len, a, avg = 0, a_0 = DBL_MAX, a_1 = -DBL_MAX;
    auto ginfo = hTensor->ginfo;
    if (0) {  // verify
        int nRow = hTensor->shape[0], nCol = hTensor->shape[1], i, j, pos = 0;
        floatX *devX = hTensor->GetDataX(), *dat0 = (floatX*)srcData, *dat1 = new floatX[nRow * nCol], *gama = new floatX[nRow * 16];
        D2H(devX, dat1, sizeof(floatX) * nRow * nCol);
        D2H(hTensor->gama_T() + nRow + nCol, gama, sizeof(floatX) * nRow * 2);
        for (i = 0; i < nRow; i++) {
            float zero = T2Float(gama + 2 * i), step = T2Float(gama + 2 * i + 1);
            for (e_0 = DBL_MAX, e_1 = -DBL_MAX, err = 0, len = 0, j = 0; j < nCol; j++, pos++) {
                a   = T2Float(dat0 + pos);
                a_0 = std::min(a, a_0), a_1 = std::max(a, a_1);
                len += a * a;
                e = T2Float_delta(dat1, dat0, pos);
                if (j == 369 && i == 0) {  // j=369 -0.6015625
                    DEBUG_HERE;
                }
                e_1 = std::max(e_1, e), e_0 = std::min(e_0, e);
                err += e * e;
            }
            err /= len, avg += err;
            err_0 = std::min(err, err_0), err_1 = std::max(err, err_1);
            if (i < 16) {
                _INFO("[RTN]_%d arr=[%.4g-%.4g] err=[%.4g-%.4g]\n", i, a_0, a_1, e_0, e_1);
            }
        }
        delete[] dat1;
        _INFO("[Quant] average=%.4g sum=[%.4g,%.4g]\t@%s\n", ginfo->nrm_1 / nRow / nCol, ginfo->sum_1, ginfo->sum_2, hTensor->name);
        _INFO("\terr=%.4g[%.4g-%.4g] A=[%.4g-%.4g]", avg / nRow, err_0, err_1, a_0, a_1);
    }
    return err;
}

float GeQuant::RTN(shared_ptr<GTensor> hTensor, void* cpuData, int flag) {
    double t0 = GST_ms(), err = 0, e = 0, e_0, e_1;
    assert(hTensor->isWMAT());
    int nRow = hTensor->shape[0], nCol = hTensor->shape[1], nQuant = 0x1 << this->bits, i, pos, row;
    floatX *mat0 = (floatX*)(cpuData), *dat = nullptr, b;

    for (row = 0; row < nRow; row++) {
        float vmax = -FLT_MAX, vmin = FLT_MAX, a;
        dat = mat0 + row * nCol;
        for (i = 0; i < nCol; i++) {
            a    = T2Float(dat + i);
            vmax = std::max(vmax, a), vmin = std::min(vmin, a);
        }

        floatGama* gama_ = this->gama + 2 * row;
        float step = (vmax - vmin) / (nQuant - 1), zero = vmin;
        gama_[0] = zero, gama_[1] = step;
        hBITARR quanti = quant_data + nCol * row * this->bits / 8;
        for (e_0 = DBL_MAX, e_1 = -DBL_MAX, i = 0; i < nCol; i++) {
            pos = i;
            a   = T2Float(dat + i);
            // value = std::clamp(value, min_val, max_val);
            int qid = std::round((a - zero) / step);
            assert(qid >= 0 && qid < nQuant);
            BIT_SET_k(quanti, pos, qid, this->bits);
            assert(BIT_GET_k(quanti, pos, this->bits) == qid);
            e   = a - (zero + step * qid);  //+ step*0.5
            e_1 = std::max(e_1, e), e_0 = std::min(e_0, e);
            err += e * e;
            if (i == 369 && row == 0) {
                DEBUG_HERE;
            }
        }
        // if (row < 16) {
        //     _INFO("[RTN]_%d arr=[%.4g-%.4g] err=[%.4g-%.4g]\n", row, vmin, vmax, e_0, e_1);
        // }
    }
    err = sqrt(err / nRow / nCol);  // 0.004094
    e   = hTensor->ginfo->nrm_1 / nRow / nCol;
    if (err >= e) {
        _WARN("[RTN]_Q<%d> %g>%g\n", this->bits, err, e);
    }
    return err;
}

template class Quantizer<float>;
template class Quantizer<bf16>;

FeatsOnFold* Mat2DORT(LiteBOM_Config config, ExploreDA* edaX, typNUMBER tpN, void* hData, int nMostFeat, int nSamp, DORT_wrap* dort, int flag);

/*
    It's really a suprize to find that GBDT would improve quant.    cys 11/22/2025
*/
template <typename T>
float Q_Impurity<T>::LowBit_GBDT(shared_ptr<GTensor> hTensor, void* srcData, int flag) {
    double t0 = GST_ms();
    assert(hTensor->isWMAT());
    int nRow = hTensor->shape[0], nCol = hTensor->shape[1];
    try {
        int nSamp = nCol, nFeat_0 = 1, nQuant = 0x1 << this->bits;  // hFold->config.num_trees;
        assert(nSamp > 0 && nFeat_0 > 0 && nSamp % 8 == 0);
        dort                   = static_cast<DORT_wrap*>(LiteMORT_init(nullptr, 0, nullptr, 0x0));
        LiteBOM_Config& config = dort->config;
        // config.objective       = "quant";
        // _INFO("\n[DORT] nSamp=%d,nFeat_0=%d %s hExDA=%p********* \n\n", nSamp, nFeat_0, dort->merge_info.c_str(), dort->hEDA_train);
        FeatsOnFold* hFold = Mat2DORT(config, dort->hEDA_train, hTensor->type, srcData, nFeat_0, nSamp, dort, flag | FeatsOnFold::DF_TRAIN);
        this->gama         = new floatGama[nRow * nQuant];
        if (config.objective == "quant") {
            // this->impurity = hFold->Quant(this->bits, this->tpGama, this->gama, this->quant_data, string(hTensor->name), 0x0);
            float impurity = 0.f, a, a0 = FLT_MAX, a1 = 0.f, t0 = GST_ms();
            vector<float> errs;
            errs.resize(hFold->feats.size());
            //  #pragma omp parallel for
            for (auto hFeat : hFold->feats) {
                // if (hFeat->id != 23)
                //     continue;  // ony for debug
                floatGama* gama_ = this->gama + nQuant * hFeat->id;
                hBITARR quant_   = this->quant_data + nSamp * this->bits / 8 * hFeat->id;
                // MI_Tree(hFold, hFeat);
                errs[hFeat->id] = hFeat->Quant_RTN(this->bits, this->tpGama, gama_, quant_, hFold, flag);

                // PrintT<floatGama>("feat_gama", gama, 16, 1, 1, 1, -1);
            }
            for (auto hFeat : hFold->feats) {
                a1 = std::max(a, a1), a0 = std::min(a, a0), impurity += a;
            }
            impurity /= hFold->feats.size();
            // _INFO("<QUANT_%d>@%s nF=%d(%ld) impurity=%g[%g-%g]\t t=%.5gms\n", bits, desc.c_str(), feats.size(), nSamp, impurity, a0, a1, GST_ms() - t0);
        } else {
            int nTree   = 1;                                                                                                       // just grow leaf
            dort->hGBRT = new GBRT(hFold, nullptr, 0.0, flag == 0 ? BoostingForest::REGRESSION : BoostingForest::CLASIFY, nTree);  //
            dort->hGBRT->Train("", 0x0);
            // memory
            dort->hGBRT->ClearData(), dort->hGBRT->ClearHisto();
        }

        hTensor->BeforeQuant(this);
        H2D(hTensor->data, this->quant_data, this->szQuant);
        H2D(hTensor->gama_T(), this->gama, sizeof(floatGama) * (nRow+nCol+nRow * nQuant));
        delete[] this->gama;
        delete hFold;
        delete dort;
        hTensor->Print(hTensor->name, 0x0, -1);
        //_INFO("<QUANT_%d> @\"%s\" t=%.3gms\t \n", this->bits, hTensor->name, GST_ms() - t0);
        GeQuant::LowBit(hTensor, srcData, flag);  //
    } catch (char* sInfo) {
        _ERROR("\n!!!!!! EXCEPTION@Q_Impurity::LowBit \n!!!!!!\"%s\"\n\n", sInfo);
        throw sInfo;
    } catch (...) {
        _ERROR("\n!!!!!! EXCEPTION@Q_Impurity::LowBit %s!!!!!!\n\n", "...");
    }
    fflush(stdout);

    // hTensor->AfterQuant();
    return this->impurity;
}

// the shape&type of srcData is defined in hTensor
template <typename T>
float Q_Impurity<T>::LowBit(shared_ptr<GTensor> hTensor, void* srcData, int flag) {
    // return LowBit_GBDT(hTensor, srcData, flag);
    double t0 = GST_ms(), a0 = DBL_MAX, a1 = 0;
    assert(hTensor->isWMAT());
    int nRow = hTensor->shape[0], nCol = hTensor->shape[1], minLeaf = 2, gama_off = nRow + nCol;

    int nSplit = -1, nQuant = 0x1 << this->bits;  // hFold->config.num_trees;
    this->gama = new floatGama[nRow * nQuant + nRow + nCol];
    if (this->isSinkNormal) {
        this->SinkNormal(hTensor, srcData, flag);

        if (!this->isSinkNormal) {  // sinknormal may fail, so all rs & cs are 1.0
        }
    }
    if (this->params.type == QUANT_MODE::MIQ) {
        dort                   = static_cast<DORT_wrap*>(LiteMORT_init(nullptr, 0, nullptr, 0x0));
        LiteBOM_Config& config = dort->config;
        config.feat_quanti     = 1024;
        config.verbose         = 1000;
        config.objective       = "quant";
        // return this->imbalance;  // only for debug
        FeatsOnFold* hFold = Mat2DORT(dort->config, dort->hEDA_train, hTensor->type, srcData, nRow, nCol, dort, flag | FeatsOnFold::DF_TRAIN);

#pragma omp parallel for
        for (int id = 0; id < nRow; id++) {
            auto hFeat       = hFold->feats[id];
            floatGama* gama_ = this->gama + nQuant * id;
            hBITARR quant_   = this->quant_data + nCol * this->bits / 8 * id;
            T* val           = TO<T>(hTensor) + nCol * id;
            // Quant_TREE(this->bits, this->tpGama, gama_, quant_, flag);
            hMIHISTO root = std::make_shared<MINI_HISTO>(hFeat->myDistri(), -1, -1, 0);  // hData_, hFeat->myDistri()
            hFeat->miniTree.push(root);
            while (hFeat->miniTree.size() < nQuant) {
                hMIHISTO head = hFeat->miniTree.top();
                hFeat->miniTree.pop();
                if (head->Split_v2(minLeaf)) {
                    hFeat->miniTree.push(head->left), hFeat->miniTree.push(head->rigt);
                }
                nSplit++;
            }
            hFeat->Quant_MITREE(this->bits, this->tpGama, gama_, quant_, 0x100);
            //  PrintT<floatGama>("feat_gama", gama, 16, 1, 1, 1, -1);
        }
        for (auto hFeat : hFold->feats) {
            a1 = std::max(hFeat->wGain, a1), a0 = std::min(hFeat->wGain, a0), this->impurity += hFeat->wGain;
        }
        this->impurity /= nRow;
        _INFO("<QUANT_MI_%d>@%s nF=%d(%ld) impurity=%g[%g-%g]\t split=%g t=%.5gms\n", this->bits, "", nRow, nCol, this->impurity, a0, a1, nSplit * 1.0 / nRow,
              GST_ms() - t0);
        delete hFold;
        delete dort;
    } else {
        Quantizer<T>::RTN(hTensor, srcData, flag);
    }
    hTensor->BeforeQuant(this);
    H2D(hTensor->data, this->quant_data, this->szQuant);
    H2D(hTensor->gama_T() + gama_off, this->gama, sizeof(floatGama) * nRow * nQuant);
    delete[] this->gama;

    hTensor->Print(hTensor->name, 0x0, -1);
    //_INFO("<QUANT_%d> @\"%s\" t=%.3gms\t \n", this->bits, hTensor->name, GST_ms() - t0);
    GeQuant::AfterLowBit(hTensor, srcData, flag);  //
    return this->impurity;
}

template <typename T>
Q_Impurity<T>::~Q_Impurity() {
    if (dort != nullptr) {
        LiteMORT_clear(dort);
        dort = nullptr;
    }
}
template class Q_Impurity<bf16>;
template class Q_Impurity<float>;

// template <>
// DORT_wrap* Q_Impurity<bf16>::dort = nullptr;
// template <>
// DORT_wrap* Q_Impurity<float>::dort = nullptr;

// Dual-scaling of a matrix
template <typename T, typename Tscal>
double G_Scale_RC(T* mat, int nRow, int nCol, Tscal* row_scal, Tscal* col_scal, double T_imb, int flag = 0x0) {
    double sr = 1.0, imbalance = 0.0, clip_min = 0.001, clip_max = 1000;
    double sr0 = FLT_MAX, sc0 = FLT_MAX, sr1 = -FLT_MAX, sc1 = -FLT_MAX, a;

    for (int r = 0; r < nRow; r++) {
        a   = T2Float(row_scal + r);
        sr0 = std::min(sr0, a), sr1 = std::max(sr1, a);
        row_scal[r] = std::min(a, clip_max), row_scal[r] = std::max(a, clip_min);
    }
    for (int c = 0; c < nCol; c++) {
        a   = T2Float(col_scal + c);
        sc0 = std::min(sc0, a), sc1 = std::max(sc1, a);
        col_scal[c] = std::min(a, clip_max), col_scal[c] = std::max(a, clip_min);
    }
    a         = std::max(std::min(sr0, sc0), 1.0e-12);
    imbalance = std::max(sr1, sc1) / a;
    if (imbalance < T_imb)
        return imbalance;
    T *row = mat, *col = mat;
    for (int r = 0; r < nRow; r++, row += nCol) {
        sr        = T2Float(row_scal + r);
        Tscal* sc = col_scal;
        for (int c = 0; c < nCol; c++, sc++) {
            row[c] /= sr;
            row[c] /= T2Float(sc);
        }
    }
    return imbalance;
}

// def imbalance(mat):
//         s1, s2 = measure(mat, 1), measure(mat, 0)
//         s_min = torch.minimum(s1.min(), s2.min()).clamp_min(1e-12)
//         s_max = torch.maximum(s1.max(), s2.max())
//         return s_max / s_min          # scalar
float GeQuant::SinkNormal(shared_ptr<GTensor> hTensor, void* cpuData, int flag) {
    double t0   = GST_sec();
    size_t nEle = hTensor->size(), dGrid = CEIL_DIV(nEle, CU_T4B_MIDDLE);
    int nRow = hTensor->shape[0], nCol = hTensor->shape[1], i = 0, loop = 0;
    double imb = 0.0, imb0 = 0.0, clip_min = 0.001, clip_max = 1000, off = 0, len = hTensor->ginfo->sum_2, len1 = 0, a, gs_0 = DBL_MAX, gs_1 = -DBL_MAX;
    imbalance     = FLT_MAX;  //
    bool isVerify = true;
    /**/
    if (cpuData == nullptr) {
        floatGama *rowStdDev = hTensor->gama_T(), *colStdDev = rowStdDev + nRow;
        void* quant  = hTensor->BeforeQuant(this);
        floatX* mat0 = TO<floatX>(hTensor);
        for (int loop = 0; loop < this->nMostLoop; loop++) {
            // CU_RowStdDev<<<CEIL_DIV(nRow, CU_T4B_MIDDLE), CU_T4B_MIDDLE>>>(mat0, rowStdDev, nRow, nCol);
            // CU_ColStdDev<<<CEIL_DIV(nCol, CU_T4B_MIDDLE), CU_T4B_MIDDLE>>>(mat0, colStdDev, nRow, nCol);
            // s_min = torch.minimum(s1.min(), s2.min()).clamp_min(1e-12)
            // s_max = torch.maximum(s1.max(), s2.max())
            // imbalance = s_max / s_min          # scalar
            // CU_DualScale<<<CEIL_DIV(M*N,CU_T4B_MIDDLE),CU_T4B_MIDDLE>>>(mat0, rowStdDev,colStdDev, M, N);
            if (this->impurity > 1.0)
                break;
        }
        hTensor->AfterQuant(this, typNUMBER::F8E5M2, nullptr);
    } else {
        floatX *mat0 = (floatX*)(cpuData), *mat1 = nullptr, b;
        if (isVerify) {
            mat1 = new floatX[nRow * nCol];
            memcpy(mat1, mat0, sizeof(floatX) * nRow * nCol);
            // for (int i = 0; i < nRow * nCol; i++) {
            //     a = T2Float(mat1 + i);
            //     len += a * a;
            //     // assert(fabs(a)<1.0e2);
            // }
            // assert(len==hTensor->ginfo->sum_2);
        }
        // _INFO("SinkNormal A=[%dx%d]=%g \t@\"%s\"\n", nRow, nCol, len, hTensor->name);  // return 0.0;
        double *rowStdDev = new double[(nRow + nCol) * 2], *colStdDev = rowStdDev + nRow, *rowS = colStdDev + nCol, *colS = rowS + nRow;
        for (int r = 0; r < nRow; r++) rowS[r] = 1.0;
        for (int c = 0; c < nCol; c++) colS[c] = 1.0;
        for (loop = 0; loop < this->nMostLoop; loop++) {
            for (int r = 0; r < nRow; r++) {
                rowStdDev[r] = G_StdDev(nCol, mat0 + r * nCol, 1);
            }
            for (int c = 0; c < nCol; c++) {
                colStdDev[c] = G_StdDev(nRow, mat0 + c, nCol);
            }
            imb = G_Scale_RC(mat0, nRow, nCol, rowStdDev, colStdDev, T_imbal);
            if (loop == 0)
                imb0 = imb;
            if (imb < T_imbal) {
                imbalance = imb;
                break;
            }
            for (int r = 0; r < nRow; r++) {
                rowS[r] *= rowStdDev[r];
            }
            for (int c = 0; c < nCol; c++) {
                colS[c] *= colStdDev[c];
            }
            if (imb > imbalance)  // recue memory copy
                break;
            imbalance = imb;
        }
        hTensor->ginfo->imbalance = imbalance;
        // copy to gama
        floatGama *gamaR = gama, *gamaC = gama + nRow;
        for (int r = 0; r < nRow; r++) {
            gamaR[r] = rowS[r];
            gs_0 = std::min(gs_0, rowS[r]), gs_1 = std::max(gs_1, rowS[r]);
        }
        for (int c = 0; c < nCol; c++) {
            gamaC[c] = colS[c];
        }
        if (isVerify) {
            G_Scale_RC(mat1, nRow, nCol, gamaR, gamaC, T_imbal);
            // G_Scale_RC(mat1, nRow, nCol, rowS, colS);
            for (int i = 0; i < nRow * nCol; i++) {
                b = mat1[i] - mat0[i];
                a = T2Float(&b), off += a * a;
                a = T2Float(mat1 + i), len1 += a * a;
            }
            off = sqrt(off / len);  // sqrt(off / (nRow * nCol));
            delete[] mat1;
        }
        delete[] rowStdDev;
        isSinkNormal = loop > 0;
    }  //  SinkNormal A=[1024x3072] loop=10 imbalance=1.0078 err=18.7696   @"model.layers.19.mlp.down_proj.weight"
    if (off > 1.0) {
        _WARN("SinkNormal A=[%dx%d] loop=%d imbalance=(%g->%g) err=%.4g(%g) T=%.3g \t@\"%s\"\n", nRow, nCol, loop, imb0, imbalance, off, len, GST_sec() - t0,
              hTensor->name);
    } else
        _INFO("SinkNormal A=[%dx%d] loop=%d imbalance=(%g->%g) err=%.4g(%g) T=%.3g gs=[%g-%g] \t@\"%s\"\n", nRow, nCol, loop, imb0, imbalance, off, len, gs_0,
              gs_1, GST_sec() - t0, hTensor->name);
    return imbalance;
}