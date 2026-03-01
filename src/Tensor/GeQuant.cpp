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

hQUANT GeQuant::MakeInstance(GeNeuron* hNeuron, const std::string& nam_, QUANT_CARD& params, std::vector<GeNeuron*> neurons, int flag) {
    hQUANT hQuant = nullptr;
    if (params.isPass())
        return hQuant;

    QUANT_FACTORY& quants = hNeuron->GetFish()->quants;
    auto key              = params.Hash(params, typeid(hNeuron), nam_);
    if (quants.find(key) != quants.end()) {
        hQuant = quants[key];
    } else {
        switch (params.type) {
            // case AWQ:
            //     hQuant = std::make_shared<Q_AWQ<float>>(nam_ + "_awq", hNeuron, params, 0x0);
            //     break;
            case RTN_ZS:
                hQuant = std::make_shared<Q_AWQ<float>>(nam_ + "_rtnzs", hNeuron, params, 0x0);
                break;
            case MINI:
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
        for (auto n : neurons) {
            std::vector<hGTensor> aux;
            hGensor t = n->w;
            assert(t->hQuant == nullptr);
            t->hQuant = hQuant;
            hQuant->ExTensor(t, aux);
            n->out->AddSrc(aux);
        }
    }
    return hQuant;
}

GeQuant::GeQuant(const std::string& nam_, void* hBase, QUANT_CARD& param_, int flag) : name(nam_), params(param_) {
    hNeuron = (GeNeuron*)(hBase);
    std::hash<std::string> hasher;
    size_t hash = hasher(name);  // Returns a size_t (unsigned integer)
    rander.Init(hash);
    bits = params.default_bits;
    assert(params.isValid());
    tpGama = TYPE_<floatGama>();

    // T_imbal = 1.0f;
    nzMost = 1;
    for (auto n : params.spMost) nzMost *= n;
    if (nzMost > 1) {
        assert(nzMost * bits % 8 == 0);
        szMostQuant = nzMost * bits / 8;
        quant_data  = new BIT_8[szMostQuant * 2];
        best_quant  = quant_data + szMostQuant;
    }
    int nRow = params.spMost[0], nCol = params.spMost[1], nQuant = 0x1 << bits;
    nMostGama = std::max(nRow, nCol) * nQuant + nRow + nCol;
    gama      = new floatGama[nMostGama * 2];
    best_gama = this->gama + nMostGama;
}

typNUMBER GeQuant::bit2typ(int flag) {
    /*switch (bits) {
        case 4:
            return typNUMBER::Q4;
        case 3:
            return typNUMBER::Q3;
        case 2:
            return typNUMBER::Q2;
        default:
            assert(0);
            break;
    }
    return typNUMBER::Q4;*/
    return Bits2Type(bits, flag);
}

GeQuant::~GeQuant() {
    FREE_a(quant_data);
    FREE_a(gama);
}

int GeQuant::ExTensor(hGTensor hBase, std::vector<hGTensor>& aux, int flag) {
    assert(hBase->isWMAT());
    // assert(hBase->data != nullptr);

    SHAPE sp;
    int group       = params.T_group;
    size_t offset   = hBase->nByte();
    string basename = G_prefix_(hBase->name, "."), a = basename + ".qweight";
    if (G_Has_(basename, {"model.layers.5.mlp.up_proj"})) {
        DEBUG_HERE;
    }
    strcpy(hBase->name, a.c_str());
    // hBase->type = params.tpQWeight;
    string sS = basename + MODEL_CARD::sQscale, sZ = basename + MODEL_CARD::sQzeros;
    switch (params.type) {
        case RTN_ZS:
            std::swap(hBase->shape[0], hBase->shape[1]);  // qweight is transposition of original weight
            hBase->ne[0] = hBase->shape[0], hBase->ne[1] = hBase->shape[1];
            sp            = {hBase->shape[0] / group, hBase->shape[1]};     // 20x4096 = 32x2560
            hBase->qScale = GT(hBase->hFish, params.tpScale, sp, 0x0, sS);  // hBase->Partial(sS, offset, sp, params.tpScale);
            aux.push_back(hBase->qScale);
            // hBase->hFish->InitGensor(nullptr, sS.c_str(), hBase->qScale, false);
            offset += hBase->qScale->nByte();
            if (params.isZeroPoint) {
                sp           = {hBase->shape[0] / group, hBase->shape[1]};    // 80x512 = 32x2560/2
                hBase->qZero = GT(hBase->hFish, params.tpZero, sp, 0x0, sZ);  // hBase->Partial(sZ, offset, sp, params.tpZero);
                aux.push_back(hBase->qZero);
                // hBase->hFish->InitGensor(nullptr, sZ.c_str(), hBase->qZero, false);
            }

            break;
        default:
            break;
    }
    for (auto t : aux) {
        t->hQuant = shared_from_this();
    }

    return 0x0;
}

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
Q_Impurity<T>::Q_Impurity(const std::string& nam_, void* hN, QUANT_CARD& param_, int flag) : Quantizer<T>(nam_, hN, param_, flag) {
    assert(this->bits > 0 && this->bits <= 8);

    // if (dort == nullptr)
    //     dort = static_cast<DORT_wrap*>(LiteMORT_init(nullptr, 0, nullptr, 0x0));
}

float GeQuant::AfterLowBit(shared_ptr<GTensor> hTensor, const void* srcData, int flag) {
    SUM::nQuantTensor++;
    SUM::szQuantBits += 0;
    hTensor->disq.rc_normal = this->params.norm;

    double e, e_0, e_1, err = 0, err_1 = -DBL_MAX, err_0 = DBL_MAX, len, a, avg = 0, a_0 = DBL_MAX, a_1 = -DBL_MAX, errQ = 0;
    auto disq = hTensor->disq;
    int nRow = hTensor->shape[0], nCol = hTensor->shape[1], i, j, pos = 0, outlier = 0;
    double average = hTensor->disq.nrm_1 / nRow / nCol, rOutlier = 0;
    if (isCheckErr) {  // verify
        floatX *devX = hTensor->GetDataX(), *dat0 = (floatX*)srcData, *dat1 = new floatX[nRow * nCol], *host_gama = new floatX[nRow * 16 + nRow + nCol];
        D2H(devX, dat1, sizeof(floatX) * nRow * nCol);
        D2H(hTensor->gama_T(), host_gama, sizeof(floatX) * (nRow * 2 + nRow + nCol));
        for (i = 0; i < nRow; i++, pos += nCol) {
            // if (i != 151644)                continue;
            float f1 = T2Float(host_gama + nRow + nCol + 2 * i), f2 = T2Float(host_gama + nRow + nCol + 2 * i + 1);
            for (e_0 = DBL_MAX, e_1 = -DBL_MAX, err = 0, len = 0, j = 0; j < nCol; j++) {
                a   = T2Float(dat0 + pos + j);
                a_0 = std::min(a, a_0), a_1 = std::max(a, a_1);
                len += a * a;
                e = T2Float_delta(dat1, dat0, pos + j);
                // printf("%g\t%g\n", T2Float(dat1 + pos + j), T2Float(dat0 + pos + j));
                if (fabs(e / average) > rOutlier) {
                    rOutlier = fabs(e / average);
                    outlier  = pos + j;
                }
                if (pos + j == 393712) {  // i=384 j=496    -0.2578=>0.2578
                    DEBUG_HERE;
                    Distri_PIPE disR(f1, f2);
                    int qid = disR.X2NormalF(bits, a);
                }
                e_1 = std::max(e_1, e), e_0 = std::min(e_0, e);
                err += e * e;
            }
            errQ += err;
            err /= len, avg += err;
            if (err_1 < err) {  // get max_err row
                err_1 = err;
            }
            err_0 = std::min(err, err_0);
            // if (i < 16) {
            //     _INFO("[]_%d arr=[%.4g-%.4g] err=[%.4g-%.4g]\n", i, a_0, a_1, e_0, e_1);
            // }
        }
        delete[] dat1;
        errQ     = sqrt(errQ / nRow / nCol);
        impurity = errQ / average;
        _INFO("[Quant] average=%.4g sum=[%.4g,%.4g]\t@%s\n", disq.nrm_1 / nRow / nCol, disq.sum_1, disq.sum_2, hTensor->name);
        _INFO("\terr=%.4g[%.4g-%.4g] A=[%.4g-%.4g]", avg / nRow, err_0, err_1, a_0, a_1);
    }

    hTensor->disq.err  = impurity;
    hTensor->disq.info = best_.ToString();
    for (auto t : hTensor->refered) {
        assert(t->type != hTensor->type);
        t->type = hTensor->type;
        assert(t->hQuant == nullptr);
        t->hQuant = hTensor->hQuant;
    }

    e            = hTensor->disq.nrm_1 / nRow / nCol;
    float T_errQ = bits == 3 ? params.T_errQ + 0.1 : params.T_errQ;
    if (impurity >= T_errQ) {
        _WARN("[]_Q<%d>_fnorm=%d impurity=%.3g |a|=%.3g @%s\n", bits, params.norm, impurity, e, hTensor->name);
    }
    hTensor->Print(hTensor->name, 0, 0);

    return err;
}

float GeQuant::RTN(shared_ptr<GTensor> hTensor, const void* srcData, int flag) {
    if (params.isNormalFloat)
        return RT_NormalF(hTensor, srcData, flag);

    double t0 = GST_ms(), err = 0, e = 0, e_0, e_1;

    int nRow = hTensor->shape[0], nCol = hTensor->shape[1], nQuant = 0x1 << this->bits, i, pos, row;
    const floatX *mat0 = (floatX*)(srcData), *dat = nullptr;
    floatX b;
    floatGama *gamaRow = gama, *gamaCol = gamaRow + nRow;
    float sR = 1.0, sC = 1.0, *tmpRow = new float[nCol];
    for (row = 0; row < nRow; row++) {
        float vmax = -FLT_MAX, vmin = FLT_MAX, a, a0 = 0;
        assert(0);  // RT_NormalF is much better than RTN!
        /*if (isSinkNormal)
            sR = gamaRow[row];
        dat = mat0 + row * nCol;
        for (i = 0; i < nCol; i++) {
            if (isSinkNormal)
                sC = T2Float(gamaCol + i);
            a         = T2Float(dat + i) / sR / sC;
            tmpRow[i] = a;
            vmax = std::max(vmax, a), vmin = std::min(vmin, a);
        }

        floatGama* gama_ = this->gama + nRow + nCol + 2 * row;
        float step = (vmax - vmin) / (nQuant - 1), zero = vmin;
        gama_[0] = zero, gama_[1] = step;
        hBITARR quanti = quant_data + nCol * row * this->bits / 8;
        for (e_0 = DBL_MAX, e_1 = -DBL_MAX, i = 0; i < nCol; i++) {
            pos = i;
            // a0  = T2Float(dat + i);
            a = tmpRow[i];
            // value = std::clamp(value, min_val, max_val);
            int qid = std::round((a - zero) / step);
            assert(qid >= 0 && qid < nQuant);
            BIT_SET_k(quanti, pos, qid, this->bits);
            assert(BIT_GET_k(quanti, pos, this->bits) == qid);
            e   = a - (zero + step * qid);  //+ step*0.5
            e_1 = std::max(e_1, e), e_0 = std::min(e_0, e);
            err += e * e;
            // if (i == 369 && row == 0) {
            //     DEBUG_HERE;
            // }
        }*/
    }
    delete[] tmpRow;
    err = sqrt(err / nRow / nCol);  // 0.00369
    e   = hTensor->disq.nrm_1 / nRow / nCol;
    // if (err >= e) {
    //     _WARN("[RTN]_Q<%d>_%s %g>%g\n", this->bits, isSinkNormal ? "normal" : "", err, e);
    // }
    return err;
}
float Distri_PIPE::Normal01(const float weight, int flag) {
    float normalized = (weight - zero) * scale;  // map weights to [-1, 1] range
    normalized       = std::clamp(normalized, -1.0f, 1.0f);
    return normalized;
}

static NF4_LUT LUT_nf4;
static NF3_LUT LUT_nf3;
/*
    1.  weight matrices (often symmetric) & activations (often non-negative & highly asymmetric)
*/
void Distri_PIPE::Prepare(int nQuant, int flag) {
    abs_max  = std::max(std::abs(vmin), std::abs(vmax));
    int QMAX = nQuant - 1, QMIN = 0;
    maxP = std::max(0.f, vmax), minN = std::min(0.f, vmin);
    auto table = nQuant == 16 ? LUT_nf4.table : LUT_nf3.table;
    float a    = 0;
    switch (asymmetry) {
        case ZERO_OFF: {
            assert(0);
            scale = (vmax - vmin) / static_cast<float>(QMAX - QMIN);
            // Calculate zero point: zp = qmin - round(min / scale)
            // This ensures: min ≈ (qmin - zp) * scale
            float zero_point_f = QMIN - std::round(vmin / scale);
            // Clamp zero point to quantized range
            zero = std::max(static_cast<float>(QMIN), std::min(static_cast<float>(QMAX), zero_point_f));
        } break;
        case PN_SCALE:
            scaleP = maxP > 0 ? maxP : 1.0f;
            scaleN = -minN > 0 ? -minN : 1.0f;
            for (int i = 0; i < nQuant; i++) {
                if (table[i] < 0) {
                    a = table[i] * scaleN;
                } else {
                    a = table[i] * scaleP;
                }
                codebook.push_back(a);
            }
            break;
        default:  // symmetric:  zero_point = 0,        Zero maps exactly to quantized value 0
            scale = (abs_max > 0) ? (1.0f / abs_max) : 1.0f;
            zero  = 0.f;
            for (int i = 0; i < nQuant; i++) {
                a = table[i] / scale;
                codebook.push_back(a);
            }
            break;
    }
}

BIT_8 Distri_PIPE::X2NormalF(int bits, const float weight, int flag) {
    int nBin       = codebook.size();
    float min_dist = std::numeric_limits<float>::max();
    BIT_8 best_idx = 0;
    for (BIT_8 i = 0; i < nBin; i++) {
        float dist = std::abs(weight - codebook[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    float e = fabs(weight - codebook[best_idx]);
    e_1 = std::max(e_1, e), e_0 = std::min(e_0, e);
    err += e * e;
    return best_idx;
}

// thread-safe single row quant on LUT
float GeQuant::_row_lut(int row, shared_ptr<GTensor> hTensor, const floatX* dat, float* tmpRow, int flag) {
    int nRow = hTensor->shape[0], nCol = hTensor->shape[1], nQuant = 0x1 << bits;
    floatGama *gamaRow = gama, *gamaCol = gamaRow + nRow;
    float sR = 1.0, sC = 1.0, a, a0 = 0;
    // if (row != 151644)            continue;
    Distri_PIPE disR;  //(Distri_PIPE::PN_SCALE);
    bool isColNorm = params.norm == NORMAL_MODE::SINKHORN;
    if (params.norm != NORMAL_MODE::NO_NORMAL)
        sR = gamaRow[row];
    for (int i = 0; i < nCol; i++) {
        if (isColNorm)
            sC = T2Float(gamaCol + i);
        a         = T2Float(dat + i) / sR / sC;
        tmpRow[i] = a;
        disR.Next(a);
    }

    floatGama* gama_ = gama + nRow + nCol + nQuant * row;
    disR.Prepare(nQuant);
    // gama_[0] = disR.zero, gama_[1] = 1.0 / disR.scale;    // to be consistent with gamaR
    assert(disR.codebook.size() == nQuant);
    for (int i = 0; i < disR.codebook.size(); i++) gama_[i] = Float2T<floatGama>(disR.codebook.data() + i);
    hBITARR quanti = quant_data + nCol * row * bits / 8;  // quanti[77641728]=0x00
    for (int i = 0; i < nCol; i++) {
        a       = tmpRow[i];
        int qid = disR.X2NormalF(bits, a);
        assert(qid >= 0 && qid < nQuant);
        BIT_SET_k(quanti, i, qid, bits);
        assert(BIT_GET_k(quanti, i, bits) == qid);
    }
    return disR.err;
}
/*
    1.  quantization bins are placed to be “information-theoretically optimal” for normally distributed values-N(0,1)
*/
float GeQuant::RT_NormalF(shared_ptr<GTensor> hTensor, const void* srcData, int flag) {
    assert(bits == 4 || bits == 3);
    double t0 = GST_ms(), err = 0;
    int nRow = hTensor->shape[0], nCol = hTensor->shape[1], nQuant = 0x1 << bits, nMostThread = omp_get_max_threads();
    const floatX *mat0 = (floatX*)(srcData), *dat = nullptr;
    floatX b;
    floatGama *gamaRow = gama, *gamaCol = gamaRow + nRow;
    float* tmpRows = new float[nCol * nMostThread];
    vector<double> err_rows(nRow);
#pragma omp parallel for
    for (int row = 0; row < nRow; row++) {
        int thread_id = omp_get_thread_num();
        err_rows[row] = _row_lut(row, hTensor, mat0 + row * nCol, tmpRows + nCol * thread_id, flag);
    }
    delete[] tmpRows;
    for (int row = 0; row < nRow; row++) err += err_rows[row];
    err     = sqrt(err / nRow / nCol);  // 0.00282
    float e = err / (hTensor->disq.nrm_1 / nRow / nCol);
    return err;
}

template class Quantizer<float>;
template class Quantizer<bf16>;

FeatsOnFold* Mat2DORT(LiteBOM_Config config, ExploreDA* edaX, typNUMBER tpN, void* hData, int nMostFeat, int nSamp, DORT_wrap* dort, int flag);

/*
    It's really a suprize to find that GBDT would improve quant.    cys 11/22/2025
*/
template <typename T>
float Q_Impurity<T>::LowBit_GBDT(shared_ptr<GTensor> hTensor, const void* srcData, int flag) {
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
        FeatsOnFold* hFold = Mat2DORT(config, dort->hEDA_train, hTensor->type, (void*)srcData, nFeat_0, nSamp, dort, flag | FeatsOnFold::DF_TRAIN);
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

        // hTensor->BeforeQuant(this);
        // H2D(hTensor->data, this->quant_data, this->szQuant);
        // H2D(hTensor->gama_T(), this->gama, sizeof(floatGama) * (nRow + nCol + nRow * nQuant));
        // hTensor->Print(hTensor->name, 0x0, -1);
        // delete[] this->gama;
        delete hFold;
        delete dort;

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

/*
    1. errQ = err/average   (average = disq.nrm_1 / nRow / nCol);
*/
float GeQuant::LowBit(shared_ptr<GTensor> hTensor, const void* srcData, int flag) {
    int nRow = hTensor->shape[0], nCol = hTensor->shape[1], nTry = 0;  // 4-bit is enough
    int nBit0 = params.default_bits;
    // std::vector<GeQuant::_q_sweep> methods = {{NORMAL_MODE::ROW_01, 4, 1}};  //{0,2} ,{0,8},{true, 2}, {false, 3, 1}, {false, 4}
    std::vector<GeQuant::_q_sweep> methods = {{NORMAL_MODE::NO_NORMAL, nBit0, 0}};
    float errQ = FLT_MAX, err, average = hTensor->disq.nrm_1 / nRow / nCol;

    for (auto method : methods) {
        bits        = method.bits;
        params.norm = method.normal;
        params.type = method.mi == 0 ? (params.isNormalFloat ? QUANT_MODE::RTNf : QUANT_MODE::RTN) : QUANT_MODE::MINI;
        assert(bits <= params.default_bits);
        method.szQuant = hTensor->size() * bits / 8;
        err            = Core(hTensor, srcData, gama, flag);  // 0.021420
        err /= average;
        if (err < errQ) {
            errQ  = err;
            best_ = method;
            memcpy(best_gama, gama, sizeof(floatGama) * nMostGama);
            memcpy(best_quant, quant_data, method.szQuant);
            if (errQ < params.T_errQ)
                break;
        }
        nTry++;
    }
    int nQuant = 1 << best_.bits, nGama = nRow * nQuant + nRow + nCol;
    assert(nGama <= nMostGama);
    bits = best_.bits;
    hTensor->BeforeQuant(this);
    assert(hTensor->size() <= nzMost);  // K.w->size()<Q.w->size()
    H2D(hTensor->data, best_quant, best_.szQuant);
    H2D(hTensor->gama_T(), best_gama, sizeof(floatGama) * nGama);
    // FREE_a(gama);

    //_INFO("<QUANT_%d> @\"%s\" t=%.3gms\t \n", bits, hTensor->name, GST_ms() - t0);
    impurity = errQ;                               // may update if isCheckErr=true
    GeQuant::AfterLowBit(hTensor, srcData, flag);  // may update impurity
    return impurity;
}

// the shape&type of srcDat is defined in hTensor
template <typename T>
float Q_Impurity<T>::Core(shared_ptr<GTensor> hTensor, const void* srcData, floatGama* curGama, int flag) {
    // return LowBit_GBDT(hTensor, srcData, flag);
    double t0 = GST_ms(), a0 = DBL_MAX, a1 = 0;
    // assert(hTensor->isWMAT());
    int nRow = hTensor->shape[0], nCol = hTensor->shape[1], minLeaf = 2, nSplit = -1, nQuant = 0x1 << this->bits;  // hFold->config.num_trees;
    const floatX* mat0 = (floatX*)(srcData);
    floatX* mat1       = (floatX*)mat0;
    if (this->params.norm != NORMAL_MODE::NO_NORMAL) {
        mat1 = new floatX[nRow * nCol];
        memcpy(mat1, mat0, sizeof(floatX) * nRow * nCol);
    }
    this->isCheckErr = false;
    switch (this->params.norm) {
        case NORMAL_MODE::SINKHORN:
            this->SinkNormal(hTensor, srcData, curGama, flag);  // sinknormal may fail, so all rs & cs are 1.0
            break;
        case NORMAL_MODE::ROW_01:
            this->Normal_ROW01(hTensor, mat1, curGama, flag);
            break;
        default:
            break;
    }
    if (this->params.type == QUANT_MODE::MINI) {
        dort                   = static_cast<DORT_wrap*>(LiteMORT_init(nullptr, 0, nullptr, 0x0));
        LiteBOM_Config& config = dort->config;
        config.feat_quanti = 1024, config.verbose = 1000, config.objective = "quant";
        // return this->imbalance;  // only for debug
        FeatsOnFold* hFold = Mat2DORT(dort->config, dort->hEDA_train, hTensor->type, mat1, nRow, nCol, dort, flag | FeatsOnFold::DF_TRAIN);

        // #pragma omp parallel for
        for (int id = 0; id < nRow; id++) {
            auto hFeat       = hFold->feats[id];
            floatGama* gama_ = curGama + nRow + nCol + nQuant * id;
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
            a1 = std::max(hFeat->wGain, a1), a0 = std::min(hFeat->wGain, a0);  // this->impurity += hFeat->wGain;
            this->impurity += hFeat->errQ;
        }
        this->impurity = sqrt(this->impurity / nRow / nCol);
        _INFO("<QUANT_MI_%d>@%s nF=%d(%ld) impurity=%g[%g-%g]\t split=%g t=%.5gms\n", this->bits, "", nRow, nCol, this->impurity, a0, a1, nSplit * 1.0 / nRow,
              GST_ms() - t0);
        delete hFold;
        delete dort;
    } else {
        this->impurity = Quantizer<T>::RTN(hTensor, srcData, flag);
    }
    if (mat1 != srcData)
        delete[] mat1;
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

template <typename T>
Q_AWQ<T>::Q_AWQ(const std::string& nam_, void* hN, QUANT_CARD& param_, int flag) : Quantizer<T>(nam_, hN, param_, flag) {
    assert(this->bits > 0 && this->bits <= 8);
    this->params.TransA = 0;
}
template <typename T>
float Q_AWQ<T>::AfterLowBit(shared_ptr<GTensor> tensor, const void* cpuData, int flag) {
    //  AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    return 0.0;
}
template class Q_AWQ<bf16>;
template class Q_AWQ<float>;

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

// no need quant of __nv_fp8_e5m2 !
template <>
double G_Scale_RC(__nv_fp8_e5m2* mat, int nRow, int nCol, double* row_scal, double* col_scal, double T_imb, int flag) {
    assert(0 && "no need quant of __nv_fp8_e5m2 !");
    return 0.0;
}

//  set the group size to 64, batch-size to 8
float GeQuant::SinkNormal(shared_ptr<GTensor> hTensor, const void* srcData, floatGama* curGama, int flag) {
    double t0   = GST_sec();
    size_t nEle = hTensor->size(), dGrid = CEIL_DIV(nEle, CU_T4B_MIDDLE);
    int nRow = hTensor->shape[0], nCol = hTensor->shape[1], i = 0, loop = 0;
    double imb = 0.0, imb0 = 0.0, clip_min = 0.001, clip_max = 1000, off = 0, len = hTensor->disq.sum_2, len1 = 0, a, gs_0 = DBL_MAX, gs_1 = -DBL_MAX;
    imbalance     = FLT_MAX;  //
    bool isVerify = true;
    /**/
    if (srcData == nullptr) {  //  GPU version
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
        const floatX* mat0 = (floatX*)(srcData);
        floatX *mat1       = new floatX[nRow * nCol], b;
        memcpy(mat1, mat0, sizeof(floatX) * nRow * nCol);
        // _INFO("SinkNormal A=[%dx%d]=%g \t@\"%s\"\n", nRow, nCol, len, hTensor->name);  // return 0.0;
        double *rowStdDev = new double[(nRow + nCol) * 2], *colStdDev = rowStdDev + nRow, *rowS = colStdDev + nCol, *colS = rowS + nRow;
        for (int r = 0; r < nRow; r++) rowS[r] = 1.0;
        for (int c = 0; c < nCol; c++) colS[c] = 1.0;
        for (loop = 0; loop < this->nMostLoop; loop++) {
            for (int r = 0; r < nRow; r++) {
                rowStdDev[r] = G_StdDev(nCol, mat1 + r * nCol, 1);
            }
            for (int c = 0; c < nCol; c++) {
                colStdDev[c] = G_StdDev(nRow, mat1 + c, nCol);
            }
            imb = G_Scale_RC(mat1, nRow, nCol, rowStdDev, colStdDev, T_imbal);
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
        hTensor->disq.imbalance = imbalance;
        // copy to curGama
        floatGama *gamaR = curGama, *gamaC = gamaR + nRow;
        for (int r = 0; r < nRow; r++) {
            gamaR[r] = rowS[r];
            gs_0 = std::min(gs_0, rowS[r]), gs_1 = std::max(gs_1, rowS[r]);
        }
        for (int c = 0; c < nCol; c++) {
            gamaC[c] = colS[c];
        }
        /*if (isVerify) {
            G_Scale_RC(mat1, nRow, nCol, gamaR, gamaC, T_imbal);
            // G_Scale_RC(mat1, nRow, nCol, rowS, colS);
            for (int i = 0; i < nRow * nCol; i++) {
                b = mat1[i] - mat0[i];
                a = T2Float(&b), off += a * a;
                a = T2Float(mat1 + i), len1 += a * a;
            }
            off = sqrt(off / len);  // sqrt(off / (nRow * nCol));
        }*/
        delete[] rowStdDev;
        delete[] mat1;
    }  //  SinkNormal A=[1024x3072] loop=10 imbalance=1.0078 err=18.7696   @"model.layers.19.mlp.down_proj.weight"

    _LOG(off > 1.0 ? DUMP_WARN : DUMP_INFO, "SinkNormal A=[%dx%d] loop=%d imbalance=(%g->%g) err=%.4g(%g) T=%.3g \t@\"%s\"\n", nRow, nCol, loop, imb0,
         imbalance, off, len, GST_sec() - t0, hTensor->name);

    return imbalance;
}

float GeQuant::Normal_ROW01(shared_ptr<GTensor> hTensor, void* srcData, floatGama* curGama, int flag) {
    double t0   = GST_sec();
    size_t nEle = hTensor->size();
    int nRow = hTensor->shape[0], nCol = hTensor->shape[1], i = 0, loop = 0, nQuant = 0x1 << bits;
    double imb = 0.0, imb0 = 0.0, clip_min = 0.001, clip_max = 1000, off = 0, len = hTensor->disq.sum_2, len1 = 0, a, gs_0 = DBL_MAX, gs_1 = -DBL_MAX;
    floatX *mat0 = (floatX*)(srcData), *dat = nullptr;

    floatGama *gamaR = curGama, *gamaC = gamaR + nRow;
    for (int r = 0; r < nRow; r++) {
        Distri_PIPE disR;
        dat = mat0 + r * nCol;
        for (int c = 0; c < nCol; c++) {
            a = T2Float(dat + c);
            disR.Next(a);
        }
        disR.Prepare(nQuant);
        gamaR[r] = a = 1.0 / disR.scale;
        gs_0 = std::min(gs_0, a), gs_1 = std::max(gs_1, a);
        for (int c = 0; c < nCol; c++) {
            a       = T2Float(dat + c);
            float b = disR.Normal01(a);  //  (weight - zero) * scale;       //0.003433=>0.039
            dat[c]  = Float2T<floatX>(&b);
        }
    }
    isCheckErr = true;  // to get errQ
    // _LOG(off > 1.0 ? DUMP_WARN : DUMP_INFO, "Normal_ROW01 A=[%dx%d] loop=%d imbalance=(%g->%g) err=%.4g(%g) T=%.3g \t@\"%s\"\n", nRow, nCol, loop, imb0,
    //      imbalance, off, len, GST_sec() - t0, hTensor->name);

    return imbalance;
}

/*
"quantization_config": {
    "bits": 4,
    "group_size": 128,
    "modules_to_not_convert": null,
    "quant_method": "awq",
    "version": "gemm",
    "zero_point": true
  },*/
void QUANT_CARD::Init4Neuron(const std::string& name, const JSON& jQuant, int flag) {
    type = NO_QUANT;
    if (jQuant.find("VendorQuant") != jQuant.end()) {
        isVendorQuant = true;
        nPassLayer    = -1;  // no pass layer in QWen3
    }
    string s0 = "";
    for (JSON::const_iterator it = jQuant.begin(); it != jQuant.end(); ++it) {
        auto k = it.key();
        if (!k.empty() && k[0] == '#')
            continue;
        if (k == "debug") {
            continue;
        }
        if (G_Has_(name, {k})) {
            const JSON& jQ = it.value();
            string info;
            info         = jKV(jQ, {"quant_method"}, info, false);
            default_bits = jKV(jQ, {"bits"}, default_bits, false);
            T_group      = jKV(jQ, {"group_size"}, T_group, false);
            assert(T_group > 0 && T_group < 10240);
            isZeroPoint = jKV(jQ, {"zero_point"}, isZeroPoint, false);
            // quant.filter_MIQ        = jKV_arr(jConfig, {"quantizer", "MINI"}, quant.filter_MIQ, false);
            // quant.filter_WeightF8Ex = jKV_arr(jConfig, {"quantizer", "F8Ex"}, quant.filter_WeightF8Ex, false);
            if (default_bits != 4 && default_bits != 3 && default_bits != 8) {
                default_bits = 4;
                assert(0);
            }
            tpScale = typNUMBER::BF16;
            if (isZeroPoint)
                tpZero = default_bits == 4 ? typNUMBER::Q4 : default_bits == 3 ? typNUMBER::Q3 : typNUMBER::Q2;
            tpQWeight        = default_bits == 4 ? typNUMBER::Q4 : default_bits == 3 ? typNUMBER::Q3 : typNUMBER::Q2;
            string norm_type = jKV(jQ, {"normal"}, s0);
            norm             = G_Aa(norm_type, "SINKHORN") ? NORMAL_MODE::SINKHORN : NORMAL_MODE::NO_NORMAL;

            if (info == "awq") {
                type =
                    RTN_ZS;  // awq is the quant method(find 1% salient/outlier weights by activataion), and the qdata format is still RTN(round to nereast int)
            } else {
                type = default_bits == 8 ? F8Ex : MINI;
            }
        }
    }
}
