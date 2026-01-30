
/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief PIPE - transfer data between device & host
 *  \author Yingshi Chen
 */
#include "Pipe.hpp"
#include "../Utils/GST_util.hpp"

template struct PIPE_Muon<floatX, floatMV>;   // Force compilation
template struct PIPE_Adamw<floatX, floatMV>;  // Force compilation

template <typename Tp, typename Tmv>
void PIPE_Muon<Tp, Tmv>::Update(GTensor* tensor_, float wd, float _grad_scale, unsigned int _seed, int flag) {
    PIPE_Adamw<Tp, Tmv>::Update(tensor_, wd, _grad_scale, _seed, flag);
    bool isAdamw = muon.isAdamW(this->tensor);
    if (isAdamw) {
        assert(this->learning_rate == this->hOPT->LearningRate());
    } else {
        this->learning_rate *= muon.lr_scale;  // automatic learning rate transfer
        // int64_t m = this->ne[0], n = this->ne[1];
        int64_t m = this->ne[1], n = this->ne[0];
        isTrans = m > n && muon.isTransDown; 
        isTrans = false;
        switch(muon.tpDecay){
        case 0:
            this->weight_decay = 0;
            break;
        case 1:
            this->weight_decay /= muon.lr_scale;
            break;
        default:
            break;
        }
       
        dimA    = isTrans ? n : m;
        mG      = (Tmv*)this->tensor->gm;
        assert(GTensor::buff != nullptr);
        size_t offset = 0;
        A             = (Tmv*)((char*)(GTensor::buff) + offset), offset += sizeof(Tmv) * dimA * dimA;
        B             = (Tmv*)((char*)A + offset), offset += sizeof(Tmv) * dimA * dimA;
        BX            = (Tmv*)((char*)B + offset), offset += sizeof(Tmv) * tensor_->size();
        X             = (Tmv*)((char*)BX + offset), offset += sizeof(Tmv) * tensor_->size();
        if (isTrans) {
            Xt = (Tmv*)((char*)X + offset), offset += sizeof(Tmv) * tensor_->size();
        }
        assert(offset <= GTensor::buff_len);
        assert(PTR_is_aligned(A));
        assert(PTR_is_aligned(B));
        assert(PTR_is_aligned(BX));
        assert(PTR_is_aligned(X));
    }
    // assert(this->tensor->gv == nullptr);
}
