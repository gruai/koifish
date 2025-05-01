/**
 *  SPDX-FileCopyrightText: 2018-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief FeatVector
 *  \author Yingshi Chen
 */

#include "./FeatVector.hpp"
using namespace Grusoft;

FeatVector::~FeatVector() {
	if (select_bins != nullptr)
		delete select_bins;
	if (wBins != nullptr)
		delete[] wBins;
	if (distri_ != nullptr && distri_->histo==nullptr) {		//delete in EDA
		delete distri_;
	}
}

void  FeatVector::SetDistri(Distribution*d_, int flag) {
	assert(distri_ == nullptr);
	distri_ = d_;
	distri_->nam = this->nam;
	distri_->type = this->type;
}

//�μ�ExploreDA::AddDistri
void FeatVector::UpdateType(int flag){
	assert(PY != nullptr);
	if (PY->isCategory()) {
		//if(distri_!=nullptr)	//expanded feat has no hDistri
		//	BIT_SET(distri_->type, Distribution::CATEGORY);
		BIT_SET(type, Distribution::CATEGORY);
	}
	if (PY->isDiscrete()) {
		//if (distri_ != nullptr)
		//	BIT_SET(distri_->type, Distribution::DISCRETE);
		BIT_SET(type, Distribution::DISCRETE);
	}
	if (PY->representive > 0) {
		//if (distri_ != nullptr)
		//	BIT_SET(distri_->type, Distribution::DISCRETE);
		BIT_SET(type, Distribution::DISCRETE);
		BIT_SET(type, FeatVector::REPRESENT_);
	}
}
/*

void FeatVec_EXP::Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_0, HistoGRAM* histo, int nMostBin, const tpSAMP_ID *samps4quanti, int flag0)	const {
	//SAMP_SET samp_1;
	//size_t nRightSamp = hRight->nSamp();
	//samp_1.Alloc(nRightSamp);
	hRight->Samp2Histo(hData_, samp_0, histo, nMostBin, hLeft->samp4quanti, flag0);
}

void FeatVec_EXP::Value_AtSamp(const SAMP_SET*samp_set, void *samp_values, int flag){
	hLeft->Merge4Quanti(samp_set,0x0);
	SAMP_SET samp1(samp_set->nSamp,hLeft->samp4quanti);	
	hRight->Value_AtSamp(&samp1, samp_values);
	//hRight->SplitOn(hData_, hBest, flag);
}	


void FeatVec_EXP::SplitOn(FeatsOnFold *hData_, MT_BiSplit *hBest, int flag) {
//hRight->Merge4Quanti(nullptr, 0x0);
	//void *hOldVal = hRight->ExpandOnSamp(hLeft->samp4quanti);
	//hRight->SplitOn(hData_, hBest, flag);
	//hRight->CloseOnSamp(hOldVal,hLeft->samp4quanti);	
	hRight->SplitOn(hData_, hBest, flag);
}*/
