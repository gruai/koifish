#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <assert.h>
#include <time.h>

#include "../util/GST_def.h"
#include "../util/samp_set.hpp"
#include "../../Utils/GST_util.hpp"
#include "../data_fold/Histogram.hpp"
#include "../data_fold/Binfold.hpp"
#include "../learn/Regression.hpp"

using namespace std;
#if (defined _WINDOWS) || (defined WIN32)
//���߳�crash����������
	#include <Windows.h>
	#define ATOM_INC_64(a)		InterlockedIncrement64(&(a))	
#else
#endif

//���ڸ�������ԭ�򣬺�С�ĸ���Ҳ��0
double FLOAT_ZERO(double a, double ruler);

/*
	WeakLearner��������ʵΪBiSplit
*/
namespace Grusoft {

	class FeatVector;
	class FeatsOnFold;
	class BoostingForest;		//BoostingForest��Ҫ����ΪBOModel
	class ManifoldTree;

	/*
		ѵ��ʱд�룬train,eval,predict�����ȡ��ֵ�����Ҹ�ֵ�в�ͬ�ı�ʾ
		tpFRUIT_Ӧ��Ϊģ����
		SetSplitInfo��Ҫ������ƣ�����
	*/


	class MT_BiSplit {
	protected:
		//FeatsOnFold *hData=nullptr;
		virtual double AGG_CheckGain(FeatsOnFold *hData_, FeatVector *hFeat, int flag = 0x0);
		virtual int PickOnGain(FeatsOnFold *hData_, const vector<FRUIT *>& arrFruit, int flag = 0x0);
		const BoostingForest *hBForest=nullptr;
	public:
		static double tX;
		tpDOWN down_step;	//ManifoldTree��Ŀ�꺯����negative_gradient
		double lr_eta=1.0;		//adaptive learning rate
		//HistoGRAM histo;
		Regression *regression = nullptr;
		BinFold *bsfold=nullptr;
		FRUIT *fruit = nullptr;		//only ONE!!!		thrsh at fork, Y value at leaf
		std::string sX;

		int id=-1;				//Ϊ�˽�ʡ�ڴ棬��������fold�ᳬ��65536
		//float *distri = nullptr;
		SAMP_SET samp_set;
		MT_BiSplit	*left = nullptr,	*right = nullptr;
		MT_BiSplit	*parent = nullptr,	*brother = nullptr;
		//size_t nzSamp = 0;
		int feat_id = -1, feat_regress = -1, depth = -1;
		double gain_train = 0,  gain_=0;	//gain_train����׼ȷ	gain_
		double confi = 0, devia = 0;
		union {
			double impuri = DBL_MAX;
			double score ;
		};
		double Y_mean = 0, Y2_mean = 0, G2_sum = 0, G_sum = 0, H_sum = 0;

		MT_BiSplit(FeatsOnFold *hData_, const BoostingForest *hBoosting, int d, int rnd_seed, int flag = 0x0);
		MT_BiSplit() : feat_id(-1) { ; }
		MT_BiSplit(MT_BiSplit *hDad, int flag = 0x0) {
			hBForest = hDad->hBForest;
			depth = hDad->depth + 1;
			assert(hDad->feat_id != -1);
			feat_regress = hDad->feat_id;
			parent = hDad;
		}
		virtual ~MT_BiSplit() {
			if (regression != nullptr)
				delete regression;
			if (fruit != nullptr)
				delete fruit;
			//if (distri != nullptr)
			//	delete distri;
		}

		virtual size_t nSample() { return samp_set.nSamp; }
		bool isLeaf() const {
			return left == nullptr && right == nullptr;
		}
		//H_HistoGRAM H_HISTO;
		HistoGRAM *GetHistogram(FeatsOnFold *hData_, int pick,bool isInsert, int flag = 0x0);

		//����ֻ��ĳ�������Ĺ۲�ֵ!!!
		virtual void Observation_AtLocalSamp(FeatsOnFold *hData_, int flag = 0x0);
		virtual double CheckGain(FeatsOnFold *hData_, const vector<int> &pick_feats, int x, int flag = 0x0);
		virtual double GetGain(int flag = 0x0);
		virtual void BeforeTrain(FeatsOnFold*, int flag = 0x0) { throw "MT_BiSplit::BeforeTrain is ..."; }
		//virtual void SetSamp_(FeatsOnFold *hData_);
		virtual void Dump(FeatsOnFold *hData_, const char*, int type, int flag = 0x0);
		static bool isBig(const MT_BiSplit *l, const MT_BiSplit *r) { return true; }/**/

		virtual tpDOWN GetDownStep();

		virtual void Init_BFold(FeatsOnFold *hData_,int flag=0x0);

		template<typename Tx>
		inline void  _core_1_(bool isQuanti, const tpSAMP_ID samp, const Tx&a, const double thrsh, tpSAMP_ID*left, G_INT_64&nLeft, tpSAMP_ID*rigt,G_INT_64&nRigt, int flag) {
			bool isNana = (isQuanti && a == -1) || (!isQuanti && IS_NAN_INF(a));
			if (isNana) {
				if (fruit->isNanaLeft) {
					left[nLeft++] = samp;	
					//samps[ATOM_INC_64(nLeft)] = samp;	continue;
				}
				else {
					rigt[nRigt++] = samp;	
					//rigt[ATOM_INC_64(nRigt)] = samp;	continue;
				}
			}	else {
				if (a < thrsh) {
					left[nLeft++] = samp;
					//samps[ATOM_INC_64(nLeft)] = samp;
				}
				else {
					rigt[nRigt++] = samp;
					//rigt[ATOM_INC_64(nRigt)] = samp;
				}
			}
		}
		
		//ע�⣬samp_vals��samp_setһһ��Ӧ
		template<typename Tx>
		//void SplitOn(FeatsOnFold *hData_, size_t nSamp_, const Tx* vals, bool isQuanti, int flag = 0x0) {
		void SplitOn(FeatsOnFold *hData_,const Tx* samp_vals, bool isQuanti, int flag) {
			GST_TIC(t1);
			SAMP_SET& lSet = left->samp_set,& rSet = right->samp_set;
			lSet = this->samp_set;		rSet = this->samp_set;
			lSet.isRef = true;			rSet.isRef = true;
			size_t i, nSamp = samp_set.nSamp, step;
			tpSAMP_ID *samps = samp_set.samps, *rigt = samp_set.rigt, *left = samp_set.left;
			//double thrsh = isQuanti ? fruit->T_quanti : fruit->thrshold;
			double thrsh = fruit->Thrshold(isQuanti);
			//clock_t t1 = clock();
			int num_threads = OMP_FOR_STATIC_1(nSamp, step);
			G_INT_64 *pL = new G_INT_64[num_threads](), *pR = new G_INT_64[num_threads](),nLeft=0,nRigt=0;
#pragma omp parallel for schedule(static,1)
			for (int th_ = 0; th_ < num_threads; th_++) {
				size_t start = th_*step, end = min(start + step, nSamp),i;
				if (end <= start)
				{		continue;				}
				G_INT_64	nL=0,nR=0;
				for (i = start; i < end; i++) {
					tpSAMP_ID samp = samps[i];
					_core_1_(isQuanti, samp, samp_vals[i], thrsh, samps + start, nL, rigt + start, nR, flag);
					//_core_1_(isQuanti, samp, vals[samp], thrsh, samps+start, nL, rigt+start, nR, flag);
				}
				pL[th_] = nL;	 pR[th_] = nR;
				assert(pL[th_]+ pR[th_]== end-start);
			}
			for (int th_ = 0; th_ < num_threads; th_++) {
				size_t start = th_*step, end = min(start + step, nSamp);
				if (end <= start)		{	continue;		}
				memcpy(samps + nLeft, samps+start, sizeof(tpSAMP_ID)*pL[th_]);
				nLeft += pL[th_];
			}
			for (int th_ = 0; th_ < num_threads; th_++) {
				size_t start = th_*step, end = min(start + step, nSamp);
				if (end <= start)		{		continue;		}				
				memcpy(samps + nLeft+ nRigt, rigt + start, sizeof(tpSAMP_ID)*pR[th_]);
				nRigt += pR[th_];
			}
			//nLeft++;							nRigt++;
			//memcpy(samps, samp_set.left, sizeof(tpSAMP_ID)*nLeft);

			samp_set.nLeft = nLeft;				samp_set.nRigt = nRigt;
			assert(nLeft + nRigt == nSamp);
			//tX += ((clock() - (t1))*1.0f / CLOCKS_PER_SEC);
			lSet.samps = samps;				lSet.nSamp = nLeft;
			//std::sort(lSet.samps, lSet.samps + lSet.nSamp);
			rSet.samps = samps + nLeft;		rSet.nSamp = nRigt;
			//std::sort(rSet.samps, rSet.samps + rSet.nSamp);
			delete[] pL;		delete[] pR;
			//FeatsOnFold::stat.tX += GST_TOC(t1);
		}
	};
	typedef MT_BiSplit *hMTNode;
	typedef std::vector<MT_BiSplit*>MT_Nodes;

}

