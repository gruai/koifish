#pragma once
#include <vector>
#include <map>  
#include <algorithm>
#include <cmath>
#include "../util/samp_set.hpp"
#include "GST_rander.hpp"

namespace Grusoft {
	

	//�ε��������ռ���ĺ���(only true or false)
	struct LogicSalp {
		static GRander rander_;
		typedef enum {
			BIT_MASK
		}SPACE_TYPE;
		SPACE_TYPE space=BIT_MASK;

		int x=0;
		float fitness;	//greater fitness will have a greater probability of being selected for recombination.
		vector<double> position;
		LogicSalp(const int dim, int flag = 0x0);
		LogicSalp(const int dim, const vector<int>&picks, int flag = 0x0);
		//LogicSalp(const vector<bool>&pick_mask, int flag = 0x0);		//������boost�е� integer_mask
		int DIM() const		{ return position.size(); }

		virtual void Copy(const LogicSalp*src,int flag=0x0) {
			position = src->position;
			fitness = src->fitness;
			x = src->x;
		}

		//aA+b*B
		virtual void MixPosition(double alpha, const LogicSalp*A, double beta, const LogicSalp*B, int flag) {
			int dim = position.size(), i;
			for (i = 0; i < dim; i++) {
				position[i] = alpha*A->position[i] + beta*B->position[i];
			}
		}

		virtual void cross_over(const LogicSalp*A,  const LogicSalp*B,int flag=0x0);
		virtual void mutatioin(double T_mut,int flag=0x0);
	};

	//special swarm algortim on GBDT trees
	class Feature_Selection {

	protected:
		static GRander rander_;
		typedef enum {
			SIGMOID
		} TRANSFER_FUNC;

		int DIM_, iter = 0, maxIter;		// the current iteration and the maximum number of iterations
		double T_mute = 0.01;
		double T_pressure = 1.5;		//selective pressure
		int T_elitism = 2;				//Elitism selection 
		int nMostSalp = 0;
		vector<LogicSalp*>salps;
		LogicSalp *food = nullptr, *leader = nullptr;

		virtual vector<int> roulette_sample(int nPick,vector<float> roulette, int flag = 0x0);
	public:
		LogicSalp cand;

		Feature_Selection(int nMostSalp_,int dim_, int flag = 0x0);
		virtual ~Feature_Selection() {

		}

		int DIM() { return DIM_; }
		bool isFull() 
		{		return salps.size() >= nMostSalp;	}


		virtual void AddSalp(int dim, const vector<int>&picks, int x_, int flag = 0x0);
		virtual void AddCandSalp(int flag = 0x0);

		virtual void SetCost(double cost, int flag = 0x0) {
			assert(salps.size()>0);
			LogicSalp *salp = salps[salps.size() - 1];
			salp->fitness = cost;
		}

		virtual bool Step(int nSalp,int flag = 0x0)			{	throw "!!!Feature_Selection Step is ...!!!";				}
		//isMask(True-picks[i] = salp->position[i] == 1)
		virtual bool PickOnStep(int nSalp, vector<int>&picks,bool isMask, int flag = 0x0)	{	throw "!!!Feature_Selection SubStep_1 is ...!!!";		}

		virtual void GetPicks(const LogicSalp *salp,vector<int>&picks,bool isMask, int flag = 0x0);

	};

	class FS_gene_ : public Feature_Selection {
	protected:
		vector<LogicSalp*> inter_samps_;
		LogicSalp  offspring;
		//virtual void UpdateLeader(double loss, int flag = 0x0);
		virtual void Intermediate_Select(int flag=0x0);
	public:
		FS_gene_(const string nam_,int nBird_, int dim_, int nMaxIter_,int flag = 0x0);
		virtual bool PickOnStep(int nSalp, vector<int>&picks,bool isMask, int flag = 0x0);
	};


}