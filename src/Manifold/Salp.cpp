#include "Fish.hpp"

    // LogicSalp::LogicSalp(const int dim, int flag) {
    // 	position.resize(dim);
    // }

    // LogicSalp::LogicSalp(const int dim, const vector<int>&picks, int flag) {
    // 	position.resize(dim);
    // 	for (int i = 0; i < dim; i++) {
    // 		position[i] = 0;
    // 	}
    // 	for (auto x : picks) {
    // 		position[x] = 1;
    // 	}
    // }

LogicSalp::LogicSalp(const std::string& nam_, struct CLI_params params,tpSWARM& swarm_,int flag) : swarm(swarm_)    {

}

void LogicSalp::Train(int flag)       {
    assert(swarm.size()>=1);
    swarm[0]->Train(flag);
}
    
void LogicSalp::cross_over(const LogicSalp*A, const LogicSalp*B, int flag) {
	assert(space == BIT_MASK);
	int DIM = position.size(),i;
	int pos = 0;    //rander_.RandInt32() % DIM;
	for (i = 0; i < DIM; i++) {
		position[i] = i < pos ? A->position[i] : B->position[i];
	}


}
void LogicSalp::mutatioin(double T_mut, int flag) {
	assert(space == BIT_MASK);
	int DIM = position.size(),i;
	for (i = 0; i < DIM; i++) {
		double p = 0;   //rander_.Uniform_(0, 1);
		if (p < T_mut) {
			position[i] = position[i] == 1 ? 0 : 1;
		}	
	}

}