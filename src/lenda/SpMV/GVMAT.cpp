#include <memory>
#include <iostream>
#include <fstream>
#include "GVMAT.h"
#include "Matrix.hpp"
#include "../util/BLAS_t.hpp"	


static const int inc_1=1;
static char trans[]={'N','T'};

void Unitary(hGVEC &mA, int flag)	{
	double norm = mA->Nrm2(flag);
	mA->Scald(1.0/norm, flag);
}




