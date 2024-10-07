#include "GST_util.hpp"

int GST_util::dump=10;
double GST_util::tX=0.0;
int GST_util::verify=0;

#ifdef _GST_MATLAB_
#pragma message("\t\tMATLAB:	R2012b \r\n")
#include "mat.h"
#include "matrix.h"
#pragma comment( lib, "G:\\MATLAB\\R2011a\\extern\\lib\\win32\\microsoft\\libmat.lib" )
#pragma comment( lib, "G:\\MATLAB\\R2011a\\extern\\lib\\win32\\microsoft\\libmx.lib" )

int GST_util::LoadDoubleMAT( const char *sPath,const char* sVar,int *nRow,int *nCol,double **val,int flag )	{
	MATFile *pMat;
	mxArray *problem,*A;
	mxClassID cls;
	double *Ax, *Az;
	int nz,*Ap,*Ai;
	
	pMat = matOpen( sPath, "r");
	if (pMat == NULL) {
		printf("Error reopening file %s\n", sPath );
		return -101;
	}
	problem = matGetVariable(pMat, sVar );
	if (problem == NULL) {
		printf("Error reading existing matrix Problem\n");
		return -102;
	}
	cls = mxGetClassID(problem) ;
	*nRow=mxGetM (problem ) ;		*nCol=mxGetN ( problem ) ;
	if ( *nRow<=0 || *nCol<=0 ) {
		printf("Error get element A\n");
		return -103;
	}
	Ax = mxGetPr(problem) ;			Az = mxGetPi (problem);
	nz = (*nRow)*(*nCol);
	*val = new double[nz];
	memcpy( *val,Ax,sizeof(double)*nz );
	mxDestroyArray(problem);
	if (matClose(pMat) != 0) {
		printf("Error closing file %s\n",sPath );
		return -104;
	}

	return 0x0;
}
#endif