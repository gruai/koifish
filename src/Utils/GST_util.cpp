#include <filesystem> // C++17
#include "GST_util.hpp"

int GST_util::dump=10;
double GST_util::tX=0.0,GST_util::tX1=0.0;
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

//	std::filesystem::exists.
extern inline FILE *fexist_check(const char *path, const char *mode, const char *file, int line) {
    FILE *fp = fopen(path, mode);
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr, "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    return fp;
}

std::string FILE_EXT(const std::string&path)	{
    std::filesystem::path filePath = path;
    return filePath.extension();
}	