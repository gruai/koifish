#include "GST_bmp.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <tchar.h>
#include <xstring>
using namespace std;
wstring GST_bitmap::sDumpFolder;

//位操作
#define BIT_SET( val,flag ) ((val) |= (flag))	
#define BIT_RESET( val,flag ) ((val) &= (~(flag)) ) 
#define BIT_TEST( val,flag ) (((val)&(flag))==(flag))
#define BIT_IS( val,flag ) (((val)&(flag))!=0)
/*
int SaveColorBMP( BIT_32 * pixel32s, int width, int height, int bpp,char *bmpName )	{
	GST_bitmap bmp;
	bmp.Create( width,height );			
	int r,c,nz=0;
	for( r = 0 ; r < height; r++ )	{
	for( c = 0 ; c < width; c++ )	{
		bmp.SetPixel( c,r,pixel32s[nz++] );
	}
	}
	bmp.Save( _T("F:\\GiSeek\\trace\\Dataset\\1.bmp") );
	return 1;
}*/

#include "F:/GIPack/src/CxImage/ximage.h""
int SaveColorBMP( BIT_32 * pixel32s, int width, int height, int bpp,std::wstring sPath )	{
	bool bSave=false;
	CxImage bmp;
	bmp.Create( width,height,24,CXIMAGE_FORMAT_BMP );		
	if( bmp.IsValid( ) )	{
		int r,c,nz=0;	
		for( r = 0 ; r < height; r++ )	{
		for( c = 0 ; c < width; c++ )	{
			bmp.SetPixelColor( c,r,pixel32s[nz++] );
		}	
		}
		bmp.Flip( );		//为了与matlab的demo一致
		bSave = bmp.Save( sPath.c_str(),CXIMAGE_FORMAT_BMP );
	}
	return bSave;
}


int Save8bitBMP(BIT_8 *imgBuf, int width, int height, wchar_t *bmpName)	{
	int biBitCount=8,i,j; //每个像素所占的位数(bit)
	int colorTablesize=1024;//颜色表大小，以字节为单位，灰度图像颜色表为1024字节
	//待存储图像数据每行字节数为4的倍数
	int lineByte=(width * biBitCount/8+3)/4*4;
	BITMAPFILEHEADER fileHead;
	BITMAPINFOHEADER head;
	RGBQUAD pColorTable[256];

	FILE *fp=_tfopen(bmpName,_T("wb") );
	if(fp==0)
		return -100;
	//申请位图文件头结构变量，填写文件头信息
	fileHead.bfType=0x4D42; //bmp类型
	fileHead.bfSize=sizeof(BITMAPFILEHEADER)+sizeof(BITMAPINFOHEADER)+colorTablesize+lineByte*height;//bfSize是图像文件4个组成部分之和
	fileHead.bfReserved1=0;
	fileHead.bfReserved2=0;
	fileHead.bfOffBits=54+colorTablesize;		//bfOffBits是图像文件前3个部分所需空间之和
	//写文件头进文件
	fwrite(&fileHead, sizeof(BITMAPFILEHEADER),1, fp);
	//申请位图信息头结构变量，填写信息头信息
	head.biSize=40; //本结构的长度
	head.biWidth=width; //位图的宽度，以像素为单位
	head.biHeight=height; //位图的宽度，以像素为单位
	head.biPlanes=1; //目标设备的级别，必须是1
	head.biBitCount=8; //每个像素所占的位数(bit)，其值必须为1（黑白图像），4（16色图），8（256色），24（真彩色图）
	head.biCompression=BI_RGB; //位图压缩类型，有效的值为BI_RGB（未经压缩）、BI_RLE4，BI_RLE8，BI_BITFIEDS（均为Windows定义常量）。
	head.biSizeImage=lineByte*height; //实际的位图数据占用的字节数
	head.biXPelsPerMeter=0; //指定目标数据的水平分辨率，单位是像素/米。
	head.biYPelsPerMeter=0; ////指定目标数据的垂直分辨率，单位是像素/米。
	head.biClrUsed=0; //位图实际用到的颜色数，如果该值为零，则用到的颜色数为2的biBitCount次幂
	head.biClrImportant=0; //位图显示过程中重要的颜色数，如果该值为零，则认为所有的颜色都是重要的。
	//写位图信息头进内存
	fwrite(&head, sizeof(BITMAPINFOHEADER),1, fp);
	//申请颜色表所需要的空间，写颜色表进文件
	for ( i=0;i<256;i++)	{
		pColorTable[i].rgbRed=i;
		pColorTable[i].rgbGreen=i;
		pColorTable[i].rgbBlue=i;
		pColorTable[i].rgbReserved=0;
	}
	fwrite(pColorTable,sizeof(RGBQUAD),256,fp);
	//判断位图数据宽度是否正确，并写数据进入BMP
	if (width<lineByte) {	//如果有效数据宽度小于BMP格式要求宽度
		BIT_8 *imgBufBMP=(BIT_8*)malloc( sizeof(BIT_8)*(lineByte*height) );
		//将有效数据赋给用于写BMP的数据空间
		for ( i=0;i<height;i++)		{
		for ( j=0;j<width;j++)		{
			*(imgBufBMP+i*lineByte+j)=*(imgBuf+i*width+j);
		}
		}
		for ( i=0;i<height;i++)		{//将大于有效数据宽度的部分补0
		for ( j=width;j<lineByte;j++)		{
			*(imgBufBMP+i*lineByte+j)=0;
		}
		}
		fwrite(imgBufBMP, height*lineByte, 1, fp);//写BMP格式要求的图数据写进文件
		free( imgBufBMP );
	}
	else	{
	//写有效图数据写进文件
		fwrite(imgBuf, height*lineByte, 1, fp);
	}
	//关闭文件
	fclose(fp);
	return 0;
}
int GST_bitmap::save_doublebmp_n(int nBmp,int no, int width, int h, double * pdata,int x,int type )	{
	bool isColor=BIT_TEST( type,COLOR );
	int i_1=(int)sqrt(nBmp*1.0),j_1=(int)ceil(nBmp*1.0/i_1),i,j,pos,r,c,cur,ret=0;
//	j_1=1;	i_1=1;	
	int ldBmp=j_1*(width+1),rB,cB;
	int patch = width*h,nPixel=(width+1)*(h+1)*i_1*j_1;
	double *pd,S_1=-DBL_MAX,S_0=FLT_MAX,s=1.0;
	BIT_8 *tmp8=isColor ? nullptr : new BIT_8[nPixel]( ),cr,cg,cb;
	BIT_32 *tmp32=isColor ? new BIT_32[nPixel]( ) : nullptr;
//	for( i = sizeof(BIT_8)*(width+1)*(h+1)*i_1*j_1-1; i>=0; i-- )tmp8[i]=0xFF;

	for( i = 0; i < i_1; i++ )	{		//纵向
	for( j = 0; j < j_1; j++ )	{		//横向
		if( (cur = i*j_1+j)>=nBmp )
			break;
		rB=i*(h+1),		cB=j*(width+1);
		S_1=-DBL_MAX,S_0=FLT_MAX;
		if( isColor )	{
			pd = pdata+cur*patch*3;			
			for( pos=0; pos<patch*3; pos++ )	{
				S_0 = min( S_0,pd[pos] );		S_1=max( S_1,pd[pos] );	
			}
		} else		{
			pd = pdata+cur*patch;	
			for( pos=0; pos<patch; pos++ )	{
				S_0 = min( S_0,pd[pos] );		S_1=max( S_1,pd[pos] );	
			}
		}
		s = ( S_0==S_1 ) ? 1.0 : 1.0/(S_1-S_0);
		for( pos=0; pos<patch; pos++ )	{
			r=pos/width;		c=pos%width;
			if( isColor )	{
				cr=(BIT_8)(255*(pd[pos]-S_0)*s);
				cg=(BIT_8)(255*(pd[patch+pos]-S_0)*s);
				cb=(BIT_8)(255*(pd[2*patch+pos]-S_0)*s);
				tmp32[(rB+r)*ldBmp+c+cB]=RGB(cr,cg,cb);//(BIT_32)(0xff000000 | (cb<<16) | (cg<<8) | cr);
			} else			{
				tmp8[(rB+r)*ldBmp+c+cB]=(BIT_8)(255*(pd[pos]-S_0)*s);
			}
		}
	}
	}
NEXT:
	if( isColor )	{
		ret = save_colorbmp( no,(width+1)*j_1,(h+1)*i_1,tmp32,x,type );
		delete[] tmp32;
	}else	{
		ret = save_graybmp( no,(width+1)*j_1,(h+1)*i_1,tmp8,x,type );
		delete[] tmp8;
	}

	return ret;
}

int GST_bitmap::save_doublebmp(int no, int w, int h, double * pdata,int x,int type )	{
	int i,ld=w*h,ret=0x0;
	BIT_8 *tmp8=(BIT_8*)malloc(sizeof(BIT_8)*ld);
	float S_1=-DBL_MAX,S_0=FLT_MAX;
	for( i =0; i<ld; i++ ){
		S_0 = min( S_0,pdata[i] );		S_1=max( S_1,pdata[i] );	
	}
	if( S_0==S_1 )
		return -10;
	for( i =0; i<ld; i++ ){
		tmp8[i]=(BIT_8)(255*(pdata[i]-S_0)/(S_1-S_0));
	}
	ret = save_graybmp( no,w,h,tmp8,x,type );
	free(tmp8);

	return ret;
}

int GST_bitmap::save_graybmp(int no, int w, int h, BIT_8 * pdata,int x,int type )	{
#ifdef WIN32
	wchar_t sPath[256] = _T("\0");
#else
	char sPath[256] = "\0",sDir[]="/storage/sdcard0/1_cys/";
#endif
	int ret;
//	if( ge_dump<1 )
//		return 0;
	switch( type )	{
	default:
		if( no<0 )
			_stprintf( sPath,_T("%s%3d.bmp"),sDumpFolder.c_str(),x );
		else
			_stprintf( sPath,_T("%s%3d_%d.bmp"),sDumpFolder.c_str(),no,x );
		break;
	}
	ret = Save8bitBMP( pdata,w,h,sPath );
END:
	return ret;
}

int GST_bitmap::save_colorbmp(int no, int w, int h, BIT_32 * pdata,int x,int type )	{
#ifdef WIN32
	wchar_t sPath[256] = _T("\0");
#else
	wchar_t sPath[256] = "\0",sDir[]="/storage/sdcard0/1_cys/";
#endif
	int ret;
//	if( ge_dump<1 )
//		return 0;

	switch( type )	{
	default:
		if( no<0 )
			_stprintf( sPath,_T("%s%3d.bmp"),sDumpFolder.c_str(),x );
		else
			_stprintf( sPath,_T("%s%3d_%d.bmp"),sDumpFolder.c_str(),no,x );
		break;
	}
	ret = SaveColorBMP( pdata,w,h,32,sPath );
END:
	return ret;
}