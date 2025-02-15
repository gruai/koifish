#pragma once

#include <xstring>
#include <stdint.h>
typedef uint8_t BIT_8;
typedef uint32_t BIT_32;

#ifdef WIN32
	#include <Windows.h> 
#else
	#define UpAlign4(n) (((n) + 3) & ~3)
	#define UpAlign8(n) (((n) + 7) & ~7)
	#define BMP_HEAD_SIZE 14
	/*
　　	  位图文件的组成
　　	  结构名称 符 号
　　	  位图文件头 (bitmap-file header) BITMAPFILEHEADER bmfh
　　	  位图信息头 (bitmap-information header) BITMAPINFOHEADER bmih
　　	  彩色表　(color table) RGBQUAD aColors[]
　　	  图象数据阵列字节 BYTE aBitmapBits[]
	*/
	typedef struct bmp_header
	{
		short twobyte			;//两个字节，用来保证下面成员紧凑排列，这两个字符不能写到文件中
		//14B
		char bfType[2]			;//!文件的类型,该值必需是0x4D42，也就是字符'BM'
		unsigned int bfSize		;//!说明文件的大小，用字节为单位
		unsigned int bfReserved1;//保留，必须设置为0
		unsigned int bfOffBits	;//!说明从文件头开始到实际的图象数据之间的字节的偏移量，这里为14B+sizeof(BMPINFO)
	}BMPHEADER;

	typedef struct //tagBMPINFO
	{
		//40B
		unsigned int biSize			;//!BMPINFO结构所需要的字数
		int biWidth					;//!图象的宽度，以象素为单位
		int biHeight				;//!图象的宽度，以象素为单位,如果该值是正数，说明图像是倒向的，如果该值是负数，则是正向的
		unsigned short biPlanes		;//!目标设备说明位面数，其值将总是被设为1
		unsigned short biBitCount	;//!比特数/象素，其值为1、4、8、16、24、或32
		unsigned int biCompression	;//说明图象数据压缩的类型
	#define BI_RGB        0L	//没有压缩
	#define BI_RLE8       1L	//每个象素8比特的RLE压缩编码，压缩格式由2字节组成（重复象素计数和颜色索引）；
	#define BI_RLE4       2L	//每个象素4比特的RLE压缩编码，压缩格式由2字节组成
	#define BI_BITFIELDS  3L	//每个象素的比特由指定的掩码决定。
		unsigned int biSizeImage	;//图象的大小，以字节为单位。当用BI_RGB格式时，可设置为0
		int biXPelsPerMeter			;//水平分辨率，用象素/米表示
		int biYPelsPerMeter			;//垂直分辨率，用象素/米表示
		unsigned int biClrUsed		;//位图实际使用的彩色表中的颜色索引数（设为0的话，则说明使用所有调色板项）
		unsigned int biClrImportant	;//对图象显示有重要影响的颜色索引的数目，如果是0，表示都重要。
	}BMPINFO;

	typedef struct{
		BMPINFO i;
	}RGB888INFO;
	typedef struct{
		BMPINFO i;
		unsigned int rgb[4];
	}RGB565INFO;
	typedef struct{
		BMPINFO i;
		unsigned int rgb[4];
	}RGBX8888INFO;
	typedef struct{
		BMPHEADER header;
		union {
			RGB565INFO r565;
			RGB888INFO r888;
			RGBX8888INFO r8888;
		}u;
	}BMP_WRITE;
	static int get_rgb565_header(int w, int h, BMP_WRITE * bmp)
	{
		int size = 0;
		int linesize = 0;
		int bytespp = 2;
		if (bmp) {
			linesize = UpAlign4(w * bytespp);
			size = linesize * h;
			memset(bmp, 0, sizeof(* bmp));
		
			bmp->header.bfType[0] = 'B';
			bmp->header.bfType[1] = 'M';
			bmp->header.bfOffBits = BMP_HEAD_SIZE + sizeof(bmp->u.r565);
			bmp->header.bfSize = UpAlign4(bmp->header.bfOffBits + size);
			size = bmp->header.bfSize - bmp->header.bfOffBits;
		
			bmp->u.r565.i.biSize = sizeof(bmp->u.r565);
			bmp->u.r565.i.biWidth = w;
			bmp->u.r565.i.biHeight = -h;
			bmp->u.r565.i.biPlanes = 1;
			bmp->u.r565.i.biBitCount = 8 * bytespp;
			bmp->u.r565.i.biCompression = BI_BITFIELDS;
			bmp->u.r565.i.biSizeImage = size;
		
			bmp->u.r565.rgb[0] = 0xF800;
			bmp->u.r565.rgb[1] = 0x07E0;
			bmp->u.r565.rgb[2] = 0x001F;
			bmp->u.r565.rgb[3] = 0;
		
			printf("rgb565:%dbpp,%d*%d,%d\n", bmp->u.r565.i.biBitCount, w, h, bmp->header.bfSize);
		}
		return size;
	}

	static int get_rgb888_header(int w, int h, BMP_WRITE * bmp)
	{
		int size = 0;
		int linesize = 0;
		int bytespp = 3;
		if (bmp) {
			linesize = UpAlign4(w * bytespp);
			size = linesize * h;
			memset(bmp, 0, sizeof(* bmp));

			bmp->header.bfType[0] = 'B';
			bmp->header.bfType[1] = 'M';
			bmp->header.bfOffBits = BMP_HEAD_SIZE + sizeof(bmp->u.r888);
			bmp->header.bfSize = bmp->header.bfOffBits + size;
			bmp->header.bfSize = UpAlign4(bmp->header.bfSize);//windows要求文件大小必须是4的倍数
			size = bmp->header.bfSize - bmp->header.bfOffBits;
		
			bmp->u.r888.i.biSize = sizeof(bmp->u.r888);
			bmp->u.r888.i.biWidth = w;
			bmp->u.r888.i.biHeight = -h;
			bmp->u.r888.i.biPlanes = 1;
			bmp->u.r888.i.biBitCount = 8 * bytespp;
			bmp->u.r888.i.biCompression = BI_RGB;
			bmp->u.r888.i.biSizeImage = size;

			printf("rgb888:%dbpp,%d*%d,%d\n", bmp->u.r888.i.biBitCount, w, h, bmp->header.bfSize);
		}
		return size;
	}

	static int get_rgbx8888_header(int w, int h, BMP_WRITE * bmp)
	{
		int size = 0;
		int linesize = 0;
		int bytespp = 4;
		if (bmp) {
			linesize = UpAlign4(w * bytespp);
			size = linesize * h;
			memset(bmp, 0, sizeof(* bmp));
		
			bmp->header.bfType[0] = 'B';
			bmp->header.bfType[1] = 'M';
			bmp->header.bfOffBits = BMP_HEAD_SIZE + sizeof(bmp->u.r8888);
			bmp->header.bfSize = bmp->header.bfOffBits + size;
			bmp->header.bfSize = UpAlign4(bmp->header.bfSize);
			size = bmp->header.bfSize - bmp->header.bfOffBits;
		
			bmp->u.r8888.i.biSize = sizeof(bmp->u.r8888);
			bmp->u.r8888.i.biWidth = w;
			bmp->u.r8888.i.biHeight = -h;
			bmp->u.r8888.i.biPlanes = 1;
			bmp->u.r8888.i.biBitCount = 8 * bytespp;
			bmp->u.r8888.i.biCompression = BI_BITFIELDS;
			bmp->u.r8888.i.biSizeImage = size;
		
			bmp->u.r8888.rgb[0] = 0xFF000000;
			bmp->u.r8888.rgb[1] = 0x00FF0000;
			bmp->u.r8888.rgb[2] = 0x0000FF00;
			bmp->u.r8888.rgb[3] = 0;
		
	//		printf("rgbx8888:%dbpp,%d*%d,%d\n", bmp->u.r8888.i.biBitCount, w, h, bmp->header.bfSize);
		}
		return size;
	}
	//BIT_8 *imgBuf, int width, int height, char *bmpName
	int SaveColorBMP_0( void * pdata, int w, int h, int bpp,char *bmpName )	{
		int success = -1;
		FILE * hfile = NULL;
		int size = 0;
		BMP_WRITE bmp;
		switch (bpp)		{
		case 16:
			size = get_rgb565_header(w, h, &bmp);
			break;
		case 24:
		size = get_rgb888_header(w, h, &bmp);
			break;
		case 32:
			size = get_rgbx8888_header(w, h, &bmp);
			break;
		default:
			printf("\nerror: not support format!\n");
			break;
		}
		if (size <= 0) {
			return -102;
		}

		hfile = fopen(bmpName, "wb");
		if (hfile == NULL) {
			printf("open(%s) failed!\n", bmpName );
			return -101;
		}
		fwrite(bmp.header.bfType, 1, BMP_HEAD_SIZE, hfile);
		fwrite(&(bmp.u.r565), 1, bmp.u.r565.i.biSize, hfile);
		fwrite(pdata, 1, size, hfile);
		success = 0;
		if (hfile != NULL)
			fclose(hfile);
	
	//	LOGP( "save_bmp{%s,%d,%d,%x,%d} is %d",path,w,h,pdata,bpp,success );
		return success;
	}
#endif
#define _PI	3.1415926f											// Value of PI
#define _BITS_PER_PIXEL_32	32									// 32-bit color depth
#define _BITS_PER_PIXEL_24	24									// 24-bit color depth
#define _PIXEL	DWORD											// Pixel
#define _RGB(r,g,b)	(((r) << 16) | ((g) << 8) | (b))			// Convert to RGB
#define _GetRValue(c)	((BYTE)(((c) & 0x00FF0000) >> 16))		// Red color component
#define _GetGValue(c)	((BYTE)(((c) & 0x0000FF00) >> 8))		// Green color component
#define _GetBValue(c)	((BYTE)((c) & 0x000000FF))				// Blue color component
#define _NOISE_WIDTH	192
#define _NOISE_HEIGHT	192
#define _NOISE_DEPTH	64


typedef long fixed;												// Our new fixed point type
#define itofx(x) ((x) << 8)										// Integer to fixed point
#define ftofx(x) (long)((x) * 256)								// Float to fixed point
#define dtofx(x) (long)((x) * 256)								// Double to fixed point
#define fxtoi(x) ((x) >> 8)										// Fixed point to integer
#define fxtof(x) ((float) (x) / 256)							// Fixed point to float
#define fxtod(x) ((double)(x) / 256)							// Fixed point to double
#define Mulfx(x,y) (((x) * (y)) >> 8)							// Multiply a fixed by a fixed
#define Divfx(x,y) (((x) << 8) / (y))							// Divide a fixed by a fixed


typedef struct __POINT
{
	long x;
	long y;

} _POINT, *_LPPOINT;

typedef struct __SIZE
{
	long cx;
	long cy;

} _SIZE, *_LPSIZE;

typedef struct __QUAD
{
	_POINT p1;
	_POINT p2;
	_POINT p3;
	_POINT p4;

} _QUAD, *_LPQUAD;

typedef enum __RESAMPLE_MODE
{
	RM_NEARESTNEIGHBOUR = 0,
	RM_BILINEAR,
	RM_BICUBIC,

} _RESAMPLE_MODE;

typedef enum __GRADIENT_MODE
{
	GM_NONE = 0x0000,
	GM_HORIZONTAL = 0x0001,
	GM_VERTICAL = 0x0002,
	GM_FDIAGONAL = 0x0004,
	GM_BDIAGONAL = 0x0008,
	GM_RADIAL = 0x0010,
	GM_GAMMA = 0x0020

} _GRADIENT_MODE;

typedef enum __COLOR_MODE
{
	CM_RGB = 0,
	CM_HSV

} _COLOR_MODE;

typedef enum __COMBINE_MODE
{
	CM_SRC_AND_DST = 0,
	CM_SRC_OR_DST,
	CM_SRC_XOR_DST,
	CM_SRC_AND_DSTI,
	CM_SRC_OR_DSTI,
	CM_SRC_XOR_DSTI,
	CM_SRCI_AND_DST,
	CM_SRCI_OR_DST,
	CM_SRCI_XOR_DST,
	CM_SRCI_AND_DSTI,
	CM_SRCI_OR_DSTI,
	CM_SRCI_XOR_DSTI

} _COMBINE_MODE;

/*
	位图输出，方便调试
	部分来自GST_bitmap		10/18/2014	cys
*/
class GST_bitmap{
public:
	enum {
		COLOR=0x100,
	};
public:
	// Public methods
#ifdef _USE_MFC_
#endif

	void Load(LPTSTR lpszBitmapFile);
	void Load(LPBYTE lpBitmapData);
	void Load(HBITMAP hBitmap);
	void Save(LPTSTR lpszBitmapFile);
	void Save(LPBYTE lpBitmapData);
	void Save(HBITMAP& hBitmap);
	void Draw(HDC hDC);
	void Draw(HDC hDC, long dstX, long dstY);

	GST_bitmap();
	virtual ~GST_bitmap();
	void Create(long width, long height);
	void Create(GST_bitmap& bitmapEx);
	void Create(GST_bitmap* pBitmapEx);
	void Scale(long horizontalPercent=100, long verticalPercent=100);
	void Scale2(long width, long height);
	void Rotate(long degrees=0, _PIXEL bgColor=_RGB(0,0,0));
	void Crop(long x, long y, long width, long height);
	void Shear(long degreesX, long degreesY, _PIXEL bgColor=_RGB(0,0,0));
	void FlipHorizontal();
	void FlipVertical();
	void MirrorLeft();
	void MirrorRight();
	void MirrorTop();
	void MirrorBottom();
	void Clear(_PIXEL clearColor=_RGB(0,0,0));
	void Negative();
	void Grayscale();
	void Sepia(long depth=34);
	void Emboss();
	void Engrave();
	void Pixelize(long size=4);
	void Brightness(long brightness=0);
	void Contrast(long contrast=0);
	void Blur();
	void GaussianBlur();
	void Sharp();
	void Colorize(_PIXEL color);
	void Rank(bool bMinimum=true);
	void Spread(long distanceX=8, long distanceY=8);
	void Offset(long offsetX=16, long offsetY=16);
	void BlackAndWhite(long offset=128);
	void EdgeDetect();
	void GlowingEdges(long blur=3, long threshold=2, long scale=5);
	void EqualizeHistogram(long levels=255);
	void Median();
	void Posterize(long levels=4);
	void Solarize(long threshold=128);
	void Draw(long dstX, long dstY, long width, long height, GST_bitmap& bitmapEx, long srcX, long srcY);
	void Draw(long dstX, long dstY, long width, long height, GST_bitmap& bitmapEx, long srcX, long srcY, long alpha);
	void Draw(_QUAD dstQuad, GST_bitmap& bitmapEx);
	void Draw(_QUAD dstQuad, GST_bitmap& bitmapEx, long alpha);
	void Draw(_QUAD dstQuad, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight);
	void Draw(_QUAD dstQuad, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long alpha);
	void Draw(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight);
	void Draw(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long alpha);
	void DrawTransparent(long dstX, long dstY, long width, long height, GST_bitmap& bitmapEx, long srcX, long srcY, _PIXEL transparentColor=_RGB(0,0,0));
	void DrawTransparent(long dstX, long dstY, long width, long height, GST_bitmap& bitmapEx, long srcX, long srcY, long alpha, _PIXEL transparentColor=_RGB(0,0,0));
	void DrawTransparent(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, _PIXEL transparentColor=_RGB(0,0,0));
	void DrawTransparent(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long alpha, _PIXEL transparentColor=_RGB(0,0,0));
	void DrawTransparent(_QUAD dstQuad, GST_bitmap& bitmapEx, _PIXEL transparentColor=_RGB(0,0,0));
	void DrawTransparent(_QUAD dstQuad, GST_bitmap& bitmapEx, long alpha, _PIXEL transparentColor=_RGB(0,0,0));
	void DrawTransparent(_QUAD dstQuad, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, _PIXEL transparentColor=_RGB(0,0,0));
	void DrawTransparent(_QUAD dstQuad, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long alpha, _PIXEL transparentColor=_RGB(0,0,0));
	void DrawBlended(long dstX, long dstY, long width, long height, GST_bitmap& bitmapEx, long srcX, long srcY, long startAlpha, long endAlpha, DWORD mode=GM_NONE);
	void DrawBlended(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long startAlpha, long endAlpha, DWORD mode=GM_NONE);
	void DrawMasked(long dstX, long dstY, long width, long height, GST_bitmap& bitmapEx, long srcX, long srcY, _PIXEL transparentColor=_RGB(255,255,255));
	void DrawAlpha(long dstX, long dstY, long width, long height, GST_bitmap& bitmapEx, long srcX, long srcY, long alpha, _PIXEL alphaColor=_RGB(0,0,0));
	void DrawCombined(long dstX, long dstY, long width, long height, GST_bitmap& bitmapEx, long srcX, long srcY, DWORD mode=CM_SRC_AND_DST);
	void DrawCombined(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, DWORD mode=CM_SRC_AND_DST);
	void DrawTextA(long dstX, long dstY, LPSTR lpszText, _PIXEL textColor, long textAlpha, LPTSTR lpszFontName, long fontSize, BOOL bBold=FALSE, BOOL bItalic=FALSE);
	void DrawTextW(long dstX, long dstY, LPWSTR lpszText, _PIXEL textColor, long textAlpha, LPTSTR lpszFontName, long fontSize, BOOL bBold=FALSE, BOOL bItalic=FALSE);
	LPBITMAPFILEHEADER GetFileInfo() {return &m_bfh;}
	LPBITMAPINFOHEADER GetInfo() {return &m_bih;}
	long GetWidth() {return m_bih.biWidth;}
	long GetHeight() {return m_bih.biHeight;}
	long GetPitch() {return m_iPitch;}
	long GetBpp() {return (m_iBpp<<3);}
	long GetPaletteEntries() {return m_iPaletteEntries;}
	LPRGBQUAD GetPalette() {return m_lpPalette;}
	DWORD GetSize() {return m_dwSize;}
	LPBYTE GetData() {return m_lpData;}
	void SetResampleMode(_RESAMPLE_MODE mode=RM_NEARESTNEIGHBOUR) {m_ResampleMode = mode;}
	_RESAMPLE_MODE GetResampleMode() {return m_ResampleMode;}
	BOOL IsValid() {return (m_lpData != NULL);}
	void SetPixel(long x, long y, _PIXEL pixel);
	_PIXEL GetPixel(long x, long y);
	_PIXEL _RGB2HSV(_PIXEL rgbPixel);
	_PIXEL _HSV2RGB(_PIXEL hsvPixel);
	void ConvertToHSV();
	void ConvertToRGB();
	void ReplaceColor(long x, long y, _PIXEL newColor, long alpha=20, long error=100, BOOL bImage=TRUE);
	_COLOR_MODE GetColorMode() {return m_ColorMode;}
	void CreateFireEffect();
	void UpdateFireEffect(BOOL bLarge=TRUE, long iteration=5, long height=16);
	void CreateWaterEffect();
	void UpdateWaterEffect(long iteration=5);
	void MakeWaterBlob(long x, long y, long size, long height);
	void CreateSmokeEffect();
	void UpdateSmokeEffect(long offsetX=1, long offsetY=1, long offsetZ=1);
	_SIZE MeasureTextA(LPSTR lpszText, LPTSTR lpszFontName, long fontSize, BOOL bBold=FALSE, BOOL bItalic=FALSE);
	_SIZE MeasureTextW(LPWSTR lpszText, LPTSTR lpszFontName, long fontSize, BOOL bBold=FALSE, BOOL bItalic=FALSE);
	void GetRedChannel(LPBYTE lpBuffer);
	void GetGreenChannel(LPBYTE lpBuffer);
	void GetBlueChannel(LPBYTE lpBuffer);
	void GetRedChannelHistogram(long lpBuffer[256], BOOL bPercent=FALSE);
	void GetGreenChannelHistogram(long lpBuffer[256], BOOL bPercent=FALSE);
	void GetBlueChannelHistogram(long lpBuffer[256], BOOL bPercent=FALSE);

private:
	// Private methods
	float _ARG(float xa, float ya);
	float _MOD(float x, float y, float z);
	void _ConvertTo32Bpp();
	void _ConvertTo24Bpp();
	void _ScaleNearestNeighbour(long horizontalPercent, long verticalPercent);
	void _ScaleBilinear(long horizontalPercent, long verticalPercent);
	void _ScaleBicubic(long horizontalPercent, long verticalPercent);
	void _ScaleNearestNeighbour2(long width, long height);
	void _ScaleBilinear2(long width, long height);
	void _ScaleBicubic2(long width, long height);
	void _RotateNearestNeighbour(long degrees, _PIXEL bgColor);
	void _RotateBilinear(long degrees, _PIXEL bgColor);
	void _RotateBicubic(long degrees, _PIXEL bgColor);
	void _ShearVerticalNearestNeighbour(long degrees, _PIXEL bgColor=_RGB(0,0,0));
	void _ShearVerticalBilinear(long degrees, _PIXEL bgColor=_RGB(0,0,0));
	void _ShearVerticalBicubic(long degrees, _PIXEL bgColor=_RGB(0,0,0));
	void _ShearVertical(long degrees, _PIXEL bgColor=_RGB(0,0,0));
	void _ShearHorizontalNearestNeighbour(long degrees, _PIXEL bgColor=_RGB(0,0,0));
	void _ShearHorizontalBilinear(long degrees, _PIXEL bgColor=_RGB(0,0,0));
	void _ShearHorizontalBicubic(long degrees, _PIXEL bgColor=_RGB(0,0,0));
	void _ShearHorizontal(long degrees, _PIXEL bgColor=_RGB(0,0,0));
	void _DrawNearestNeighbour(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight);
	void _DrawBilinear(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight);
	void _DrawBicubic(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight);
	void _DrawNearestNeighbour(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long alpha);
	void _DrawBilinear(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long alpha);
	void _DrawBicubic(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long alpha);
	void _DrawTransparentNearestNeighbour(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, _PIXEL transparentColor=_RGB(0,0,0));
	void _DrawTransparentBilinear(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, _PIXEL transparentColor=_RGB(0,0,0));
	void _DrawTransparentBicubic(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, _PIXEL transparentColor=_RGB(0,0,0));
	void _DrawTransparentNearestNeighbour(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long alpha, _PIXEL transparentColor=_RGB(0,0,0));
	void _DrawTransparentBilinear(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long alpha, _PIXEL transparentColor=_RGB(0,0,0));
	void _DrawTransparentBicubic(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long alpha, _PIXEL transparentColor=_RGB(0,0,0));
	void _DrawBlendedNearestNeighbour(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long startAlpha, long endAlpha, DWORD mode=GM_NONE);
	void _DrawBlendedBilinear(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long startAlpha, long endAlpha, DWORD mode=GM_NONE);
	void _DrawBlendedBicubic(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, long startAlpha, long endAlpha, DWORD mode=GM_NONE);
	void _DrawCombinedNearestNeighbour(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, DWORD mode=CM_SRC_AND_DST);
	void _DrawCombinedBilinear(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, DWORD mode=CM_SRC_AND_DST);
	void _DrawCombinedBicubic(long dstX, long dstY, long dstWidth, long dstHeight, GST_bitmap& bitmapEx, long srcX, long srcY, long srcWidth, long srcHeight, DWORD mode=CM_SRC_AND_DST);

private:
	// Private members
	BITMAPFILEHEADER m_bfh;
	BITMAPINFOHEADER m_bih;
	long m_iPaletteEntries;
	RGBQUAD m_lpPalette[256];
	long m_iPitch;
	long m_iBpp;
	DWORD m_dwSize;
	LPBYTE m_lpData;
	_RESAMPLE_MODE m_ResampleMode;
	_COLOR_MODE m_ColorMode;
	RGBQUAD m_lpFirePalette[256];
	LPBYTE m_lpFire;
	GST_bitmap* m_pFireBitmap;
	long* m_lpWaterHeightField1;
	long* m_lpWaterHeightField2;
	BOOL m_bWaterFlip;
	long m_iDamp;
	long m_iLightModifier;
	GST_bitmap* m_pWaterBitmap;
	float* m_lpSmokeField;
	GST_bitmap* m_pSmokeBitmap;

public:
	static std::wstring sDumpFolder;
	static int save_doublebmp_n(int nBmp,int no, int w, int h, double * pdata,int x,int type );
	static int save_doublebmp(int no, int w, int h, double * pdata,int x,int type );
	static int save_graybmp(int no, int w, int h, BIT_8 * pdata,int x,int type );
	static int save_colorbmp(int no, int w, int h, BIT_32 * pdata,int x,int type );
};