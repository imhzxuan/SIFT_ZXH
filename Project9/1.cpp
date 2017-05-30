#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include<math.h>
#define PI 3.1415926535
#define E 2.718281828
using namespace cv;
using namespace std;

struct feature
{
	int sigma;
	int x;
	int y;
	int octave;
	double m;
	double ori;
	double theta;
	double desc[4][4][8];
};
feature feat[10000];
void Dog(Mat &srcImage, Mat dog[3][5]);
void getmax_min(Mat dog[3][5], Mat dstmat[3][3]);
void downsize(Mat &srcImage, Mat &dstImage);
int compare_ptr(uchar *ptr1, uchar *ptr2, uchar *ptr3, uchar *ptr4, uchar *ptr5, uchar *ptr6, uchar *ptr7, uchar *ptr8, uchar *ptr9, int n);

void init_feat(feature a[10000])
{
	for (int i = 0; i < 10000; i++)
	{
		a[i].m = 0;
		a[i].x = 0;
		a[i].y = 0;
		a[i].ori = 0;
		a[i].sigma = 0;
		a[i].octave = 0;
	}
}

void downsize(Mat &srcImage, Mat &dstImage)
{
	int i, j;
	uchar *ptr1, *ptr2;
	int col, row;
	col = srcImage.cols;
	row = srcImage.rows;
	if (row % 2 != 0) row--;
	if (col % 2 != 0) col--;
	for (i = 0; i < row; i+=2)
	{
		ptr1 = srcImage.ptr(i);
		ptr2 = dstImage.ptr(i / 2);
		for (j = 0; j < col; j+=2)
		{
			ptr2[j / 2] = ptr1[j];
		}
	}
}
			
void Dog(Mat &srcImage, Mat dog[3][5],Mat Gaussian[3][6])
{
	int i, j,col,row;
	uchar *ptr1, *ptr2,*ptr3,*ptr4;
	double sigma = 1.6;
	double k = pow(2.0, 1.0 / 3);
//	double k = 1.259921;

	//构建第0组第0层
	col = srcImage.cols;
	row = srcImage.rows;
	for (i = 0; i < 6; i++) Gaussian[0][i].create(Size(2 * col, 2 * row), CV_8UC1);
	for (i = 0; i < 6; i++) Gaussian[1][i].create(Size(col,row), CV_8UC1);
	for (i = 0; i < 6; i++) Gaussian[2][i].create(Size(col/2,row/2), CV_8UC1);
	for (i = 0; i < row-1; i++)
	{
		ptr1 = srcImage.ptr(i);
		ptr4 = srcImage.ptr(i+1);
		ptr2 = Gaussian[0][0].ptr(i * 2);
		ptr3 = Gaussian[0][0].ptr(i * 2 + 1);
		for (j = 0; j < col-1; j++)
		{
			ptr2[j * 2] = ptr1[j];
			ptr2[j * 2 + 1] = (ptr1[j] + ptr1[j + 1])/2;
			ptr3[j * 2] = (ptr1[j]+ptr4[j])/2;
			ptr3[j * 2 + 1] = (ptr1[j] + ptr4[j] + ptr1[j+1] + ptr4[j+1]) / 4;
		}
	}

	


	//高斯滤波构建DOG部分
	for (i = 0; i < 3; i++)
	{
		if (i) downsize(Gaussian[i - 1][2], Gaussian[i][0]);
		for (j = 1; j < 6; j++) GaussianBlur(Gaussian[i][0], Gaussian[i][j], Size(0, 0), pow(k, j)*sigma, pow(k, j)*sigma);
	}
	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 5; j++) dog[i][j] = Gaussian[i][j+1] - Gaussian[i][j];
	}
//	cout << dog[0][1] << endl;
}
/*
int compare_ptr(uchar *ptr1, uchar *ptr2, uchar *ptr3, uchar *ptr4, uchar *ptr5, uchar *ptr6, uchar *ptr7, uchar *ptr8, uchar *ptr9,int n)
{
	int i,j,a,b,c;
	uchar *ptr[9] = { ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7, ptr8, ptr9 };
	a = ptr5[n];
	b = ptr1[n];
	c = ptr1[n];
	for (i = 0; i < 9; i++)
		for (j = -1; j < 2; j++)
		{
			if (ptr[i][n + j] > b) b = ptr[i][n + j];
			if (ptr[i][n + j] < c) c = ptr[i][n + j];
		}
	if ((a == b)) return 1;
	else if ((a == c)) return 2;			//这里可能要改
	else return 0;
}
*/
int compare_ptr(uchar *ptr1, uchar *ptr2, uchar *ptr3, uchar *ptr4, uchar *ptr5, uchar *ptr6, uchar *ptr7, uchar *ptr8, uchar *ptr9, int n)
{
	int i, j, a, b1, c1, b2, c2;
	uchar *ptr[9] = { ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7, ptr8, ptr9 };
	a = ptr5[n];
	b1 = ptr1[n];
	c1 = ptr1[n];
	b2 = 0;
	c2 = 255;
	int tempi1, tempj1, tempi2, tempj2;
	for (i = 0; i < 9; i++)
		for (j = -1; j < 2; j++)
		{
			if (ptr[i][n + j] > b1)
			{
				b1 = ptr[i][n + j];
				tempi1 = i;
				tempj1 = j;
			}
			if (ptr[i][n + j] < c1)
			{
				c1 = ptr[i][n + j];
				tempi2 = i;
				tempj2 = j;
			}
		}
	for (i = 0; i < 9; i++)
		for (j = -1; j < 2; j++)
		{
			if (((i != tempi1) || (j != tempj1)) && (ptr[i][n + j] > b2)) b2 = ptr[i][n + j];
			if (((i != tempi2) || (j != tempj2)) && (ptr[i][n + j] < c2)) c2 = ptr[i][n + j];
		}
	if ((a == b1) && (b1>b2)) return 1;
	if ((a == c1) && (c1<c2)) return 2;			//这里可能要改
	else return 0;
}

void drawcircle(Mat &srcImage, Mat &dstImage)
{
	int col, row, i, j;
	col = srcImage.cols;
	row = srcImage.rows;
	for (i = 1; i < row - 1; i++)
	{
		uchar *ptr1 = srcImage.ptr(i);
		uchar *ptr2 = dstImage.ptr(i);
		uchar *ptr3 = dstImage.ptr(i - 1);
		uchar *ptr4 = dstImage.ptr(i + 1);
		for (j = 5; j < col - 5; j++)
		{
			if (ptr2[j]>250) circle(srcImage, Point(j, i), 3, Scalar(0), 2, 8);
		}
	}
}

void getmax_min(Mat dog[3][5],Mat dstmat[3][3])
{
	int i, j,m=0,n=0;
	uchar *ptr1, *ptr2, *ptr3, *ptr4, *ptr5, *ptr6, *ptr7, *ptr8, *ptr9,*ptr_src;
	int row, col;

	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++) dstmat[i][j].create(Size(dog[i][0].cols, dog[i][0].rows), CV_8UC1);
	//给dstmat刷100
	for (i = 0; i < 3; i++)
	{
		row = dstmat[i][0].rows;
		col = dstmat[i][0].cols;
		for (j = 0; j < 3; j++)
		{
			for (m = 0; m < row; m++)
			{
				ptr1 = dstmat[i][j].ptr(m);
				for (n = 0; n < col; n++)
				{
					ptr1[n] = 100;
				}
			}
		}
	}

	

	//给特征点刷255和0
	for (i = 0; i < 3; i++)
	{
		row = dstmat[i][0].rows;
		col = dstmat[i][0].cols;
		for (j = 1; j < 4; j++)
		{
			for (m = 1; m < row - 1; m++)
			{
				ptr1 = dog[i][j - 1].ptr(m - 1);
				ptr2 = dog[i][j - 1].ptr(m);
				ptr3 = dog[i][j - 1].ptr(m + 1);
				ptr4 = dog[i][j].ptr(m - 1);
				ptr5 = dog[i][j].ptr(m);
				ptr6 = dog[i][j].ptr(m + 1);
				ptr7 = dog[i][j + 1].ptr(m - 1);
				ptr8 = dog[i][j + 1].ptr(m);
				ptr9 = dog[i][j + 1].ptr(m + 1);
				ptr_src = dstmat[i][j-1].ptr(m);
				for (n = 1; n < col - 1; n++)
				{
					if (compare_ptr(ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7, ptr8, ptr9, n)==1) ptr_src[n]=255;
					if (compare_ptr(ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7, ptr8, ptr9, n) == 2) ptr_src[n] = 0;
				}
			}
		}
	}

}

void mat_to_array(Mat srcImageROI, double dstarray[3][3])		//把图像转化为数组
{
	int i, j;
	for (i = 0; i < 3; i++)
	{
		uchar *ptr = srcImageROI.ptr(i);
		for (j = 0; j < 3; j++)
		{
			dstarray[i][j] = ptr[j];
		}
	}
}

int mat_invert(double srcarray[3][3], double dstarray[3][3])		//行列式为0返回0
{
	double det;
	double *ptr0, *ptr1, *ptr2;
	ptr0 = srcarray[0];
	ptr1 = srcarray[1];
	ptr2 = srcarray[2];
	det = ptr0[0] * ptr1[1] * ptr2[2]
		+ ptr0[1] * ptr1[2] * ptr2[0]
		+ ptr0[2] * ptr1[0] * ptr2[1]
		- ptr0[0] * ptr1[2] * ptr2[1]
		- ptr0[1] * ptr1[0] * ptr2[2]
		- ptr0[2] * ptr1[1] * ptr2[0];
	if (abs(det) < 1e-6)
	{
		return 0;
	}
	dstarray[0][0] = (ptr1[1] * ptr2[2] - ptr2[1] * ptr1[2]) / det; 
	dstarray[0][1] = (ptr2[1] * ptr0[2] - ptr0[1] * ptr2[2]) / det;
	dstarray[0][2] = (ptr0[1] * ptr1[2] - ptr0[2] * ptr1[1]) / det;
	dstarray[1][0] = (ptr1[2] * ptr2[0] - ptr2[2] * ptr1[0]) / det;
	dstarray[1][1] = (ptr2[2] * ptr0[0] - ptr0[2] * ptr2[0]) / det;
	dstarray[1][2] = (ptr0[2] * ptr1[0] - ptr0[0] * ptr1[2]) / det;
	dstarray[2][0] = (ptr1[0] * ptr2[1] - ptr2[0] * ptr1[1]) / det;
	dstarray[2][1] = (ptr2[0] * ptr0[1] - ptr0[0] * ptr2[1]) / det;
	dstarray[2][2] = (ptr0[0] * ptr1[1] - ptr0[1] * ptr1[0]) / det;
	return 1;
}

void mat_mul(double src1[3],double src2[3][3],double dst[3])
{
	dst[0] = src2[0][0] * src1[0] + src2[0][1] * src1[1] + src2[0][2] * src1[2];
	dst[1] = src2[1][0] * src1[0] + src2[1][1] * src1[1] + src2[1][2] * src1[2];
	dst[2] = src2[2][0] * src1[0] + src2[2][1] * src1[1] + src2[2][2] * src1[2];
}

double **fir_der(Mat srcImage,Mat srcImageup,Mat srcImagedown)  //一阶微分
{
	int col, row, i, j,k;
	uchar *ptr1,*ptr2,*ptr3;
	double **res,*ptr;
	col = srcImage.cols;
	row = srcImage.rows;
	res = (double**)malloc(sizeof(double *) * 3);
	for (i = 0; i < 3; i++)
	{
		res[i] = (double*)malloc(sizeof(double)*row*col);
	}
	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < row*col; j++) res[i][j] = 0;
	}
	//求x方向的导数
	for (i = 0; i < row; i++)
	{
		
		ptr1 = srcImage.ptr(i);
		ptr = res[0]+i*col;
		for (j = 1; j < col - 1; j++)
		{
			*(ptr+j) = ((double)ptr1[j + 1] - (double)ptr1[j - 1])/2.0;
		}
	}

	//求y方向的导数
	for (i = 1; i < row-1; i++)
	{
		ptr2 = srcImage.ptr(i - 1);
		ptr3 = srcImage.ptr(i + 1);
		ptr = res[1] + i*col;
		for (j = 0; j < col; j++)
		{
			*(ptr + j) = ((double)ptr3[j] - (double)ptr2[j]) / 2.0;
		}
	}

	//求sigma方向的导数
	for (i = 0; i < row; i++)
	{
		ptr2 = srcImageup.ptr(i);
		ptr3 = srcImagedown.ptr(i);
		ptr = res[2] + i*col;
		for (j = 0; j < col; j++)
		{
			*(ptr + j) = ((double)ptr2[j] - (double)ptr3[j]) / 2.0;
		}
	}

	return res;
}

int get_int(double a)
{
	int b=a;
	if ((a-b)>0.5) return b+1;
	else return b;
}

double **sec_der(Mat srcImage,Mat srcImageup,Mat srcImagedown)  //二阶微分
{
	int col, row, i, j, k;
	uchar *ptr1, *ptr2, *ptr3,*ptr4;
	double **res, *ptr;
	col = srcImage.cols;
	row = srcImage.rows;
	res = (double**)malloc(sizeof(double *) * 6);
	for (i = 0; i < 6; i++)
	{
		res[i] = (double*)malloc(sizeof(double)*row*col);
	}
	for (i = 0; i < 6; i++)
	{
		for (j = 0; j < row*col; j++) res[i][j] = 0;
	}

	//求x方向的二阶导数
	for (i = 0; i < row; i++)
	{
		ptr1 = srcImage.ptr(i);
		ptr = res[0] + i*col;
		for (j = 1; j < col - 1; j++)
		{
			*(ptr + j) = (double)ptr1[j + 1] + (double)ptr1[j - 1]-2.0*(double)ptr1[j];
		}
	}

	//求y方向的二阶导数
	for (i = 1; i < row - 1; i++)
	{
		ptr1 = srcImage.ptr(i);
		ptr2 = srcImage.ptr(i - 1);
		ptr3 = srcImage.ptr(i + 1);
		ptr = res[1] + i*col;
		for (j = 0; j < col; j++)
		{
			*(ptr + j) = (double)ptr3[j] + (double)ptr2[j]-2.0*(double)ptr1[j];
		}
	}

	//求sigma方向的二阶导数
	for (i = 0; i < row; i++)
	{
		ptr1 = srcImage.ptr(i);
		ptr2 = srcImageup.ptr(i);
		ptr3 = srcImagedown.ptr(i);
		ptr = res[2] + i*col;
		for (j = 0; j < col; j++)
		{
			*(ptr + j) = (double)ptr3[j] + (double)ptr2[j] - 2.0*(double)ptr1[j];
		}
	}

	//求xy方向的二阶导数
	for (i = 1; i < row - 1; i++)
	{
		ptr1 = srcImage.ptr(i);
		ptr2 = srcImage.ptr(i - 1);
		ptr3 = srcImage.ptr(i + 1);
		ptr = res[1] + i*col;
		for (j = 1; j < col-1; j++)
		{
			*(ptr + j) = ((double)ptr3[j + 1] + (double)ptr2[j - 1] - ((double)ptr3[j - 1] + (double)ptr2[j + 1]))/4;
		}
	}

	//求y,sigma方向的二阶导数
	for (i = 1; i < row - 1; i++)
	{
		ptr1 = srcImageup.ptr(i + 1);
		ptr2 = srcImageup.ptr(i - 1);
		ptr3 = srcImagedown.ptr(i + 1);
		ptr4 = srcImagedown.ptr(i - 1);
		ptr = res[1] + i*col;
		for (j = 0; j < col; j++)
		{
			*(ptr + j) = ((double)ptr4[j] + (double)ptr1[j] - ((double)ptr3[j] + (double)ptr2[j])) / 4;
		}
	}

	//求x,sigma方向的二阶导数
	for (i = 0; i < row; i++)
	{
		ptr2 = srcImageup.ptr(i);
		ptr3 = srcImagedown.ptr(i);
		ptr = res[1] + i*col;
		for (j = 1; j < col - 1; j++)
		{
			*(ptr + j) = ((double)ptr2[j-1] + (double)ptr3[j+1] - ((double)ptr3[j-1] + (double)ptr2[j+1])) / 4;
		}

	}
	return res;
}

void eliminate_point(Mat dog[3][5],Mat dstmat[3][3])			//这一部分先不要添加到main()里面。没有关键性影响。
{
	double Hessian[3][3];
	double Hessian_inv[3][3];
	double Dx[3];
	double dev_x[3];
	double **fir, **sec;
	double det, trace;
	int col, row, i, j, k, m,n;
	int x, y, sigma;
	uchar *ptr1,*ptr2,*ptr3,*ptr4,*ptr5,*ptr6,*ptr7,*ptr8,*ptr9,*ptr10,*temp;
	for (k = 0; k < 6; k++)
	{
		for (m = 0; m < 3; m++)
		{
			col = dog[m][0].cols;
			row = dog[m][0].rows;
			for (n = 1; n < 4; n++)
			{
				fir = fir_der(dog[m][n], dog[m][n + 1], dog[m][n - 1]);
				sec = sec_der(dog[m][n], dog[m][n + 1], dog[m][n - 1]);
				for (i = 0; i < row; i++)
				{
					ptr1 = dog[m][n].ptr(i);
					ptr2 = dstmat[m][n - 1].ptr(i);
					if (i == 0)
					{
						ptr9 = NULL;
						ptr10 = dstmat[m][n - 1].ptr(i + 1);
					}
					else if (i == (row - 1))
					{
						ptr9 = dstmat[m][n - 1].ptr(i - 1);
						ptr10 = NULL;
					}
					else
					{
						ptr9 = dstmat[m][n - 1].ptr(i - 1);
						ptr10 = dstmat[m][n - 1].ptr(i + 1);
					}
					if (n > 1)
					{
						ptr3 = dstmat[m][n - 2].ptr(i);
						//
						if (i == 0)
						{
							ptr5 = NULL;
							ptr6 = dstmat[m][n - 2].ptr(i + 1);
						}
						else if (i == (row - 1))
						{
							ptr5 = dstmat[m][n - 2].ptr(i - 1);
							ptr6 = NULL;
						}
						else
						{
							ptr5 = dstmat[m][n - 2].ptr(i - 1);
							ptr6 = dstmat[m][n - 2].ptr(i + 1);
						}
						//
					}
					else
					{
						ptr3 = NULL;
						ptr5 = NULL;
						ptr6 = NULL;
					}
					if (n < 3)
					{
						ptr4 = dstmat[m][n].ptr(i);
						if (i == 0)
						{
							ptr7 = NULL;
							ptr8 = dstmat[m][n].ptr(i + 1);
						}
						else if (i == (row - 1))
						{
							ptr7 = dstmat[m][n].ptr(i - 1);
							ptr8 = NULL;
						}
						else
						{
							ptr7 = dstmat[m][n].ptr(i - 1);
							ptr8 = dstmat[m][n].ptr(i + 1);
						}
					}
					else 
					{
						ptr4 = NULL;
						ptr7 = NULL;
						ptr8 = NULL;
					}
					for (j = 0; j < col; j++)
					{
						if ((ptr2[j] == 255) || (ptr2[j] == 0))
						{

							Dx[0] = fir[0][i*col + j];
							Dx[1] = fir[1][i*col + j];
							Dx[2] = fir[2][i*col + j];
							Hessian[0][0] = sec[0][i*col + j];
							Hessian[0][1] = sec[3][i*col + j];
							Hessian[0][2] = sec[5][i*col + j];
							Hessian[1][0] = sec[3][i*col + j];
							Hessian[1][1] = sec[1][i*col + j];
							Hessian[1][2] = sec[4][i*col + j];
							Hessian[2][0] = sec[5][i*col + j];
							Hessian[2][1] = sec[4][i*col + j];
							Hessian[2][2] = sec[2][i*col + j];
							if (mat_invert(Hessian, Hessian_inv))
							{
								mat_mul(Dx, Hessian_inv, dev_x);
								trace = Hessian[0][0] + Hessian[1][1];
								det = Hessian[0][0] * Hessian[1][1] - Hessian[0][1] * Hessian[0][1];
								//去除D(x）与二维HESSIAN矩阵不符合的点
								if (abs((double)ptr2[j] + 0.5*(Dx[0] * dev_x[0] + Dx[1] * dev_x[1] + Dx[2] * dev_x[2]) > 0.04) || (abs(trace*trace / det) >= 1.21))
								{
									ptr2[j] = 100;
									continue;
								}

								else
								{
									if ((abs(dev_x[0]) <= 0.5) && (abs(dev_x[1]) <= 0.5) && (abs(dev_x[2]) <= 0.5)) continue;
									else
									{
										if (dev_x[0] > 0.5) x = 1;
										else if (dev_x[0] < -0.5) x = -1;
										else x = 0;
										if (dev_x[1] > 0.5) y = 1;
										else if (dev_x[1] < -0.5) y = -1;
										else y = 0;
										if (dev_x[2] > 0.5) sigma = 1;
										else if (dev_x[2] < -0.5) sigma = -1;
										else sigma = 0;
										if ((sigma == 1) && (y == 0)) temp = ptr4;
										else if ((sigma == -1) && (y == 0)) temp = ptr3;
										else if ((sigma == 0) && (y == 0)) temp = ptr2;
										else if ((sigma == 1) && (y == 1)) temp = ptr8;
										else if ((sigma == -1) && (y == 1)) temp = ptr6;
										else if ((sigma == 0) && (y == 1)) temp = ptr10;
										else if ((sigma == 1) && (y == -1)) temp = ptr7;
										else if ((sigma == -1) && (y == -1)) temp = ptr5;
										else if ((sigma == 0) && (y == -1)) temp = ptr9;
										if ((temp != NULL) && ((j + x) >= 0) && ((j + x) <= col) && (k != 5))
										{
											temp[j + x] = ptr2[j];
											ptr2[j] = 100;
										}
										else
										{
											ptr2[j] = 100;
											break;
										}
									}
								}
							}
							else ptr2[j] = 100;
						}
						else continue;
					}
				}
			}
		}
	}
}//第四部就是用第二个公式调整特征点的位置，如果越界或者迭代了多次都不收敛就舍弃特征点，有可能在这部要对应回原图

int add_point(Mat dstmat[3][3], feature feat[10000],Mat Gaussian[3][6])			//返回值是结构体数组长度
{
	int i = 0,j = 0,k=0,m=0,n=0;
	int col, row;
	uchar *ptr1,*ptr2,*ptr3,*ptr4;
	for (i = 0; i < 3; i++)
		for (j = 0; j < 3; j++)
		{
			col = dstmat[i][j].cols;
			row = dstmat[i][j].rows;
			for (m = 1; m < row-1; m++)
			{
				ptr1 = dstmat[i][j].ptr(m);
				ptr2 = Gaussian[i][j+1].ptr(m);
				ptr3 = Gaussian[i][j + 1].ptr(m-1);
				ptr4 = Gaussian[i][j + 1].ptr(m+1);
				for (n = 0; n < col; n++)
				{
					if (ptr1[n] == 255 || ptr1[n] == 0)
					{
						feat[k].x=m;
						feat[k].y = n;
						feat[k].sigma = j + 1;
						feat[k].octave = i;
						feat[k].m = sqrt((ptr4[n] - ptr3[n])*(ptr4[n] - ptr3[n]) + (ptr2[n + 1] - ptr2[n - 1])*(ptr2[n + 1] - ptr2[n - 1]));
						feat[k].ori = atan2((ptr2[n + 1] - ptr2[n - 1]) , (ptr4[n] - ptr3[n]));
						feat[k].theta = 0;
						k++;
					}
				}
			}
		}
	return k;
}   //

void gethist(double bin[36],feature feat[10000],Mat Gaussian[3][6],int length,int radious,double sig)
{
	uchar *ptr1, *ptr2, *ptr3,*ptr4;
	double ori,m,temp1;
	int i,j,k;
	int n;
	for (i = 0; i < length; i++)
	{
//		ptr2 = Gaussian[feat[i].octave][feat[i].sigma].ptr(feat[i].x);
		if ((feat[i].x - radious > 0) && (feat[i].x + radious) < (Gaussian[feat[i].octave][feat[i].sigma].rows-1) && (feat[i].y - radious > 0) && (feat[i].y + radious < (Gaussian[feat[i].octave][feat[i].sigma].cols)-1))
		{
			for (j = 0; j < 2 * radious; j++)
			{
				ptr2 = (uchar *)Gaussian[feat[i].octave][feat[i].sigma].ptr(feat[i].x + j - radious);
				ptr3 = (uchar *)Gaussian[feat[i].octave][feat[i].sigma].ptr(feat[i].x + j - radious-1);
				ptr4 = (uchar *)Gaussian[feat[i].octave][feat[i].sigma].ptr(feat[i].x + j - radious+1);
				for (k = 0; k < 2 * radious; k++)
				{
					n = feat[i].y + k - radious;
					ori = atan2((ptr2[n + 1] - ptr2[n - 1]) , (ptr4[n] - ptr3[n]));
					m=sqrt((ptr4[n] - ptr3[n])*(ptr4[n] - ptr3[n]) + (ptr2[n + 1] - ptr2[n - 1])*(ptr2[n + 1] - ptr2[n - 1]));
					if (ori >= -PI && ori < -17*PI / 18) bin[0] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -16 * PI / 18) bin[1] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -15 * PI / 18) bin[2] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -14 * PI / 18) bin[3] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -13 * PI / 18) bin[4] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -12 * PI / 18) bin[5] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -11 * PI / 18) bin[6] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -10 * PI / 18) bin[7] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -9 * PI / 18) bin[8] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -8 * PI / 18) bin[9] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -7 * PI / 18) bin[10] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -6 * PI / 18) bin[11] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -5 * PI / 18) bin[12] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -4 * PI / 18) bin[13] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -3 * PI / 18) bin[14] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -2 * PI / 18) bin[15] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < -1 * PI / 18) bin[16] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 0) bin[17] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < PI / 18) bin[18] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 2 * PI / 18) bin[19] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 3 * PI / 18) bin[20] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 4 * PI / 18) bin[21] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 5 * PI / 18) bin[22] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 6 * PI / 18) bin[23] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 7 * PI / 18) bin[24] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 8 * PI / 18) bin[25] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 9 * PI / 18) bin[26] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));					
					else if (ori < 10 * PI / 18) bin[27] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 11 * PI / 18) bin[28] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 12 * PI / 18) bin[29] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 13 * PI / 18) bin[30] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 14 * PI / 18) bin[31] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 15 * PI / 18) bin[32] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 16 * PI / 18) bin[33] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < 17 * PI / 18) bin[34] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else if (ori < PI) bin[35] += m*pow(E, -((k - radious)*(k - radious) + (j - radious)*(j - radious)) / (2 * sig*sig));
					else printf("ori error1!\n");
					temp1 = bin[0];
					for (i = 0; i < 36; i++) if (bin[i]>temp1) temp1 = bin[i];
					feat[i].theta = temp1;
				}
			}
		}
	}
}

void smoothhist(double bin[36])
{
	int i;
	for (i = 1; i < 35; i++)
	{
		bin[i] = 0.25*bin[i - 1] + 0.5*bin[i] + 0.25*bin[i + 1];
	}
}

void reproduce(double bin[36], feature feat[10000], int &length,feature temp)
{
	int i;
	double temp1=0,temp2=0,tempi=0;
	for (i = 0; i < 36; i++) if (bin[i]>temp1) temp1 = bin[i];
	for (i = 0; i < 36; i++) 
		if ((abs(bin[i] - temp1)>1e-5) && (bin[i]>temp2))
		{
			temp2 = bin[i];
			tempi = i;
			
		}
	if (temp2 > temp1*0.8)
	{
		length++;
		feat[length]= temp;
		feat[length].m = temp2;
		feat[length].ori = (i-18+0.5)*PI/18;
		feat[length].theta = (i - 18 + 0.5)*PI / 18;
	}
}

void get_desc(feature feat[10000],int length,Mat Gaussian[3][6],int radious)
{
	int i, j,k,n;
	int tx, ty, tt;
	double x1 = 0, y1 = 0;
	double w,ori,m;
	double dx, dy, dt;
	uchar *ptr2, *ptr3, *ptr4;
	double x2 = 0, y2 = 0;
	for (k = 0; k < length; k++)
	{
		for (i = -radious; i <= radious; i++)
		{
			if ((feat[k].x - radious > 0) && (feat[k].x + radious) < (Gaussian[feat[k].octave][feat[k].sigma].rows - 1) && (feat[k].y - radious > 0) && (feat[k].y + radious < (Gaussian[feat[k].octave][feat[k].sigma].cols) - 1))
			{
				ptr2 = (uchar *)Gaussian[feat[k].octave][feat[k].sigma].ptr(feat[k].x + i);
				ptr3 = (uchar *)Gaussian[feat[k].octave][feat[k].sigma].ptr(feat[k].x + i - 1);
				ptr4 = (uchar *)Gaussian[feat[k].octave][feat[k].sigma].ptr(feat[k].x + i + 1);
				for (j = -radious; j <= radious; j++)
				{
					//x1 = Gaussian[feat[k].octave][feat[k].sigma].ptr(feat[k].x + (int)x1)[feat[k].y + (int)y1]
					x1 = i*cos(feat[k].theta) - j*sin(feat[k].theta);
					y1 = i*sin(feat[k].theta) + j*cos(feat[k].theta);
					n = feat[k].y + j;
					ori = atan2((ptr2[n + 1] - ptr2[n - 1]), (ptr4[n] - ptr3[n]));
					ori -= feat[k].theta;
					if (ori < -PI) ori += 2 * PI;
					if (ori > PI) ori -= 2 * PI;
					m = sqrt((ptr4[n] - ptr3[n])*(ptr4[n] - ptr3[n]) + (ptr2[n + 1] - ptr2[n - 1])*(ptr2[n + 1] - ptr2[n - 1]));
					x2 = (x1 / (3 * (feat[k].octave + 1)) + 2);
					y2 = (y1 / (3 * (feat[k].octave + 1)) + 2);
					tx = (int)x2;
					ty = (int)y2;
					dx = x2 - tx;
					dy = y2 - ty;
					if (-PI <= ori&&ori > -3 * PI / 4) dt = ori + PI;
					else if (-3 * PI / 4 <= ori&&ori > -PI / 2) dt = ori + PI * 3 / 4;
					else if (-PI / 2 <= ori&&ori > -PI / 4) dt = ori + PI/2;
					else if (-PI / 4 <= ori&&ori >0) dt = ori + PI/ 4;
					else if (0<= ori&&ori > PI / 4) dt = ori;
					else if (PI / 4 <= ori&&ori >PI/2) dt = ori - PI / 4;
					else if (PI / 2 <= ori&&ori > 3*PI / 4) dt = ori - PI / 2;
					else if (3 * PI / 4 <= ori&&ori > PI) dt = ori - PI * 3 / 4;
					else printf("error2!");
					w = pow(E, -(x1*x1 + y1*y1) / 8);
					if (-PI <= ori&&ori > -3 * PI / 4)
					{
						feat[k].desc[tx][ty][0] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][1] += w*m*dx*dy*(1-dt);
						feat[k].desc[tx][ty+1][0] += w*m*dx*(1-dy)*dt;
						feat[k].desc[tx][ty+1][1] += w*m*dx*(1-dy)*(1 - dt);
						feat[k].desc[tx][ty][0] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][1] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx+1][ty + 1][0] += w*m*(1-dx)*(1 - dy)*dt;
						feat[k].desc[tx+1][ty + 1][1] += w*m*(1-dx)*(1 - dy)*(1 - dt);
					}
					else if (-3 * PI / 4 <= ori&&ori > -PI / 2) {
						feat[k].desc[tx][ty][1] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][2] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx][ty + 1][1] += w*m*dx*(1 - dy)*dt;
						feat[k].desc[tx][ty + 1][2] += w*m*dx*(1 - dy)*(1 - dt);
						feat[k].desc[tx][ty][1] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][2] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx + 1][ty + 1][1] += w*m*(1 - dx)*(1 - dy)*dt;
						feat[k].desc[tx + 1][ty + 1][2] += w*m*(1 - dx)*(1 - dy)*(1 - dt);
					}
					else if (-PI / 2 <= ori&&ori > -PI / 4) {
						feat[k].desc[tx][ty][2] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][3] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx][ty + 1][2] += w*m*dx*(1 - dy)*dt;
						feat[k].desc[tx][ty + 1][3] += w*m*dx*(1 - dy)*(1 - dt);
						feat[k].desc[tx][ty][2] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][3] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx + 1][ty + 1][2] += w*m*(1 - dx)*(1 - dy)*dt;
						feat[k].desc[tx + 1][ty + 1][3] += w*m*(1 - dx)*(1 - dy)*(1 - dt);
					}
					else if (-PI / 4 <= ori&&ori >0) {
						feat[k].desc[tx][ty][3] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][4] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx][ty + 1][3] += w*m*dx*(1 - dy)*dt;
						feat[k].desc[tx][ty + 1][4] += w*m*dx*(1 - dy)*(1 - dt);
						feat[k].desc[tx][ty][3] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][4] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx + 1][ty + 1][3] += w*m*(1 - dx)*(1 - dy)*dt;
						feat[k].desc[tx + 1][ty + 1][4] += w*m*(1 - dx)*(1 - dy)*(1 - dt);
					}
					else if (0 <= ori&&ori > PI / 4) {
						feat[k].desc[tx][ty][4] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][5] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx][ty + 1][4] += w*m*dx*(1 - dy)*dt;
						feat[k].desc[tx][ty + 1][5] += w*m*dx*(1 - dy)*(1 - dt);
						feat[k].desc[tx][ty][4] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][5] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx + 1][ty + 1][4] += w*m*(1 - dx)*(1 - dy)*dt;
						feat[k].desc[tx + 1][ty + 1][5] += w*m*(1 - dx)*(1 - dy)*(1 - dt);
					}
					else if (PI / 4 <= ori&&ori >PI / 2) {
						feat[k].desc[tx][ty][5] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][6] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx][ty + 1][5] += w*m*dx*(1 - dy)*dt;
						feat[k].desc[tx][ty + 1][6] += w*m*dx*(1 - dy)*(1 - dt);
						feat[k].desc[tx][ty][5] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][6] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx + 1][ty + 1][5] += w*m*(1 - dx)*(1 - dy)*dt;
						feat[k].desc[tx + 1][ty + 1][6] += w*m*(1 - dx)*(1 - dy)*(1 - dt);
					}
					else if (PI / 2 <= ori&&ori > 3 * PI / 4) {
						feat[k].desc[tx][ty][6] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][7] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx][ty + 1][6] += w*m*dx*(1 - dy)*dt;
						feat[k].desc[tx][ty + 1][7] += w*m*dx*(1 - dy)*(1 - dt);
						feat[k].desc[tx][ty][6] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][7] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx + 1][ty + 1][6] += w*m*(1 - dx)*(1 - dy)*dt;
						feat[k].desc[tx + 1][ty + 1][7] += w*m*(1 - dx)*(1 - dy)*(1 - dt);
					}
					else if (3 * PI / 4 <= ori&&ori > PI) {
						feat[k].desc[tx][ty][7] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][0] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx][ty + 1][7] += w*m*dx*(1 - dy)*dt;
						feat[k].desc[tx][ty + 1][0] += w*m*dx*(1 - dy)*(1 - dt);
						feat[k].desc[tx][ty][7] += w*m*dx*dy*dt;
						feat[k].desc[tx][ty][0] += w*m*dx*dy*(1 - dt);
						feat[k].desc[tx + 1][ty + 1][7] += w*m*(1 - dx)*(1 - dy)*dt;
						feat[k].desc[tx + 1][ty + 1][0] += w*m*(1 - dx)*(1 - dy)*(1 - dt);
					}
					else printf("error3!");
					//线性插值
				}
			}
		}
	}
}

void standardize(feature feat[10000], int length)
{
	int i, j,k,m,n;
	double sum;
	for (i = 0; i < length; i++)
	{
		sum = 0;
		for (m = 0; m < 4; m++)
		{
			for (n = 0; n < 4; n++)
			{
				for (k = 0; k < 8; k++) sum += feat[i].desc[m][n][k];
			}
		}
		sum = sqrt(sum);
		for (m = 0; m < 4; m++)
		{
			for (n = 0; n < 4; n++)
			{
				for (k = 0; k < 8; k++) feat[i].desc[m][n][k] = feat[i].desc[m][n][k]/sum;
			}
		}
		for (m = 0; m < 4; m++)
		{
			for (n = 0; n < 4; n++)
			{
				for (k = 0; k < 8; k++) if (feat[i].desc[m][n][k]>0.2) feat[i].desc[m][n][k]=0.2;
			}
		}
		sum = 0;
		for (m = 0; m < 4; m++)
		{
			for (n = 0; n < 4; n++)
			{
				for (k = 0; k < 8; k++) sum += feat[i].desc[m][n][k];
			}
		}
		sum = sqrt(sum);
		for (m = 0; m < 4; m++)
		{
			for (n = 0; n < 4; n++)
			{
				for (k = 0; k < 8; k++) feat[i].desc[m][n][k] = feat[i].desc[m][n][k] / sum;
			}
		}
	}
}

void match(feature feat[10000],int length,Mat Gaussian[3][6])
{
	int i,j,k,l,m,n;
	double temp[10000][10000] = { 65535 };
	double temp1,temp2;
	int tempi, tempj;
	for (i = 0; i < length; i++)
	{
		for (j = i; j < length; j++)
		{
			for (m = 0; m < 4; m++)
				for (n = 0; n < 4; n++)
					for (l = 0; l < 8; l++)
					{
						temp[i][j]+= (feat[i].desc[m][n][l] - feat[j].desc[m][n][l])*(feat[i].desc[m][n][l] - feat[j].desc[m][n][l]);
					}
		}
	}
	for (i = 0; i < length; i++)
	{
		temp1 = 65535;
		temp2 = 65535;
		for (j = i; j < length; j++)
		{
			if (temp1>temp[i][j])
			{
				temp1 = temp[i][j];
				tempi = i;
				tempj = j;
			}
		}
		for (j = i; j < length; j++)
		{
			if ((j!=tempj)&&temp2>temp[i][j]) temp2 = temp[i][j];
		}
	}
	if ((temp1 / temp2) < 0.5&&feat[tempi].octave == feat[tempj].octave) line(Gaussian[feat[tempi].octave][0], Point(feat[tempj].y, feat[tempj].x), Point(feat[tempi].y, feat[tempi].x), Scalar(0), 2, 8);
}

void main()
{
	Mat srcImage = imread("D:\\51.jpg",0);
	Mat Gaussian[3][6];
	int length;
	int radious;
	double bin[36] = { 0 };
	feature feat[10000];
	GaussianBlur(srcImage, srcImage, Size(0, 0), 0.5, 0.5);
	Mat dog[3][5],dstmat[3][3];
	Dog(srcImage, dog,Gaussian);
	imshow("第一尺度第一层", Gaussian[0][1]);
	imshow("第一尺度第二层", Gaussian[0][0]);
	imshow("第一尺度第三层", Gaussian[0][2]);
	imshow("第二尺度第一层", Gaussian[1][1]);
	imshow("第二尺度第二层", Gaussian[1][0]);
	imshow("第二尺度第三层", Gaussian[1][2]);
	imshow("第三尺度第一层", Gaussian[2][1]);
	imshow("第三尺度第二层", Gaussian[2][0]);
	imshow("第三尺度第三层", Gaussian[2][2]);
//	imshow("1", dog[1][0]);
///	getmax_min(dog, dstmat);
//	eliminate_point(dog, dstmat);
//	length=add_point(dstmat, feat, Gaussian);
//	gethist(bin,feat,Gaussian,length,)
//	imshow("1", dstmat[0][2]);
//	imshow("2", dstmat[1][2]);
//	imshow("3", dstmat[2][2]);
	waitKey(0);
}



